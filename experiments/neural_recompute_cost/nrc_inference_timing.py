"""Steady-state inference cost of the recomputed models (untimed w.r.t. the rebuild latency).

    windows/run.ps1 ../neural_recompute_cost/nrc_inference_timing.py --run <run dir>

Kept separate from update/rebuild latency (batch section 9).

8DNA: the final export (last.ckpt) at T3 through the existing evaluation path
(upstream neuralpath, mode upstream, 512^2); 1 warm-up and 5 timed 1-spp
renders per regime.  The historical-spp render times come from the
evaluation renders themselves (8dna_eval.json).

RNA: (a) the inference inputs -- rna_bridge.features (G-buffer at 4x4
sub-pixel samples + 16 area-light samples each, current-geometry visibility),
written to this run's directory so the historical buffers are untouched; the
Mitsuba part and the compressed .npz write are timed separately; (b) the
network evaluation of the validation-best checkpoint on the historical T3
buffers, in WSL (nrc_rna_infer_timing_wsl.py, checked equal to rna_infer.py's
own output).
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "8dna_deformation_replication"))

import nrc_records as R  # noqa: E402
from nrc_wsl import RNA_ROOT, run_detached, wsl_path  # noqa: E402

REGIMES = {"w21_envmap": "protocol/teaset_frozen_locked.json", "common_light": "protocol/teaset_cross_backbone_locked.json"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    run = Path(args.run).resolve()
    nrc = json.loads((HERE / "protocol" / "nrc_t3_recompute_v1.json").read_text(encoding="utf-8"))
    import numpy as np

    import ednalib as L
    import teaset_parts as T

    git = R.git_state(L.ROOT)
    R.require_clean(git, args.allow_dirty)
    mi, dr = L.init_upstream()
    import torch
    from models.integrator import load_asset

    import rna_bridge as B

    rec = {"environment": L.environment_record(), "git": git, "smoke": args.smoke, "8dna": {}, "rna": {}}
    t8 = nrc["tracks"]["8dna_envmap_primary"]["training"]
    export = run / "8dna" / f"{t8['experiment_name']}_released_layout.ckpt"
    ev8 = json.loads((run / "eval" / "8dna_eval.json").read_text(encoding="utf-8"))
    for regime, pfile in REGIMES.items():
        proto = json.loads(open(L.EXPERIMENT / pfile, encoding="utf-8").read())
        scene = mi.load_dict(T.scene_dict(proto["res"], proto["states"]["T3"], proto.get("lighting")))
        integ = L.load_neural_integrator(proto["asset"])
        torch.cuda.synchronize()
        t = time.perf_counter()
        integ.asset_models = [load_asset("8dna", str(export))]
        integ.is_prepared = False
        torch.cuda.synchronize()
        t_load = time.perf_counter() - t
        L.render_chunked(scene, integ, 1, 1, 0)  # warm-up (kernel compilation)
        times = [L.render_chunked(scene, integ, 1, 1, 1 + i)[1] for i in range(args.repeats)]
        rows = ev8["regimes"][regime]["rows"]
        final = max(rows, key=lambda r: r["epoch"])
        rec["8dna"][regime] = {"checkpoint": L.rel(export), "res": proto["res"], "load_asset_s": t_load,
                               "spp1_render_s": times, "spp1_render_s_median": float(np.median(times)),
                               "eval_spp": final["spp"], "eval_render_s": final["render_s"],
                               "eval_render_s_per_spp": final["render_s"] / final["spp"],
                               "frame_budget_spp1": R.frame_budget_ratios(float(np.median(times))),
                               "note": "upstream neuralpath, 512^2, mode upstream; 1-spp time only, no quality claim at 1 spp"}
        print(regime, "1 spp", [f"{x * 1000:.0f} ms" for x in times], flush=True)

    # --- RNA (a): inference inputs (G-buffer + area-light samples) --------------
    cb = json.loads(open(L.EXPERIMENT / "protocol/teaset_cross_backbone_locked.json", encoding="utf-8").read())
    dp = dict(cb, rna_features_dir=os.path.relpath(run / "eval" / "rna_features", L.RESULTS).replace("\\", "/"))
    dpp = run / "eval" / "rna_features_protocol.json"
    R.write_json(dpp, dp)
    savez = np.savez_compressed
    spent = {"write_s": 0.0}

    @functools.wraps(savez)
    def timed_savez(*a, **k):
        t0 = time.perf_counter()
        try:
            return savez(*a, **k)
        finally:
            spent["write_s"] += time.perf_counter() - t0

    np.savez_compressed = timed_savez
    feat_times = []
    for i in range(2 if args.smoke else 3):  # first call includes kernel compilation
        spent["write_s"] = 0.0
        torch.cuda.synchronize()
        t = time.perf_counter()
        B.features(argparse.Namespace(protocol=os.path.relpath(dpp, L.EXPERIMENT), state="T3", light_seed="A"))
        torch.cuda.synchronize()
        tot = time.perf_counter() - t
        feat_times.append({"total_s": tot, "npz_write_s": spent["write_s"], "compute_s": tot - spent["write_s"]})
    np.savez_compressed = savez
    new_f = run / "eval" / "rna_features" / "T3.npz"
    hist_f = L.RESULTS / cb["rna_features_dir"] / "T3.npz"
    a, b = np.load(new_f), np.load(hist_f)
    same = {k: bool(np.array_equal(a[k], b[k])) for k in ("hit", "position", "normal", "light_dir", "light_weight", "light_vis")}
    rec["rna"]["features"] = {"calls": feat_times, "steady_compute_s": float(np.median([c["compute_s"] for c in feat_times[1:]])),
                              "steady_npz_write_s": float(np.median([c["npz_write_s"] for c in feat_times[1:]])),
                              "equal_to_historical_T3_buffers": same, "res": cb["res"],
                              "samples": "4x4 sub-pixel x 16 area-light samples"}
    print("rna features", feat_times, same, flush=True)

    # --- RNA (b): network evaluation of the validation-best model ---------------
    evr = json.loads((run / "eval" / "rna_eval.json").read_text(encoding="utf-8"))
    best = next(r for r in evr["rows"] if r["is_validation_best"])
    out = run / "eval" / "rna_network_timing.json"
    rc = run_detached("rna_infer_timing", run / "logs",
                      [".venv/bin/python", wsl_path(HERE / "wsl" / "nrc_rna_infer_timing_wsl.py"),
                       "--checkpoint", wsl_path(L.ROOT / best["checkpoint"]), "--features", wsl_path(hist_f),
                       "--reference", wsl_path(L.ROOT / best["render"]), "--out", wsl_path(out),
                       "--repeats", str(args.repeats)], RNA_ROOT)
    if rc != 0:
        raise SystemExit(f"RNA network timing failed or not equivalent (rc={rc}); see {out}")
    net = json.loads(out.read_text())
    net["frame_budget_network"] = R.frame_budget_ratios(net["network_s_median"])
    rec["rna"]["network"] = net
    rec["rna"]["rna_infer_call_s_validation_best"] = best["rna_infer_call_s"]
    R.write_json(run / "eval" / "inference_timing.json", rec)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
