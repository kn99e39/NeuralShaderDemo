"""Quality-vs-time trace of the timed RNA rebuild (untimed evaluation).

    windows/run.ps1 ../neural_recompute_cost/nrc_eval_rna.py --run <run dir>

Renders every RNA snapshot plus the official top-k / last checkpoints at T3
with the unchanged rna_infer.py exactly as worklog 22 rendered the RNA refit
(render_rna_states.py --part refit: mode current, the historical T3 feature
buffers, light model area-sampled, training light intensity 5), and scores
each with the unchanged historical rule cross_backbone_eval.refit_block
against the historical frozen T0 RNA render and the shared common-light
references.  The historical selection (validation-best val_psnr) is found
with the unchanged render_rna_states.best_checkpoint rule.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "8dna_deformation_replication"))

import nrc_records as R  # noqa: E402
from nrc_wsl import RNA_ROOT, run_detached, wsl_path  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    run = Path(args.run).resolve()
    nrc = json.loads((HERE / "protocol" / "nrc_t3_recompute_v1.json").read_text(encoding="utf-8"))
    import numpy as np

    import ednalib as L

    git = R.git_state(L.ROOT)
    R.require_clean(git, args.allow_dirty)
    L.init_upstream()
    from cross_backbone_eval import refit_block
    from render_rna_states import best_checkpoint
    from teaset_frozen_eval import load_rois

    cb = json.loads(open(L.EXPERIMENT / "protocol/teaset_cross_backbone_locked.json", encoding="utf-8").read())
    inf = cb["rna_inference"]
    tr = nrc["tracks"]["rna_common_light"]["training"]
    name = tr["run_name"] + ("-smoke" if args.smoke else "")
    ck_root = run / "rna" / "ckpt" / name
    official = sorted(ck_root.rglob("*.ckpt"))
    snaps = sorted((run / "rna" / "snapshots").glob("epoch_*.ckpt"))
    # the historical selection rule, unchanged (it resolves names under results/8dna_replication/rna_teaset/ckpt)
    best, best_psnr = best_checkpoint(os.path.relpath(ck_root, L.RESULTS / "rna_teaset" / "ckpt"))
    best = Path(best).resolve()
    out_dir = run / "eval" / "rna"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_h5 = run / "rna" / "datasets" / "teaset_T3_train.h5"
    feats = L.RESULTS / cb["rna_features_dir"] / "T3.npz"
    jobs = []
    for p in snaps + official:
        tag = ("snap_" + p.stem) if p in snaps else ("official_" + re.sub(r"[^A-Za-z0-9_.=-]", "_", p.stem))
        jobs.append({"checkpoint": wsl_path(p), "out": wsl_path(out_dir / tag), "tag": tag, "local": p})
    spec = {"common": ["--train-h5", wsl_path(train_h5), "--features", wsl_path(feats), "--mode", "current",
                       "--training-light-intensity", str(inf["training_light_intensity"]),
                       "--light-model", inf.get("light_model", "area-sampled")],
            "jobs": [{"checkpoint": j["checkpoint"], "out": j["out"]} for j in jobs]}
    jf = out_dir / "jobs.json"
    jf.write_text(json.dumps(spec, indent=1), encoding="utf-8")
    timing_f = out_dir / "infer_timing.json"
    rc = run_detached("eval_rna_infer", run / "logs",
                      [".venv/bin/python", wsl_path(HERE / "wsl" / "nrc_rna_eval_wsl.py"), "--jobs", wsl_path(jf),
                       "--out", wsl_path(timing_f)], RNA_ROOT)
    if rc != 0:
        raise SystemExit(f"WSL rna_infer batch failed (rc={rc}); see {run / 'logs'}")
    calls = {Path(t["out"]).name: t for t in json.loads(timing_f.read_text())["calls"]}

    states = cb["states"]
    fdir = L.RESULTS / cb["frozen_output"]
    gt = {s: (L.load_exr(fdir / s / "gt_A.exr"), L.load_exr(fdir / s / "gt_B.exr")) for s in ("T0", "T3")}
    rois = load_rois(L.RESULTS / cb["gt_output"], ("T0", "T3"))
    t0_model = L.RESULTS / cb["rna_output"] / "T0_canonical.npy"
    n_t0 = np.load(t0_model).astype(np.float32)
    rows = []
    for j in jobs:
        img = np.load(str(out_dir / j["tag"]) + ".npy").astype(np.float32)
        blk = refit_block(gt, rois, n_t0, img)
        p = j["local"]
        c = calls[j["tag"]]
        m = re.search(r"epoch=(\d+)", p.name) or re.search(r"epoch_(\d+)", p.name)
        rows.append({"tag": j["tag"], "checkpoint": L.rel(p), "sha256": L.sha256(p), "kind": "snapshot" if p in snaps else "official",
                     "epoch": c["ckpt_epoch_field"] if m is None else int(m.group(1)),
                     "ckpt_epoch_field": c["ckpt_epoch_field"], "global_step": c["global_step"], "weights_digest": c["weights_digest"],
                     "is_validation_best": p == best, "render": L.rel(str(out_dir / j["tag"]) + ".npy"),
                     "rna_infer_call_s": c["seconds"], **blk})
        print(j["tag"], f"ratio {blk['error_ratio']:.3f} gain {blk['gain']:.3f} recovers {blk['recovers']}", flush=True)
    # snapshot vs official checkpoint of the same epoch must hold identical weights
    checks = []
    for r in rows:
        if r["kind"] != "official":
            continue
        twin = next((s for s in rows if s["kind"] == "snapshot" and s["ckpt_epoch_field"] == r["ckpt_epoch_field"]), None)
        if twin is not None:
            checks.append({"official": r["tag"], "snapshot": twin["tag"], "identical": r["weights_digest"] == twin["weights_digest"]})
    rec = {"environment": L.environment_record(), "git": git, "smoke": args.smoke, "features": L.rel(feats),
           "features_sha256": L.sha256(feats), "t0_model_render": L.rel(t0_model), "train_h5": L.rel(train_h5),
           "validation_best": {"checkpoint": L.rel(best), "val_psnr_db": best_psnr},
           "historical": nrc["tracks"]["rna_common_light"]["evaluation"]["historical_values"],
           "snapshot_official_identity": checks, "rows": rows}
    if any(not c["identical"] for c in checks):
        raise SystemExit(f"snapshot/official checkpoint mismatch: {checks}")
    R.write_json(run / "eval" / "rna_eval.json", rec)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
