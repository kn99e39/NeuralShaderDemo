"""Re-score the worklog-22 refit runs' surviving checkpoints through this batch's evaluation path.

    windows/run.ps1 ../neural_recompute_cost/nrc_historical_rescore.py --out <dir>

Reproducibility diagnostic (untimed; no training): the historical 8DNA T3_refit
run kept its top-8 val/loss checkpoints (epochs 20-29) and last.ckpt, the
historical RNA T3 run its top-10 val_psnr checkpoints and last.ckpt.  Each is
rendered at T3 exactly as nrc_eval_8dna.py / nrc_eval_rna.py render this
batch's checkpoints and scored with the unchanged refit_block.  This answers
(a) does the evaluation path reproduce worklog 22's numbers for the
historical selection, and (b) how far did the historical runs' own
neighbouring checkpoints scatter around the rule's thresholds.  Historical
files are only read; converted copies go to <out>.
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
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    import numpy as np

    import ednalib as L

    git = R.git_state(L.ROOT)
    R.require_clean(git, args.allow_dirty)
    mi, dr = L.init_upstream()
    from models.integrator import load_asset

    import nrc_eval_8dna as E8
    from cross_backbone_eval import refit_block
    from render_8dna_refits import as_released_layout
    from teaset_frozen_eval import load_rois

    rec = {"environment": L.environment_record(), "git": git, "8dna": {}, "rna": {}}

    # --- 8DNA ------------------------------------------------------------------
    hist = L.RESULTS / "refit" / "T3_refit"
    ckpts = sorted(hist.glob("epoch=*.ckpt"), key=lambda p: int(re.search(r"epoch=(\d+)", p.name).group(1)))
    conv = {}
    for p in ckpts:
        conv[p] = as_released_layout(p, out / "8dna" / (p.stem + "_released_layout.ckpt"))
    for regime, spec in E8.REGIMES.items():
        proto = json.loads(open(L.EXPERIMENT / spec["protocol"], encoding="utf-8").read())
        nr = proto["neural"]
        gt = {s: (L.load_exr(L.RESULTS / spec["gt"] / s / "gt_A.exr"), L.load_exr(L.RESULTS / spec["gt"] / s / "gt_B.exr"))
              for s in ("T0", "T3")}
        rois = load_rois(L.RESULTS / spec["rois"], ("T0", "T3"))
        n_t0 = L.load_exr(L.RESULTS / spec["t0_model"])
        scene = mi.load_dict(E8.T_scene(L, proto))
        rows = []
        for p in ckpts:
            integ = L.load_neural_integrator(proto["asset"])
            integ.asset_models = [load_asset("8dna", str(conv[p]))]
            integ.is_prepared = False
            img, t = L.render_chunked(scene, integ, nr["spp"], nr["chunk"], nr["seed"])
            L.save_exr(out / "8dna" / regime / f"{p.stem}.exr", img)
            blk = refit_block(gt, rois, n_t0, img)
            rows.append({"checkpoint": L.rel(p), "sha256": L.sha256(p),
                         "epoch": int(re.search(r"epoch=(\d+)", p.name).group(1)), **blk})
            print(regime, p.name, f"ratio {blk['error_ratio']:.3f} gain {blk['gain']:.3f} {blk['recovers']}", flush=True)
        # the historical selection (last.ckpt = epoch 29) re-rendered vs the historical render of it
        hist_img = L.load_exr(L.RESULTS / "refit" / "renders" / regime / "T3_refit_at_T3.exr")
        last = L.load_exr(out / "8dna" / regime / f"{ckpts[-1].stem}.exr")
        rec["8dna"][regime] = {"rows": rows, "rerender_vs_historical_render_max_abs": float(np.abs(last - hist_img).max()),
                               "rerender_vs_historical_render_mean_abs": float(np.abs(last - hist_img).mean())}

    # --- RNA -------------------------------------------------------------------
    cb = json.loads(open(L.EXPERIMENT / "protocol/teaset_cross_backbone_locked.json", encoding="utf-8").read())
    inf = cb["rna_inference"]
    rck = sorted((L.RESULTS / "rna_teaset" / "ckpt" / "rna-teaset-T3-common-light").rglob("*.ckpt"))
    feats = L.RESULTS / cb["rna_features_dir"] / "T3.npz"
    h5 = L.RESULTS / cb["rna_dataset_dir"] / "teaset_T3_train.h5"
    od = out / "rna"
    od.mkdir(parents=True, exist_ok=True)
    tags = {p: re.sub(r"[^A-Za-z0-9_.=-]", "_", p.stem) for p in rck}
    spec = {"common": ["--train-h5", wsl_path(h5), "--features", wsl_path(feats), "--mode", "current",
                       "--training-light-intensity", str(inf["training_light_intensity"]),
                       "--light-model", inf.get("light_model", "area-sampled")],
            "jobs": [{"checkpoint": wsl_path(p), "out": wsl_path(od / tags[p])} for p in rck]}
    (od / "jobs.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    rc = run_detached("historical_rescore_rna", out / "logs",
                      [".venv/bin/python", wsl_path(HERE / "wsl" / "nrc_rna_eval_wsl.py"), "--jobs", wsl_path(od / "jobs.json"),
                       "--out", wsl_path(od / "infer_timing.json")], RNA_ROOT)
    if rc:
        raise SystemExit(f"historical RNA rescore failed rc={rc}")
    fdir = L.RESULTS / cb["frozen_output"]
    gt = {s: (L.load_exr(fdir / s / "gt_A.exr"), L.load_exr(fdir / s / "gt_B.exr")) for s in ("T0", "T3")}
    rois = load_rois(L.RESULTS / cb["gt_output"], ("T0", "T3"))
    n_t0 = np.load(L.RESULTS / cb["rna_output"] / "T0_canonical.npy").astype(np.float32)
    rows = []
    for p in rck:
        img = np.load(str(od / tags[p]) + ".npy").astype(np.float32)
        blk = refit_block(gt, rois, n_t0, img)
        m = re.search(r"epoch=(\d+)-val_psnr=([0-9.]+)dB", p.name)
        rows.append({"checkpoint": L.rel(p), "tag": tags[p], "epoch": None if m is None else int(m.group(1)),
                     "val_psnr_db": None if m is None else float(m.group(2)), **blk})
        print("rna", p.name, f"ratio {blk['error_ratio']:.3f} gain {blk['gain']:.3f} {blk['recovers']}", flush=True)
    best = max((r for r in rows if r["val_psnr_db"] is not None), key=lambda r: r["val_psnr_db"])
    bimg = np.load(str(od / best["tag"]) + ".npy").astype(np.float32)
    himg = np.load(L.RESULTS / cb["rna_output"] / "refit_T3_current.npy").astype(np.float32)
    rec["rna"] = {"rows": rows, "validation_best": best["checkpoint"],
                  "rerender_vs_historical_render_max_abs": float(np.abs(bimg - himg).max())}
    R.write_json(out / "historical_rescore.json", rec)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
