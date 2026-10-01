"""Area-trained RNA sensitivity (worklog 23): training-target statistics and reviewer panels.

    windows/run.ps1 rna_sensitivity_exports.py --worklog 23

1. Measurement: the locked (delta-light) and area-trained T0 validation sets
   share cameras and light directions (same seeds), so their colour targets
   can be compared view by view: share of on-asset energy carried by the top
   0.1% of pixels, and local speckle (mean |pixel - 3x3 median| / 3x3 median).
   Written to rna_teaset/target_tail_stats.json.
2. Presentation: GT / locked RNA / area-trained RNA at T0 and T3, interaction
   crops, T0<->T3 crop flickers and validation-target pairs, copied into
   results/evaluation/<worklog>/ with a manifest of sources and SHA-256.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import ednalib as L

STATES = ("T0", "T1b", "T3")


def tail_stats(color: np.ndarray, alpha: np.ndarray) -> dict:
    from scipy.ndimage import median_filter

    res = int(np.sqrt(color.shape[0]))
    c = color.reshape(res, res, 3).astype(np.float32)
    m = alpha.reshape(res, res) > 0.99
    lum = c.mean(-1)
    v = np.sort(lum[m])
    k = max(1, len(v) // 1000)
    med = median_filter(lum, 3)
    return {"top_0p1pct_energy_share": float(v[-k:].sum() / max(v.sum(), 1e-12)),
            "speckle": float(np.abs(lum - med)[m].mean() / max(med[m].mean(), 1e-12)),
            "max": float(v[-1]), "mean": float(v.mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worklog", required=True)
    args = ap.parse_args()
    import h5py

    L.init_upstream()
    R = L.RESULTS
    locked = json.loads(open(L.EXPERIMENT / "protocol/teaset_cross_backbone_locked.json", encoding="utf-8").read())
    sens = json.loads(open(L.EXPERIMENT / "protocol/teaset_rna_area_trained_sensitivity.json", encoding="utf-8").read())
    h5a = R / locked["rna_dataset_dir"] / "teaset_T0_val.h5"
    h5b = R / sens["rna_dataset_dir"] / "teaset_T0_val.h5"

    # 1. target statistics
    stats = {"delta_trained": [], "area_trained": []}
    with h5py.File(h5a, "r") as fa, h5py.File(h5b, "r") as fb:
        n = fa["color"].shape[0]
        same = bool(np.array_equal(np.array(fa["light_dir"][:, 0]), np.array(fb["light_dir"][:, 0])))
        for v in range(n):
            stats["delta_trained"].append(tail_stats(np.array(fa["color"][v]), np.array(fa["alpha"][v])))
            stats["area_trained"].append(tail_stats(np.array(fb["color"][v]), np.array(fb["alpha"][v])))
        pair_views = [0, 7]
        pairs = {v: (np.array(fa["color"][v], np.float32), np.array(fb["color"][v], np.float32)) for v in pair_views}
        res = int(np.sqrt(fa["color"].shape[1]))
    summary = {}
    for key, rows in stats.items():
        summary[key] = {f: {"median": float(np.median([r[f] for r in rows])),
                            "p90": float(np.percentile([r[f] for r in rows], 90))}
                        for f in ("top_0p1pct_energy_share", "speckle", "max")}
    out_stats = R / "rna_teaset" / "target_tail_stats.json"
    L.write_json(out_stats, {"purpose": "training-target tails of the locked (delta) vs area-trained T0 validation sets; same cameras and light directions",
                             "same_light_directions": same, "views": n, "summary": summary, "per_view": stats,
                             "sources": [L.rel(h5a), L.rel(h5b)], "environment": L.environment_record()})

    # 2. panels
    out = L.ROOT / "results" / "evaluation" / args.worklog
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"worklog": args.worklog, "project_commit": L.git_head(L.ROOT), "display": "clamp(x^(1/2.2),0,1)",
                "regime": "teaset_cross_backbone_common_light (15-degree area light); single shared GT; post-hoc area-trained RNA sensitivity",
                "files": {}}
    sources: dict[str, list] = {}

    def save(name, img, used, gif=None):
        L.save_gif(out / name, gif) if gif is not None else L.save_png(out / name, img)
        sources[name] = [L.rel(p) for p in used]

    ld = lambda x: L.to_u8(L.tonemap(x))
    fdir = R / locked["frozen_output"]
    ra, rb = R / locked["rna_output"], R / sens["rna_output"]
    gexr = lambda s: [fdir / s / "gt_A.exr", fdir / s / "gt_B.exr"]
    gt = {s: 0.5 * (L.load_exr(gexr(s)[0]) + L.load_exr(gexr(s)[1])) for s in STATES}
    rna_a = {s: np.load(ra / f"{s}_canonical.npy").astype(np.float32) for s in STATES}
    rna_b = {s: np.load(rb / f"{s}_canonical.npy").astype(np.float32) for s in STATES}
    err = lambda a, b: L.error_map(np.abs(L.tonemap(a) - L.tonemap(b)).mean(-1), 0.2)

    rows = []
    for s in ("T0", "T3"):
        rows.append(np.concatenate([L.label(ld(gt[s]), f"GT {s} (shared)"),
                                    L.label(ld(rna_a[s]), f"RNA delta-trained (locked) {s}"),
                                    L.label(ld(rna_b[s]), f"RNA area-trained (sensitivity) {s}")], 1))
    rows.append(np.concatenate([L.label(np.zeros_like(ld(gt["T0"])), "(error maps vs the shared GT)"),
                                L.label(err(rna_a["T0"], gt["T0"]), "|locked - GT| T0 (0..0.2)"),
                                L.label(err(rna_b["T0"], gt["T0"]), "|area-trained - GT| T0 (0..0.2)")], 1))
    save("01_GT_RNA_locked_vs_area_trained.png", np.concatenate(rows, 0),
         gexr("T0") + gexr("T3") + [ra / "T0_canonical.npy", ra / "T3_canonical.npy", rb / "T0_canonical.npy", rb / "T3_canonical.npy"])

    from cross_backbone_exports import crop_box, upscale

    gt_dir = R / locked["gt_output"]
    z = np.load(gt_dir / "rois_T3.npz")
    ys, xs = crop_box(z["mask_interaction"] | z["mask_mover"] | np.load(gt_dir / "rois_T0.npz")["mask_mover"])
    crop_rows = []
    for tag, imgs in (("GT", gt), ("RNA locked", rna_a), ("RNA area-trained", rna_b)):
        crop_rows.append(np.concatenate([L.label(upscale(ld(imgs[s])[ys, xs]), f"{tag} {s}") for s in STATES], 1))
    save("02_interaction_crops.png", np.concatenate(crop_rows, 0),
         [p for s in STATES for p in gexr(s)] + [d / f"{s}_canonical.npy" for d in (ra, rb) for s in STATES])
    for tag, imgs, d in (("GT", gt, None), ("RNA_locked", rna_a, ra), ("RNA_area_trained", rna_b, rb)):
        used = [p for s in ("T0", "T3") for p in gexr(s)] if d is None else [d / "T0_canonical.npy", d / "T3_canonical.npy"]
        save(f"03_interaction_crop_flicker_{tag}.gif", None, used,
             gif=[L.label(upscale(ld(imgs[s])[ys, xs]), f"{tag} {s}") for s in ("T0", "T3")])
    trows = []
    for v, (a, b) in pairs.items():
        trows.append(np.concatenate([L.label(ld(a.reshape(res, res, 3)), f"val view {v}: delta-light target (locked)"),
                                     L.label(ld(b.reshape(res, res, 3)), f"val view {v}: area-light target (sensitivity)")], 1))
    save("04_training_targets_delta_vs_area.png", np.concatenate(trows, 0), [h5a, h5b])

    manifest["target_tail_stats"] = summary
    for name in sources:
        manifest["files"][name] = {"sha256": L.sha256(out / name), "sources": sources[name]}
    L.write_json(out / "manifest.json", manifest)
    print(json.dumps(summary, indent=1))
    print("exports:", len(sources))
    return 0


if __name__ == "__main__":
    rc = main()
    # Outputs are complete here; skip the interpreter teardown, which crashes with
    # an access violation on this Windows setup once DrJit and torch are loaded.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
