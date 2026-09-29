"""Reviewer-facing exports for the locked teaset evaluation (presentation only).

Reads the EXRs written by teaset_frozen_eval.py and run_static_baseline.py,
writes PNG/GIF panels to results/evaluation/<worklog>/ and a manifest with
source provenance and SHA-256 of every export.  Computes no new metric.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import ednalib as L


def ldr(img):
    return L.to_u8(L.tonemap(img))


def crop_box(mask: np.ndarray, pad: int = 24) -> tuple[slice, slice]:
    ys, xs = np.nonzero(mask)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, mask.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, mask.shape[1])
    side = max(y1 - y0, x1 - x0)
    return slice(y0, min(y0 + side, mask.shape[0])), slice(x0, min(x0 + side, mask.shape[1]))


def upscale(img: np.ndarray, size: int = 384) -> np.ndarray:
    from PIL import Image

    return np.array(Image.fromarray(img).resize((size, size), Image.NEAREST))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--worklog", required=True)
    args = ap.parse_args()
    proto = json.loads(open(args.protocol, encoding="utf-8").read())
    L.init_upstream()
    src = L.RESULTS / proto["frozen_output"]
    gt_dir = L.RESULTS / proto["gt_output"]
    out = L.ROOT / "results" / "evaluation" / args.worklog
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"worklog": args.worklog, "protocol": L.rel(L.EXPERIMENT / args.protocol),
                "project_commit": L.git_head(L.ROOT), "display": "clamp(x^(1/2.2),0,1)",
                "error_maps": "per-pixel mean |display difference|, inferno, fixed scale 0..0.2", "files": {}}
    sources: dict[str, list[str]] = {}

    def save(name: str, img: np.ndarray, used: list, gif: list | None = None):
        path = out / name
        if gif is not None:
            L.save_gif(path, gif)
        else:
            L.save_png(path, img)
        sources[name] = [L.rel(p) for p in used]

    states = list(proto["states"])
    gt = {s: 0.5 * (L.load_exr(src / s / "gt_A.exr") + L.load_exr(src / s / "gt_B.exr")) for s in states}
    nn = {s: L.load_exr(src / s / "8dna_attached.exr") for s in states}
    up0 = L.load_exr(src / "T0" / "8dna_upstream.exr")
    strong = "T3"
    m_int = np.load(gt_dir / f"rois_{strong}.npz")["mask_interaction"]
    m_mov = np.load(gt_dir / f"rois_{strong}.npz")["mask_mover"] | np.load(gt_dir / "rois_T0.npz")["mask_mover"]
    ys, xs = crop_box(m_int | m_mov)
    err = lambda a, b: L.error_map(np.abs(L.tonemap(a) - L.tonemap(b)).mean(-1), 0.2)
    exr = lambda s, k: src / s / f"{k}.exr"

    # 1. static / canonical baseline
    save("01_T0_canonical_GT_vs_8DNA.png",
         np.concatenate([L.label(ldr(gt["T0"]), "T0 path-traced GT"), L.label(ldr(up0), "T0 released 8DNA (upstream path)"),
                         L.label(ldr(nn["T0"]), "T0 8DNA via adapter"), L.label(err(up0, gt["T0"]), "|8DNA - GT| (0..0.2)")], 1),
         [exr("T0", "gt_A"), exr("T0", "gt_B"), exr("T0", "8dna_upstream"), exr("T0", "8dna_attached")])
    # 2. trajectory strips
    rows = [np.concatenate([L.label(ldr(gt[s]), f"GT {s}") for s in states], 1),
            np.concatenate([L.label(ldr(nn[s]), f"8DNA {s}") for s in states], 1),
            np.concatenate([L.label(err(nn[s], gt[s]), f"|8DNA-GT| {s}") for s in states], 1)]
    save("02_trajectory_GT_8DNA_error.png", np.concatenate(rows, 0),
         [exr(s, k) for s in states for k in ("gt_A", "gt_B", "8dna_attached")])
    # 3-4. flickers T0 <-> strongest
    save("03_flicker_GT_T0_vs_T3.gif", None, [exr("T0", "gt_A"), exr(strong, "gt_A")],
         gif=[L.label(ldr(gt["T0"]), "GT T0"), L.label(ldr(gt[strong]), "GT T3")])
    save("04_flicker_8DNA_T0_vs_T3.gif", None, [exr("T0", "8dna_attached"), exr(strong, "8dna_attached")],
         gif=[L.label(ldr(nn["T0"]), "frozen 8DNA T0"), L.label(ldr(nn[strong]), "frozen 8DNA T3")])
    # 5. strongest state GT vs 8DNA
    save("05_T3_GT_vs_8DNA.png",
         np.concatenate([L.label(ldr(gt[strong]), "T3 GT"), L.label(ldr(nn[strong]), "T3 frozen 8DNA"),
                         L.label(err(nn[strong], gt[strong]), "|8DNA - GT| (0..0.2)")], 1),
         [exr(strong, "gt_A"), exr(strong, "gt_B"), exr(strong, "8dna_attached")])
    # 6. interaction-region crops
    crops = [[upscale(ldr(gt[s])[ys, xs]) for s in ("T0", "T2", "T3")],
             [upscale(ldr(nn[s])[ys, xs]) for s in ("T0", "T2", "T3")]]
    crops = [[L.label(c, f"{'GT' if r == 0 else '8DNA'} {s}") for c, s in zip(row, ("T0", "T2", "T3"))] for r, row in enumerate(crops)]
    save("06_interaction_crop_T0_T2_T3.png", np.concatenate([np.concatenate(r, 1) for r in crops], 0),
         [exr(s, k) for s in ("T0", "T2", "T3") for k in ("gt_A", "gt_B", "8dna_attached")])
    save("07_interaction_crop_flicker_GT.gif", None, [exr(s, "gt_A") for s in ("T0", "T2", "T3")],
         gif=[L.label(upscale(ldr(gt[s])[ys, xs]), f"GT {s}") for s in ("T0", "T2", "T3")])
    save("08_interaction_crop_flicker_8DNA.gif", None, [exr(s, "8dna_attached") for s in ("T0", "T2", "T3")],
         gif=[L.label(upscale(ldr(nn[s])[ys, xs]), f"8DNA {s}") for s in ("T0", "T2", "T3")])
    # 9. envelope sensitivity and coordinate-mismatch diagnostic at T3
    fx, upm = L.load_exr(exr(strong, "8dna_fixed")), L.load_exr(exr(strong, "8dna_upstream"))
    save("09_T3_query_modes.png",
         np.concatenate([L.label(ldr(nn[strong]), "attached (primary)"), L.label(ldr(fx), "fixed envelope"),
                         L.label(ldr(upm), "upstream: current-world query (diagnostic)"), L.label(ldr(gt[strong]), "GT")], 1),
         [exr(strong, "8dna_attached"), exr(strong, "8dna_fixed"), exr(strong, "8dna_upstream")])
    # 10. ROI overlay
    save("10_rois_on_T0.png", L.label(np.array(__import__("PIL.Image", fromlist=["Image"]).open(gt_dir / "rois_on_T0.png")),
                                        "ROIs: red interaction, blue tray, green far, yellow mover"), [gt_dir / "rois_on_T0.png"])
    # 11. static baselines if present
    for tag in ("teaset_T0_512", "seal_scene2_official"):
        p = L.RESULTS / "static_baseline" / tag / "side_by_side.png"
        if p.exists():
            save(f"11_static_baseline_{tag}.png", np.array(__import__("PIL.Image", fromlist=["Image"]).open(p).convert("RGB")), [p])

    for name, used in sources.items():
        manifest["files"][name] = {"sha256": L.sha256(out / name), "sources": used}
    L.write_json(out / "manifest.json", manifest)
    print("exports:", len(sources))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
