"""Reviewer-facing exports for the cross-backbone batch (presentation only).

Reads the renders written by teaset_frozen_eval.py, render_rna_states.py and
render_8dna_refits.py, and writes PNG/GIF panels plus a manifest with source
provenance and SHA-256 of every export.  Computes no new metric: the numbers
printed on the panels come from cross_backbone.json.

Both models share one reference in this regime, so a single GT column is
correct here; the panels say so explicitly.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import ednalib as L

STATES = ("T0", "T1", "T1b", "T2", "T3")


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
    ap.add_argument("--protocol", default="protocol/teaset_cross_backbone_locked.json")
    ap.add_argument("--worklog", required=True)
    args = ap.parse_args()
    proto = json.loads(open(L.EXPERIMENT / args.protocol, encoding="utf-8").read())
    L.init_upstream()
    fdir = L.RESULTS / proto["frozen_output"]
    rdir = L.RESULTS / proto["rna_output"]
    gt_dir = L.RESULTS / proto["gt_output"]
    ev = json.loads(open(L.RESULTS / "cross_backbone" / "cross_backbone.json", encoding="utf-8").read())
    out = L.ROOT / "results" / "evaluation" / args.worklog
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"worklog": args.worklog, "protocol": L.rel(L.EXPERIMENT / args.protocol),
                "project_commit": L.git_head(L.ROOT), "display": "clamp(x^(1/2.2),0,1)",
                "regime": "teaset_cross_backbone_common_light: one Mitsuba scene, material set, camera and directional light shared by both models; a single GT column is therefore valid",
                "error_maps": "per-pixel mean |display difference|, inferno, fixed scale 0..0.2", "files": {}}
    sources: dict[str, list] = {}

    def save(name, img, used, gif=None):
        L.save_gif(out / name, gif) if gif is not None else L.save_png(out / name, img)
        sources[name] = [L.rel(p) for p in used]

    gt = {s: 0.5 * (L.load_exr(fdir / s / "gt_A.exr") + L.load_exr(fdir / s / "gt_B.exr")) for s in STATES}
    nn8 = {s: L.load_exr(fdir / s / "8dna_attached.exr") for s in STATES}
    rna = {s: np.load(rdir / f"{s}_canonical.npy").astype(np.float32) for s in STATES}
    err = lambda a, b: L.error_map(np.abs(L.tonemap(a) - L.tonemap(b)).mean(-1), 0.2)
    gexr = lambda s: [fdir / s / "gt_A.exr", fdir / s / "gt_B.exr"]

    z = np.load(gt_dir / "rois_T3.npz")
    ys, xs = crop_box(z["mask_interaction"] | z["mask_mover"] | np.load(gt_dir / "rois_T0.npz")["mask_mover"])

    # 1. static baselines in this regime
    save("01_T0_GT_RNA_8DNA.png",
         np.concatenate([L.label(ldr(gt["T0"]), "T0 path-traced GT (shared)"), L.label(ldr(rna["T0"]), "T0 frozen RNA"),
                         L.label(ldr(nn8["T0"]), "T0 frozen 8DNA")], 1),
         gexr("T0") + [rdir / "T0_canonical.npy", fdir / "T0" / "8dna_attached.exr"])
    # 2. strongest state, both models
    save("02_T3_GT_RNA_8DNA.png",
         np.concatenate([np.concatenate([L.label(ldr(gt["T3"]), "T3 GT"), L.label(ldr(rna["T3"]), "T3 frozen RNA"),
                                         L.label(ldr(nn8["T3"]), "T3 frozen 8DNA")], 1),
                         np.concatenate([L.label(ldr(gt["T0"]), "T0 GT (for reference)"),
                                         L.label(err(rna["T3"], gt["T3"]), "|RNA - GT| (0..0.2)"),
                                         L.label(err(nn8["T3"], gt["T3"]), "|8DNA - GT| (0..0.2)")], 1)], 0),
         gexr("T3") + gexr("T0") + [rdir / "T3_canonical.npy", fdir / "T3" / "8dna_attached.exr"])
    # 3. trajectories
    for tag, imgs, used in (("RNA", rna, [rdir / f"{s}_canonical.npy" for s in STATES]),
                            ("8DNA", nn8, [fdir / s / "8dna_attached.exr" for s in STATES])):
        save(f"03_trajectory_{tag}.png",
             np.concatenate([np.concatenate([L.label(ldr(gt[s]), f"GT {s}") for s in STATES], 1),
                             np.concatenate([L.label(ldr(imgs[s]), f"{tag} {s}") for s in STATES], 1),
                             np.concatenate([L.label(err(imgs[s], gt[s]), f"|{tag}-GT| {s}") for s in STATES], 1)], 0),
             [p for s in STATES for p in gexr(s)] + used)
    # 4. flickers T0 <-> T3
    for tag, imgs, used in (("GT", gt, gexr("T0") + gexr("T3")),
                            ("RNA", rna, [rdir / "T0_canonical.npy", rdir / "T3_canonical.npy"]),
                            ("8DNA", nn8, [fdir / "T0" / "8dna_attached.exr", fdir / "T3" / "8dna_attached.exr"])):
        save(f"04_flicker_{tag}_T0_vs_T3.gif", None, used,
             gif=[L.label(ldr(imgs["T0"]), f"{tag} T0"), L.label(ldr(imgs["T3"]), f"{tag} T3")])
    # 5. interaction crops
    rows = []
    for tag, imgs in (("GT", gt), ("RNA", rna), ("8DNA", nn8)):
        rows.append(np.concatenate([L.label(upscale(ldr(imgs[s])[ys, xs]), f"{tag} {s}") for s in ("T0", "T2", "T3")], 1))
    save("05_interaction_crop_GT_RNA_8DNA.png", np.concatenate(rows, 0),
         [p for s in ("T0", "T2", "T3") for p in gexr(s)]
         + [rdir / f"{s}_canonical.npy" for s in ("T0", "T2", "T3")]
         + [fdir / s / "8dna_attached.exr" for s in ("T0", "T2", "T3")])
    for tag, imgs, used in (("GT", gt, [p for s in ("T0", "T3") for p in gexr(s)]),
                            ("RNA", rna, [rdir / f"{s}_canonical.npy" for s in ("T0", "T3")]),
                            ("8DNA", nn8, [fdir / s / "8dna_attached.exr" for s in ("T0", "T3")])):
        save(f"06_interaction_crop_flicker_{tag}.gif", None, used,
             gif=[L.label(upscale(ldr(imgs[s])[ys, xs]), f"{tag} {s}") for s in ("T0", "T3")])
    # 7. refits
    rr = L.RESULTS / "refit" / "renders"
    for regime, label in (("common_light", "common light"), ("w21_envmap", "worklog-21 envmap")):
        t0, t3 = rr / regime / "T0_retrain_at_T0.exr", rr / regime / "T3_refit_at_T3.exr"
        if not t3.exists():
            continue
        g3 = gt["T3"] if regime == "common_light" else 0.5 * (
            L.load_exr(L.RESULTS / "w21_reference_correction" / "T3" / "gt_A.exr")
            + L.load_exr(L.RESULTS / "w21_reference_correction" / "T3" / "gt_B.exr"))
        n3 = nn8["T3"] if regime == "common_light" else L.load_exr(
            L.RESULTS / "frozen" / "teaset_locked" / "T3" / "8dna_attached.exr")
        save(f"07_8DNA_T3_frozen_vs_refit_{regime}.png",
             np.concatenate([L.label(ldr(g3), f"T3 GT ({label})"), L.label(ldr(n3), "T3 frozen (released)"),
                             L.label(ldr(L.load_exr(t3)), "T3 same-state refit"),
                             L.label(err(L.load_exr(t3), g3), "|refit - GT| (0..0.2)")], 1),
             [t0, t3])
    rna_refit = rdir / "refit_T3_current.npy"
    if rna_refit.exists():
        save("08_RNA_T3_frozen_vs_refit.png",
             np.concatenate([L.label(ldr(gt["T3"]), "T3 GT"), L.label(ldr(rna["T3"]), "T3 frozen RNA"),
                             L.label(ldr(np.load(rna_refit).astype(np.float32)), "T3 RNA same-state refit")], 1),
             [rna_refit, rdir / "T3_canonical.npy"] + gexr("T3"))
    # 9. ROI overlay
    from PIL import Image

    save("09_rois_on_T0.png", L.label(np.array(Image.open(gt_dir / "rois_on_T0.png").convert("RGB")),
                                      "ROIs: red interaction, blue tray, green far, yellow mover"),
         [gt_dir / "rois_on_T0.png"])

    manifest["summary"] = {
        "8dna": ev["8dna"]["rule_attached"]["classification"],
        "rna": ev["rna"]["rule_canonical"]["classification"],
        "static_gate_G1": ev["static_gate_G1"],
        "interaction_rise_and_gain": {
            s: {"dG": ev["8dna"]["rule_attached"]["per_state"][s]["interaction"]["dG"],
                "8dna": [ev["8dna"]["rule_attached"]["per_state"][s]["interaction"]["rise"],
                         ev["8dna"]["rule_attached"]["per_state"][s]["interaction"]["gain"]],
                "rna": [ev["rna"]["rule_canonical"]["per_state"][s]["interaction"]["rise"],
                        ev["rna"]["rule_canonical"]["per_state"][s]["interaction"]["gain"]]} for s in STATES}}
    for name in sources:
        manifest["files"][name] = {"sha256": L.sha256(out / name), "sources": sources[name]}
    L.write_json(out / "manifest.json", manifest)
    print("exports:", len(sources))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
