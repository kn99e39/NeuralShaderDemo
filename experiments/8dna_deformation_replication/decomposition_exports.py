"""Reviewer panels for the reference-only transport decomposition (presentation only).

Called as `transport_decomposition.py exports --worklog <N>`.  Reads the full
component images (image_<state>.npz, 1024 spp, seed A) and decomposition.json;
computes no new metric.  Component groups are those of
protocol/teaset_transport_decomposition.json.  Raw component images use the
reference display (clamp(x^(1/2.2), 0, 1)); signed difference images map
T3 - T0 to red (increase) / blue (decrease) on a fixed per-regime scale.
"""

from __future__ import annotations

import json

import numpy as np

import ednalib as L
import transport_decomposition as D
from cross_backbone_exports import crop_box, upscale

SHOW = (("total", None), ("direct", "direct"), ("mover-mediated", "mover_mediated"),
        ("stationary single-bounce", "stationary_single"), ("higher-order, no mover", "higher_other"))


def signed(img: np.ndarray, scale: float) -> np.ndarray:
    v = np.clip(img.mean(-1) / scale, -1, 1)
    rgb = np.stack([np.maximum(v, 0), np.zeros_like(v), np.maximum(-v, 0)], -1) ** (1 / 2.2)
    return (rgb * 255).astype(np.uint8)


def main(worklog: str) -> int:
    proto = json.loads(open(L.EXPERIMENT / D.PROTO, encoding="utf-8").read())
    L.init_upstream()
    G = D.groups(proto)
    dec = json.loads(open(L.RESULTS / D.OUT / "decomposition.json", encoding="utf-8").read())
    out = L.ROOT / "results" / "evaluation" / worklog
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"worklog": worklog, "project_commit": L.git_head(L.ROOT), "display": "clamp(x^(1/2.2),0,1)",
                "content": "reference-only transport decomposition of the teaset interaction ROI; no neural renders",
                "component_images": "diagnostic estimator, 1024 spp, seed A (decomposition_exports.py)", "files": {}}
    sources: dict[str, list] = {}

    def save(name, img, used, gif=None):
        L.save_gif(out / name, gif) if gif is not None else L.save_png(out / name, img)
        sources[name] = [L.rel(p) for p in used]

    ld = lambda x: L.to_u8(L.tonemap(x))
    for regime in proto["regimes"]:
        paths = {s: L.RESULTS / D.OUT / regime / f"image_{s}.npz" for s in ("T0", "T3")}
        img = {}
        for s, p in paths.items():
            z = np.load(p)
            comp = {"total": z["total"].astype(np.float64)}
            for _, g in SHOW[1:]:
                comp[g] = z["classes"][:, :, G[g]].sum(2).astype(np.float64)
            img[s] = comp
        roi_dir = D.regime_setup(regime, "T3")[3].parent
        m3 = np.load(roi_dir / "rois_T3.npz")
        m0 = np.load(roi_dir / "rois_T0.npz")
        ys, xs = crop_box(m3["mask_interaction"] | m3["mask_mover"] | m0["mask_mover"])
        dtot = img["T3"]["total"] - img["T0"]["total"]
        scale = float(np.percentile(np.abs(dtot.mean(-1))[ys, xs], 99)) or 1.0
        used = list(paths.values())
        # 1. raw components, full frame
        rows = [np.concatenate([L.label(ld(img[s][k if g is None else g]), f"{regime} {s}: {k}") for k, g in SHOW], 1)
                for s in ("T0", "T3")]
        rows.append(np.concatenate([L.label(signed(img["T3"][k if g is None else g] - img["T0"][k if g is None else g], scale),
                                            f"T3 - T0 {k} (+-{scale:.2f})") for k, g in SHOW], 1))
        save(f"{regime}_01_components_full.png", np.concatenate(rows, 0), used)
        # 2. interaction crops
        crow = [np.concatenate([L.label(upscale(ld(img[s][k if g is None else g])[ys, xs], 320), f"{s} {k}") for k, g in SHOW], 1)
                for s in ("T0", "T3")]
        crow.append(np.concatenate([L.label(upscale(signed(img["T3"][k if g is None else g] - img["T0"][k if g is None else g], scale)[ys, xs], 320),
                                            f"T3-T0 {k}") for k, g in SHOW], 1))
        save(f"{regime}_02_components_interaction_crop.png", np.concatenate(crow, 0), used)
        # 3. flickers: does the mover's reflection move with the part?
        for k, g in (("total", None), ("mover-mediated", "mover_mediated"), ("stationary single-bounce", "stationary_single")):
            key = k if g is None else g
            save(f"{regime}_03_flicker_{key}.gif", None, used,
                 gif=[L.label(upscale(ld(img[s][key])[ys, xs], 384), f"{regime} {s}: {k}") for s in ("T0", "T3")])
        # 4. ROI overlay on the decomposition crop
        roi = np.zeros((512, 512, 3), np.uint8)
        roi[m0["mask_interaction"].reshape(512, 512)] = (255, 60, 60)
        over = (0.6 * ld(img["T0"]["total"]) + 0.4 * roi).astype(np.uint8)
        save(f"{regime}_04_interaction_roi_on_T0.png", L.label(upscale(over[ys, xs], 384), "interaction ROI (red) on teapot4"),
             used + [roi_dir / "rois_T0.npz"])
        manifest[regime] = {"batch_answer": dec["regimes"][regime]["batch_answer"],
                            "decision": dec["regimes"][regime]["decision"], "difference_scale_linear": scale}
    for name in sources:
        manifest["files"][name] = {"sha256": L.sha256(out / name), "sources": sources[name]}
    L.write_json(out / "manifest.json", manifest)
    print("exports:", len(sources))
    return 0
