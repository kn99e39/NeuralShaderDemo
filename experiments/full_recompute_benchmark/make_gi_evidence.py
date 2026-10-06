"""Signed G0->G1 transport-change panels from render_gi_evidence.py output.

python make_gi_evidence.py <evidence_dir> <out_dir>

Needs numpy, Pillow and mitsuba (8DNA Windows venv). For one evidence
directory (G0_s0, G1_s0, G0_s1 multilayer EXR) it writes:
  G0.png, G1.png, flicker.gif, side_by_side.jpg  - display renders as saved by Blender
  diff_signed.png      - box-filtered luminance G1-G0 on stationary pixels; orange = brighter
                         in G1, blue = darker, light grey = no change; first-hit-changed pixels
                         dark grey. Symmetric scale = 99.5th percentile of |change|.
  noise_signed.png     - the same for G0 seed 0 - G0 seed 1 at the same scale (noise floor)
  crop_panel.jpg       - G0 / G1 / signed change at full resolution, for the 640x360 window with the
                         largest summed |change| (selected by change: illustration, not a test)
  stats.json
"""

from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import frb_records as R  # noqa: E402

NEG = np.array([0x2a, 0x78, 0xd6]) / 255.0   # darker in G1
POS = np.array([0xeb, 0x68, 0x34]) / 255.0   # brighter in G1
MID = np.array([0xf0, 0xef, 0xec]) / 255.0   # no change
MOVED = np.array([0x52, 0x51, 0x4e]) / 255.0  # first hit changed
K = 5


def box(a, k=K):
    p = k // 2
    c = np.cumsum(np.cumsum(np.pad(a, ((p + 1, p), (p + 1, p)), mode="edge"), 0), 1)
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def diverging(d, scale, moved):
    t = np.clip(d / max(scale, 1e-12), -1, 1)[..., None]
    img = np.where(t >= 0, MID * (1 - t) + POS * t, MID * (1 + t) + NEG * (-t))
    img[moved] = MOVED
    return (img * 255).astype(np.uint8)


def main():
    ev, out = sys.argv[1:3]
    import mitsuba as mi
    from PIL import Image
    mi.set_variant("scalar_rgb")
    os.makedirs(out, exist_ok=True)

    def read(n):
        layers = dict(mi.Bitmap(os.path.join(ev, n + ".exr")).split())
        rgb = np.array(layers["RenderLayer.Combined"])[..., :3].astype(np.float64)
        return rgb @ np.array([0.2126, 0.7152, 0.0722]), np.array(layers["RenderLayer.Depth"]).astype(np.float64)

    l0, z0 = read("G0_s0")
    l1, z1 = read("G1_s0")
    ln, _ = read("G0_s1")
    stationary = np.abs(z1 - z0) <= 1e-4 * np.maximum(np.abs(z0), 1e-6)
    moved = ~stationary
    d = box(l1 - l0)
    n = box(l0 - ln)
    m = stationary
    sel = np.abs(d) > 3 * np.abs(n)
    scale = float(np.percentile(np.abs(d[m]), 99.5))
    meta = R.load_json(os.path.join(ev, "evidence_meta.json"))
    stats = {
        "evidence_dir": ev, "spp": meta["spp"], "mover_mode": meta["mover_mode"],
        "pixels": int(m.size), "stationary_fraction": float(m.mean()),
        "mean_luminance_stationary": float(l0[m].mean()),
        "per_pixel_seed_noise_mean_abs": float(np.abs(l0 - ln)[m].mean()),
        "per_pixel_seed_noise_relative": float(np.abs(l0 - ln)[m].mean() / l0[m].mean()),
        "box_k": K,
        "box_change_mean_abs": float(np.abs(d[m]).mean()),
        "box_noise_mean_abs": float(np.abs(n[m]).mean()),
        "frac_stationary_change_gt_3x_local_noise": float(sel[m].mean()),
        "mean_abs_change_where_gt_3x": float(np.abs(d[m & sel]).mean()) if (m & sel).any() else None,
        "mean_abs_noise_where_gt_3x": float(np.abs(n[m & sel]).mean()) if (m & sel).any() else None,
        "relative_change_where_gt_3x": float(np.abs(d[m & sel]).mean() / l0[m & sel].mean()) if (m & sel).any() else None,
        "brighter_fraction_of_selected": float((d[m & sel] > 0).mean()) if (m & sel).any() else None,
        "display_scale_luminance": scale,
    }
    R.dump_json(stats, os.path.join(out, "stats.json"))

    a = Image.open(os.path.join(ev, "G0_s0.png")).convert("RGB")
    b = Image.open(os.path.join(ev, "G1_s0.png")).convert("RGB")
    w, h = a.size
    shutil.copyfile(os.path.join(ev, "G0_s0.png"), os.path.join(out, "G0.png"))
    shutil.copyfile(os.path.join(ev, "G1_s0.png"), os.path.join(out, "G1.png"))
    half = (w // 2, h // 2)
    a.resize(half).save(os.path.join(out, "flicker.gif"), save_all=True, append_images=[b.resize(half)],
                        duration=700, loop=0)
    side = Image.new("RGB", (w, h // 2))
    side.paste(a.resize(half), (0, 0))
    side.paste(b.resize(half), (w // 2, 0))
    side.save(os.path.join(out, "side_by_side.jpg"), quality=92)
    dimg = Image.fromarray(diverging(np.where(m, d, 0), scale, moved))
    dimg.resize(half).save(os.path.join(out, "diff_signed.png"))
    Image.fromarray(diverging(np.where(m, n, 0), scale, moved)).resize(half).save(
        os.path.join(out, "noise_signed.png"))

    cw, ch = 640, 360
    mag = np.where(m, np.abs(d), 0)
    integ = np.pad(mag.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    best, bxy = -1, (0, 0)
    for y in range(0, h - ch + 1, 20):
        for x in range(0, w - cw + 1, 20):
            s = integ[y + ch, x + cw] - integ[y, x + cw] - integ[y + ch, x] + integ[y, x]
            if s > best:
                best, bxy = s, (x, y)
    x, y = bxy
    box_ = (x, y, x + cw, y + ch)
    panel = Image.new("RGB", (3 * cw + 16, ch), (0xfc, 0xfc, 0xfb))
    for i, im in enumerate((a.crop(box_), b.crop(box_), dimg.crop(box_))):
        panel.paste(im, (i * (cw + 8), 0))
    panel.save(os.path.join(out, "crop_panel.jpg"), quality=92)
    stats["crop_window_xywh"] = [x, y, cw, ch]
    R.dump_json(stats, os.path.join(out, "stats.json"))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
