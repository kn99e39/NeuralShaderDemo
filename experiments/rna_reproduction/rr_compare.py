"""Worklog-35: compare native RNA renders (rna_<f>.exr) with the native Cycles references (ref_<f>.exr)
written by upstream scripts/render.py for config/render/lego_cmp.yml frames (RNA venv).

    python rr_compare.py --renders <work/renders/rna_tog> --out <json>

The paper's Fig. 8 PSNR transform, view and reference spp are not published (worklog 35 manifest), so
every number is reported under explicitly named conventions:
  ldr   = upstream utils.imageops.tonemap (clamp >= 0, x/(1+x), ^(1/2.2)) - the code's validation transform
  full  = all 1024^2 pixels (background included; both images are 0 there)
  mask  = reference alpha > 0 (asset pixels)
  logl1 = mean |log1p(pred) - log1p(ref)| over the mask; hf = top-10% reference-gradient mask pixels
The reference is Cycles CPU 2048 spp WITH OIDN denoising and direct clamp 20 (rna/renderers.py:271-350).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read(p):
    import pyexr

    a = pyexr.open(str(p)).get()
    return a.astype(np.float32)


def ldr(x):
    x = np.clip(x, 0, None)
    return (x / (1 + x)) ** (1 / 2.2)


def psnr(a, b):
    return float(-10 * np.log10(((a - b) ** 2).mean() + 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    R = Path(args.renders)
    rec = {"conventions": __doc__, "frames": {}}
    for f in (13, 19, 26, 33, 68):
        ref = read(R / "lego" / f"ref_{f}.exr")
        alpha = ref[..., 3] if ref.shape[-1] == 4 else (ref[..., :3].sum(-1) > 0).astype(np.float32)
        refc = ref[..., :3]
        m = alpha > 0
        lum = ldr(refc).mean(-1); grad = np.zeros_like(lum)
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            ok = m & np.roll(m, (dy, dx), (0, 1))
            grad = np.maximum(grad, np.where(ok, np.abs(lum - np.roll(lum, (dy, dx), (0, 1))), 0))
        hf = m & (grad >= np.quantile(grad[m], 0.9))
        row = {"ref_channels": int(ref.shape[-1]), "mask_fraction": float(m.mean()), "ref_max": float(refc.max())}
        for name, sub in (("HQ", "lego"), ("small", "lego_rt")):
            pr = read(R / sub / f"rna_{f}.exr")[..., :3]
            row[name] = {"psnr_full_ldr": psnr(ldr(pr), ldr(refc)), "psnr_mask_ldr": psnr(ldr(pr)[m], ldr(refc)[m]),
                         "logl1_mask": float(np.abs(np.log1p(np.clip(pr[m], 0, None)) - np.log1p(np.clip(refc[m], 0, None))).mean()),
                         "hf_abs_ldr": float(np.abs(ldr(pr)[hf] - ldr(refc)[hf]).mean()), "pred_max": float(pr.max())}
        rec["frames"][f] = row
    for name in ("HQ", "small"):
        rec[f"mean_{name}"] = {k: float(np.mean([rec["frames"][f][name][k] for f in rec["frames"]])) for k in rec["frames"][13][name]}
    Path(args.out).write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "conventions"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
