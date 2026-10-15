"""Worklog-33: worklog-32 arms scored against the converged (8x spp) references instead of the noisy
dataset GT, on validation views 0/10/20/30 (predictions from rso_diag maps.npz; no inference).

    python rbq_batch1_vs_ref.py --root results --out results/rna_baseline_quality/batch1_arms_vs_reference.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ARMS = ("A", "P1", "P4", "P16", "P16w", "REINIT")


def ldr(x):
    x = np.clip(x, 0, None)
    return (x / (1 + x)) ** (1 / 2.2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    out = {}
    for c in ("C1", "C2"):
        mp = np.load(root / "rna_surface_ownership/diag_wl32" / c / "maps.npz")
        res = {}
        for arm in ARMS:
            s = {"gt": 0.0, "ref": 0.0}; lg = {"gt": 0.0, "ref": 0.0}; n = 0
            for v in (0, 10, 20, 30):
                m = mp[f"owner16_{v}"].reshape(-1) >= 0
                p = mp[f"pred_{arm}_{v}"].reshape(-1, 3)[m]
                tg = {"gt": mp[f"gt_{v}"].reshape(-1, 3)[m],
                      "ref": np.load(root / "rna_baseline_quality/reference" / f"{c}_view{v}_ref.npz")["color"][m]}
                for k, t in tg.items():
                    s[k] += float(((ldr(p) - ldr(t)) ** 2).sum())
                    lg[k] += float(np.abs(np.log1p(np.clip(p, 0, None)) - np.log1p(np.clip(t, 0, None))).sum())
                n += p.size
            res[arm] = {f"psnr_vs_{k}": float(-10 * np.log10(s[k] / n)) for k in s} | {f"logl1_vs_{k}": lg[k] / n for k in lg}
        for arm in ARMS:
            for k in ("gt", "ref"):
                res[arm][f"dpsnr_vs_A_{k}"] = res[arm][f"psnr_vs_{k}"] - res["A"][f"psnr_vs_{k}"]
                res[arm][f"logl1_ratio_A_{k}"] = res[arm][f"logl1_vs_{k}"] / res["A"][f"logl1_vs_{k}"]
        out[c] = res
    Path(args.out).write_text(json.dumps(out, indent=1))
    for c, res in out.items():
        for arm, r in res.items():
            print(c, arm, {k: round(v, 3) for k, v in r.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
