"""Architecture Batch 2 diagnostic B: training support per P16 unit (worklog 34; RNA venv, CPU).

    python b2_support.py --train <C1 train h5> --partitions <dir> --diag <diag dir with C1/C2 diag.json> --band 0.0072 --out <json>

The C2 training set used the same cameras, light directions and seeds as the worklog-22 C1 set
(only the radiance differs), so its sampling support is measured on the local C1 H5. Per unit:
samples (= the unit decoder's whole training set), samples per unit area, samples within `band` of a
same-part seam (3D distance to the nearest dense sample of another same-part unit), and coverage of
128 equal-area upper-hemisphere bins (8 height bands x 16 azimuths, +y up) by light and view
directions overall and within the band. Correlated (Spearman) with the dev-view P16/A error ratio.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rna_surface_ownership"))
import rso_model  # noqa: E402,F401  (RNA path)


def hemi_bin(d):
    h = np.clip(d[:, 1], 0, 0.99999)
    phi = (np.arctan2(d[:, 2], d[:, 0]) + np.pi) / (2 * np.pi)
    return (h * 8).astype(int) * 16 + np.clip((phi * 16).astype(int), 0, 15)


def spearman(a, b):
    r = lambda x: np.argsort(np.argsort(x))
    return float(np.corrcoef(r(np.asarray(a)), r(np.asarray(b)))[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--partitions", required=True)
    ap.add_argument("--diag", required=True)
    ap.add_argument("--band", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import h5py
    from rso_partition import Owner
    from scipy.spatial import cKDTree

    z = np.load(Path(args.partitions) / "partition_P16.npz")
    X, LAB, up, area = z["x"], z["label"], z["unit_part"], z["unit_area"]
    trees = {u: cKDTree(X[LAB == u]) for u in range(16)}
    own = Owner(Path(args.partitions) / "partition_P16.npz")
    f = h5py.File(args.train, "r")
    n = np.zeros(16, np.int64); nb = np.zeros(16, np.int64)
    lb = np.zeros((16, 128), bool); vb = np.zeros((16, 128), bool); lbb = np.zeros((16, 128), bool); vbb = np.zeros((16, 128), bool)
    for v in range(f["color"].shape[0]):
        g = lambda k: np.nan_to_num(f[k][v].astype(np.float32))
        m = g("alpha")[:, 0] >= 0.999
        P, L, Cd = g("position")[m], g("light_dir")[m], g("camera_dir")[m]
        o = own(P)[0]
        dn = np.full(len(P), np.inf)
        for u in range(16):
            s = np.flatnonzero(o == u)
            if not len(s):
                continue
            for w in range(16):
                if w != u and up[w] == up[u]:
                    dn[s] = np.minimum(dn[s], trees[w].query(P[s])[0])
        band = dn <= args.band
        bl, bv = hemi_bin(L), hemi_bin(Cd)
        np.add.at(n, o, 1); np.add.at(nb, o[band], 1)
        lb[o, bl] = True; vb[o, bv] = True; lbb[o[band], bl[band]] = True; vbb[o[band], bv[band]] = True
        if v % 50 == 0:
            print("view", v, flush=True)
    rec = {"band_m": args.band, "units": []}
    for u in range(16):
        rec["units"].append({"unit": u, "part": int(up[u]), "samples": int(n[u]), "samples_per_area": float(n[u] / area[u]),
                             "band_samples": int(nb[u]), "band_fraction": float(nb[u] / max(n[u], 1)),
                             "light_cov": float(lb[u].mean()), "view_cov": float(vb[u].mean()),
                             "band_light_cov": float(lbb[u].mean()), "band_view_cov": float(vbb[u].mean())})
    for c in ("C2", "C1"):
        p = Path(args.diag) / c / "diag.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        ratio = np.array(d["unit_err"]["P16"]) / np.array(d["unit_err"]["A"])
        seamed = [u for u in range(16) if (up == up[u]).sum() > 1]
        rec[f"spearman_{c}"] = {k: spearman([rec["units"][u][k] for u in seamed], ratio[seamed])
                                for k in ("samples", "samples_per_area", "band_samples", "band_fraction", "light_cov", "view_cov", "band_view_cov")}
        rec[f"unit_ratio_P16_over_A_{c}"] = ratio.tolist()
    Path(args.out).write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k.startswith("spearman")}, indent=1))
    for r in rec["units"]:
        print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
