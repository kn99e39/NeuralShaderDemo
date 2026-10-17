"""Architecture Batch 3 (worklog 36): Surface-Owned Transport Units on the official Lego mesh (numpy/scipy only).

    python b3_partition.py --mesh <lego_mesh.npz> --units 16 --out <partition_L16.npz>

Partition contract (fixed before any candidate was trained or evaluated; protocol/b3_v1.json):
  * surface identity comes from the mesh TRIANGLE a point lies on (b3_model.LegoOwner: nearest
    normal-compatible triangle), so a rendered point is first placed on the surface it was rendered on;
  * K = 16 units (correspondence with worklog 32/34 P16): area-weighted k-means (seed 0, 50 Lloyd
    iterations, rso_partition.kmeans) of 1 M area-uniform surface samples; the unit of a surface point
    is the k-means cell (nearest centre) of that on-surface point. Units are therefore compact surface
    regions; a cell boundary cutting through one brick is an ownership cut through continuous surface,
    exactly like the worklog-32 k-means cuts inside a part. (A first variant assigned whole triangles
    by centroid; large planar triangles inflated unit AABBs to the full asset length, so it was replaced
    before any training - recorded in the worklog.);
  * the geometry-only partition never sees renders, errors or validation data;
  * per sample, its connected-component id (shared vertices) is stored - the PU composition blends only
    across ownership cuts inside one component (continuous surface), as worklog 34 blended only inside
    one part.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rna_surface_ownership"))
from rso_partition import kmeans  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--units", type=int, default=16)
    ap.add_argument("--samples", type=int, default=1_000_000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    z = np.load(args.mesh)
    V, F, comp, area, mat = z["V"], z["F"], z["comp"], z["area"], z["material"]
    rng = np.random.default_rng(0)
    t = rng.choice(len(F), args.samples, p=area / area.sum())
    u, v = rng.random(args.samples), rng.random(args.samples)
    flip = u + v > 1; u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    a, b, c = V[F[t, 0]], V[F[t, 1]], V[F[t, 2]]
    x = a + (b - a) * u[:, None] + (c - a) * v[:, None]
    lab, centres = kmeans(x, args.units, seed=0)
    K = args.units
    lo = np.stack([x[lab == k].min(0) for k in range(K)]); hi = np.stack([x[lab == k].max(0) for k in range(K)])
    unit_area = np.bincount(lab, minlength=K) * area.sum() / len(x)
    sc = comp[t]; sm = mat[t]
    unit_comps = [int(np.unique(sc[lab == k]).size) for k in range(K)]
    per_comp_units = {}
    for q, k in zip(sc, lab):
        per_comp_units.setdefault(int(q), set()).add(int(k))
    split_comps = int(sum(len(v) > 1 for v in per_comp_units.values()))
    np.savez_compressed(args.out, x=x.astype(np.float32), label=lab.astype(np.int64), comp=sc.astype(np.int64),
                        centres=centres, units=np.int64(K), unit_lo=lo, unit_hi=hi, unit_area=unit_area,
                        unit_part=np.zeros(K, np.int64))
    summ = {"units": K, "unit_area": unit_area.round(4).tolist(), "unit_samples": np.bincount(lab, minlength=K).tolist(),
            "unit_components": unit_comps, "components_split_across_units": split_comps, "components_sampled": len(per_comp_units),
            "unit_extent": (hi - lo).round(3).tolist(),
            "unit_material_area": {int(k): {int(m): round(float((sm[lab == k] == m).sum() * area.sum() / len(x)), 4) for m in np.unique(mat)} for k in range(K)}}
    Path(args.out).with_suffix(".json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != "unit_material_area"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
