"""Worklog 36: Lego pixel footprint (median 3D distance between full-coverage 4-neighbour hits) on the dev views."""
import json
import sys

import h5py
import numpy as np

f = h5py.File(sys.argv[1], "r")
res = int(f.attrs["resolution"][0]); out = []
for v in range(f["color"].shape[0]):
    a = f["alpha"][v][:, 0].astype(np.float32) >= 0.999
    P = f["position"][v].astype(np.float32).reshape(res, res, 3); A = a.reshape(res, res)
    d = []
    for dy, dx in ((0, 1), (1, 0)):
        ok = A[: res - dy, : res - dx] & A[dy:, dx:]
        d.append(np.linalg.norm(P[: res - dy, : res - dx] - P[dy:, dx:], axis=-1)[ok])
    out.append(float(np.median(np.concatenate(d))))
print(json.dumps({"per_view_median_footprint": out, "median": float(np.median(out))}))
