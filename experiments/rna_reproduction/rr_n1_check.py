"""Worklog-35 N1 sanity: the seeded 1024/4096-spp validation sets share cameras and lights; target-to-target PSNR."""
import json
import sys

import h5py
import numpy as np

a = h5py.File(sys.argv[1], "r"); b = h5py.File(sys.argv[2], "r")
rec = {k: float(np.abs(a[k][:].astype(np.float32) - b[k][:].astype(np.float32)).max()) for k in ("light_dir", "camera_dir", "position", "alpha")}
c1 = a["color"][:].astype(np.float32); c4 = b["color"][:].astype(np.float32); m = a["alpha"][:, :, 0] > 0
t = lambda x: (np.clip(x, 0, None) / (1 + np.clip(x, 0, None))) ** (1 / 2.2)
rec["psnr_1024_vs_4096_masked_pooled"] = float(-10 * np.log10(((t(c1) - t(c4)) ** 2)[m].mean()))
print(json.dumps(rec, indent=1))
