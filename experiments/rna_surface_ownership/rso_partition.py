"""Surface ownership partitions of the teaset (numpy/scipy only; runs in the RNA venv).

A partition assigns every surface point of the asset to exactly one transport unit:
  P1  : one unit (original RNA ownership)
  P4  : one unit per part (object-level ownership)
  P16 : surface domains inside parts: per part, k-means (fixed seed) over area-uniform surface
        samples, k proportional to part area (tray 9, tea pot 4, milk pot 2, tin 1)
Owner of a point = label of its nearest dense surface sample (all parts), so ownership is a pure
function of surface position (deterministic, complete, no overlaps).  Positions are the canonical
(T0) asset frame, i.e. the frame of the RNA H5 'position' channel.

    python rso_partition.py --out <dir>      (writes partition_<name>.npz + partitions.json)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TEASET = ROOT / "external" / "8dna26" / "scenes" / "teaset"
PARTS = {"plate": "teaplate.obj", "teapot2": "teapot2.obj", "teapot3": "teapot3.obj", "teapot4": "teapot4.obj"}
P16_K = {"plate": 9, "teapot4": 4, "teapot2": 2, "teapot3": 1}
DENSE = 1_000_000
SEED = 0


def read_obj(path: Path):
    v, f = [], []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
            for k in range(1, len(idx) - 1):
                f.append([idx[0], idx[k], idx[k + 1]])
    return np.array(v, np.float64), np.array(f, np.int64)


def surface_samples(n_total: int, seed: int):
    rng = np.random.default_rng(seed)
    meshes = {p: read_obj(TEASET / f) for p, f in PARTS.items()}
    areas = {}
    for p, (v, f) in meshes.items():
        t = v[f]
        areas[p] = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    tot = sum(a.sum() for a in areas.values())
    xs, parts = [], []
    for pi, (p, (v, f)) in enumerate(meshes.items()):
        n = int(round(n_total * areas[p].sum() / tot))
        k = rng.choice(len(f), n, p=areas[p] / areas[p].sum())
        a, b = rng.random(n), rng.random(n)
        flip = a + b > 1
        a[flip], b[flip] = 1 - a[flip], 1 - b[flip]
        t = v[f[k]]
        xs.append(t[:, 0] + a[:, None] * (t[:, 1] - t[:, 0]) + b[:, None] * (t[:, 2] - t[:, 0]))
        parts.append(np.full(n, pi))
    return np.concatenate(xs), np.concatenate(parts), {p: float(a.sum()) for p, a in areas.items()}


def kmeans(x, k, seed, iters=50):
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=False)]
    for _ in range(iters):
        d = ((x[:, None, :] - c[None]) ** 2).sum(-1) if len(x) * k < 4e7 else None
        if d is None:
            lab = np.concatenate([((x[i:i + 200000, None, :] - c[None]) ** 2).sum(-1).argmin(1) for i in range(0, len(x), 200000)])
        else:
            lab = d.argmin(1)
        c_new = np.stack([x[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
        if np.allclose(c_new, c):
            break
        c = c_new
    return lab, c


def build(name: str):
    x, part, areas = surface_samples(DENSE, SEED)
    if name == "P1":
        label = np.zeros(len(x), np.int64)
    elif name == "P4":
        label = part.copy()
    elif name == "P16":
        label = np.zeros(len(x), np.int64)
        base = 0
        for pi, p in enumerate(PARTS):
            m = part == pi
            lab, _ = kmeans(x[m], P16_K[p], SEED + pi)
            label[m] = base + lab
            base += P16_K[p]
    else:
        raise ValueError(name)
    units = int(label.max() + 1)
    unit_part = np.array([int(np.bincount(part[label == u]).argmax()) for u in range(units)])
    lo = np.stack([x[label == u].min(0) for u in range(units)])
    hi = np.stack([x[label == u].max(0) for u in range(units)])
    return {"name": name, "x": x.astype(np.float32), "label": label, "part": part, "units": units,
            "unit_part": unit_part, "unit_lo": lo, "unit_hi": hi, "areas": areas,
            "unit_area": np.array([areas[list(PARTS)[unit_part[u]]] * (label == u).sum() / (part == unit_part[u]).sum() for u in range(units)])}


class Owner:
    """Nearest-dense-sample owner lookup (scipy cKDTree)."""

    def __init__(self, npz):
        from scipy.spatial import cKDTree

        z = np.load(npz)
        self.x, self.label, self.part = z["x"], z["label"], z["part"]
        self.tree = cKDTree(self.x)
        self.units = int(z["units"])
        self.unit_lo, self.unit_hi = z["unit_lo"], z["unit_hi"]

    def __call__(self, pos):
        d, i = self.tree.query(np.asarray(pos, np.float64), workers=-1)
        return self.label[i], d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in ("P1", "P4", "P16"):
        b = build(name)
        np.savez_compressed(out / f"partition_{name}.npz", **{k: v for k, v in b.items() if k not in ("name", "areas")})
        summary[name] = {"units": b["units"], "unit_part": [list(PARTS)[i] for i in b["unit_part"]],
                         "unit_area": b["unit_area"].round(4).tolist(),
                         "unit_extent": (b["unit_hi"] - b["unit_lo"]).round(4).tolist()}
        print(name, b["units"], flush=True)
    (out / "partitions.json").write_text(json.dumps({"dense_samples": DENSE, "seed": SEED, "p16_k": P16_K, "partitions": summary}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
