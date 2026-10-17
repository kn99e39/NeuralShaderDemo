"""Architecture Batch 3 (worklog 36): Lego surface ownership for the worklog-32 owned-unit RNA (RNA venv).

* LegoOwner: the surface point is first placed on the mesh triangle it lies on (candidates = the 8
  triangles with nearest centroids plus the triangles of the 8 nearest of 8 M area-uniform auxiliary
  surface points, seed 0, ~1.8 mm spacing - centroids alone miss large planar triangles; exact point-to-triangle distance, rso_partition._closest_point_tri,
  restricted to triangles whose geometric normal agrees with the sample's shading normal,
  |n_t . n| >= 0.3, when any candidate does), so a point between two stacked surfaces goes to the
  surface it was rendered on; its unit is the k-means cell (nearest partition centre) of that
  on-surface point, and its connected component is the triangle's.
* LegoOwnedSurfaceDataset: the upstream dataset + owner id appended to every normalised position,
  exactly as worklog-32 OwnedSurfaceDataset, but owners use positions AND normals and are cached per
  (H5, partition) so every arm and evaluation uses identical owners.
The owned model itself is unchanged: rso_model.OwnedTriplaneSet / OwnedSurfaceTriplaneModule.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rna_surface_ownership"))
import rso_model as M  # noqa: E402  (puts upstream RNA on sys.path)
from rso_partition import _closest_point_tri  # noqa: E402

K_CAND = 8      # nearest triangle centroids
K_AUX = 8       # + triangles of the nearest auxiliary on-surface points (large triangles)
N_AUX = 8_000_000
COS_MIN = 0.3
CHUNK = 100_000


class LegoOwner:
    def __init__(self, partition_npz, mesh_npz):
        from scipy.spatial import cKDTree

        p = np.load(partition_npz); m = np.load(mesh_npz)
        self.centres = p["centres"].astype(np.float64)
        self.units = int(p["units"])
        self.tri_comp = m["comp"].astype(np.int64)
        V, F = m["V"], m["F"]
        self.a, self.b, self.c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        n = np.cross(self.b - self.a, self.c - self.a)
        self.tn = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
        self.tree = cKDTree((self.a + self.b + self.c) / 3)
        area = 0.5 * np.linalg.norm(n, axis=1)
        rng = np.random.default_rng(0)
        n_aux = min(N_AUX, 100 * len(F))
        t = rng.choice(len(F), n_aux, p=area / area.sum())
        u, v = rng.random(n_aux), rng.random(n_aux); fl = u + v > 1; u[fl], v[fl] = 1 - u[fl], 1 - v[fl]
        self.aux_tri = t
        self.aux_tree = cKDTree(self.a[t] + (self.b[t] - self.a[t]) * u[:, None] + (self.c[t] - self.a[t]) * v[:, None])
        self.key = hashlib.sha256(self.centres.tobytes()).hexdigest()[:12]

    def __call__(self, world, normals=None, return_tri=False):
        """-> (unit, distance to the surface) [, (triangle, component, on-surface point)]."""
        lab = np.zeros(len(world), np.int64); dist = np.zeros(len(world)); tri = np.zeros(len(world), np.int64)
        onsurf = np.zeros((len(world), 3))
        for s in range(0, len(world), CHUNK):
            w = world[s:s + CHUNK]
            kc, ka = min(K_CAND, len(self.a)), min(K_AUX, len(self.aux_tri))
            _, i1 = self.tree.query(w, k=kc, workers=-1)
            _, i2 = self.aux_tree.query(w, k=ka, workers=-1)
            ids = np.concatenate([i1.reshape(len(w), -1), self.aux_tri[i2.reshape(len(w), -1)]], 1)
            q = _closest_point_tri(w[:, None, :], self.a[ids], self.b[ids], self.c[ids])
            d = np.linalg.norm(q - w[:, None, :], axis=-1)
            if normals is not None:
                nr = normals[s:s + CHUNK]
                ok = np.abs((self.tn[ids] * nr[:, None, :]).sum(-1)) >= COS_MIN
                d_sel = np.where(ok, d, d + 1e3)  # prefer normal-compatible candidates when any exists
            else:
                d_sel = d
            j = d_sel.argmin(1)
            r = np.arange(len(w))
            t = ids[r, j]; qs = q[r, j]
            tri[s:s + CHUNK] = t; dist[s:s + CHUNK] = d[r, j]; onsurf[s:s + CHUNK] = qs
            lab[s:s + CHUNK] = ((qs[:, None, :] - self.centres[None]) ** 2).sum(-1).argmin(1)
        return (lab, dist, (tri, self.tri_comp[tri], onsurf)) if return_tri else (lab, dist)


def cached_owners(owner, h5_path, positions_world, normals, slice_index):
    """Owner labels for one H5 slice, cached next to the H5 (keyed by the partition hash)."""
    cache = Path(str(h5_path) + f".owners_{owner.key}")
    cache.mkdir(exist_ok=True)
    f = cache / f"{slice_index:04d}.npz"
    if f.exists():
        z = np.load(f)
        if len(z["lab"]) == len(positions_world):
            return z["lab"], z["dist"]
    lab, dist = owner(positions_world, normals)
    np.savez_compressed(f, lab=lab.astype(np.int16), dist=dist.astype(np.float32))
    return lab, dist


class LegoOwnedSurfaceDataset(M.rdatasets.NeuralSurfaceDataset):
    def __init__(self, *args, owner=None, datafile=None, **kwargs):
        import torch

        super().__init__(*args, datafile=datafile, **kwargs)
        self.owner_counts = np.zeros(owner.units, np.int64)
        self.owner_max_dist = 0.0
        for i in range(self.total_slices):
            p = self.positions[i]
            world = (p.to(self.ops_device) * (self.aabb_max - self.aabb_min) + self.aabb_min).cpu().numpy().astype(np.float64)
            nrm = self.normals[i].cpu().numpy().astype(np.float64)
            lab, d = cached_owners(owner, datafile, world, nrm, i)
            self.owner_counts += np.bincount(lab, minlength=owner.units)
            self.owner_max_dist = max(self.owner_max_dist, float(d.max()) if len(d) else 0.0)
            self.positions[i] = torch.cat([p, torch.tensor(lab, dtype=p.dtype, device=p.device)[:, None]], -1)


class LegoOwnedDataModule(M.rinterfaces.NeuralSurfaceDataModule):
    def __init__(self, conf, owner=None, **kw):
        super().__init__(conf, **kw)
        self.owner = owner

    def setup(self, stage=None):
        mk = lambda f: LegoOwnedSurfaceDataset(datafile=f, use_for_visibility=self.use_for_visibility, use_positions=True,
                                               device=self.device, ops_device=self.ops_device, owner=self.owner)
        self.trainset = mk(self.conf.training_dataset)
        self.valset = mk(self.conf.validation_dataset) if self.conf.validation_dataset else None
