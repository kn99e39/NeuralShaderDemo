"""Surface-owned explicit radiance units (worklog 31).

Representation (the one architecture candidate of this batch):

  L_ind(x, w) = sum_k  wgt_k(x) * table_{a_k}(oct(R_{a_k}^T w))

  * a *unit* (anchor) a is a surface location on ONE rigid part, stored in that part's local
    frame (so it moves with the part); it owns an explicit RGB table over the full sphere of
    outgoing directions (octahedral map in the anchor's normal frame: upper hemisphere = the
    front side, lower = the back side of two-sided surfaces);
  * wgt_k(x) are partition-of-unity weights of the anchors near the query on the same part,
    with an orientation gate (anchors facing away from the query's geometric normal never
    contribute), Wendland C2 kernel of radius RADIUS_H * h;
  * the stored quantity is outgoing radiance of paths with >= 2 asset vertices (the same
    transport scope 8DNA learns); direct light is computed physically from current geometry.

There is no learned shared decoder: the output is linear in the table entries, and an entry
is radiance.  The only shared computation is fixed (anchor search, interpolation, frames).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPL = ROOT / "experiments" / "8dna_deformation_replication"
NRC = ROOT / "experiments" / "neural_recompute_cost"
SRO = ROOT / "experiments" / "selective_refit_oracle"
for p in (HERE, REPL, NRC, SRO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

RESULTS = ROOT / "results" / "surface_owned_radiance"
PROTOCOL_PATH = HERE / "protocol" / "sor_v1.json"
PARTS = ("plate", "teapot2", "teapot3", "teapot4")
MOVER = "teapot2"
RADIUS_H = 1.5        # kernel radius in units of the anchor spacing h
NORMAL_GATE = 0.3     # min dot(query geometric normal, anchor normal)
K_MAX = 16            # max anchors per query stencil
ORIENT_BINS = 6       # anchors per voxel: one per dominant normal axis/sign


def init():
    import ednalib as L

    return L.init_upstream()


def states() -> dict:
    return json.loads((REPL / "protocol" / "teaset_frozen_locked.json").read_text(encoding="utf-8"))["states"]


def protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def write_json(path, payload) -> None:
    import nrc_records as R

    R.write_json(path, payload)


# ----------------------------------------------------------------------------- geometry helpers

_GEOM_CACHE: dict = {}


def part_geometry(part: str):
    """Canonical (part-local) vertices, faces, oriented face normals, face areas (float64 numpy; cached)."""
    if part in _GEOM_CACHE:
        return _GEOM_CACHE[part]
    import numpy as np
    import teaset_parts as T

    v, f = T.part_mesh(part, None)
    t = v[f]
    cr = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    area = 0.5 * np.linalg.norm(cr, axis=1)
    n = cr / np.maximum(np.linalg.norm(cr, axis=1, keepdims=True), 1e-30)
    _GEOM_CACHE[part] = (v, f, n, area)
    return v, f, n, area


class SurfaceSampler:
    """GPU area-uniform sampling over the four parts (canonical frame), with cached triangles."""

    def __init__(self, device="cuda"):
        import numpy as np
        import torch

        tri, nrm, area, part = [], [], [], []
        for i, p in enumerate(PARTS):
            v, f, n, a = part_geometry(p)
            tri.append(v[f]); nrm.append(n); area.append(a); part.append(np.full(len(f), i))
        self.tri = torch.tensor(np.concatenate(tri), dtype=torch.float32, device=device)
        self.nrm = torch.tensor(np.concatenate(nrm), dtype=torch.float32, device=device)
        a = torch.tensor(np.concatenate(area), dtype=torch.float64, device=device)
        self.cdf = torch.cumsum(a, 0) / a.sum()
        self.part = torch.tensor(np.concatenate(part), dtype=torch.long, device=device)
        self.device = device

    def positions(self, n: int, gen):
        import torch

        u = torch.rand(n, device=self.device, generator=gen, dtype=torch.float64)
        k = torch.searchsorted(self.cdf, u).clamp(max=len(self.cdf) - 1)
        a, b = torch.rand(n, device=self.device, generator=gen), torch.rand(n, device=self.device, generator=gen)
        flip = a + b > 1
        a, b = torch.where(flip, 1 - a, a), torch.where(flip, 1 - b, b)
        t = self.tri[k]
        x = t[:, 0] + a[:, None] * (t[:, 1] - t[:, 0]) + b[:, None] * (t[:, 2] - t[:, 0])
        return x, self.nrm[k], self.part[k]


def sample_surface(part: str, n: int, rng):
    """Area-uniform samples on a part (canonical frame): positions, face normals, face ids."""
    import numpy as np

    v, f, nrm, area = part_geometry(part)
    k = rng.choice(len(f), n, p=area / area.sum())
    u, w = rng.random(n), rng.random(n)
    flip = u + w > 1
    u[flip], w[flip] = 1 - u[flip], 1 - w[flip]
    t = v[f[k]]
    x = t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + w[:, None] * (t[:, 2] - t[:, 0])
    return x, nrm[k], k


def frames(n):
    """Orthonormal (t, b) for unit normals n (torch), Duff et al. 2017."""
    import torch

    sign = torch.where(n[:, 2] >= 0, 1.0, -1.0)
    a = -1.0 / (sign + n[:, 2])
    b = n[:, 0] * n[:, 1] * a
    t = torch.stack([1 + sign * n[:, 0] ** 2 * a, sign * b, -sign * n[:, 0]], -1)
    bb = torch.stack([b, sign + n[:, 1] ** 2 * a, -n[:, 1]], -1)
    return t, bb


def oct_encode(v):
    """Unit vectors (local frame, z = normal) -> octahedral uv in [0,1]^2 (full sphere)."""
    import torch

    p = v[:, :2] / (v.abs().sum(-1, keepdim=True) + 1e-12)
    neg = v[:, 2] < 0
    q = (1 - p.flip(-1).abs()) * torch.where(p >= 0, 1.0, -1.0)
    p = torch.where(neg[:, None], q, p)
    return p * 0.5 + 0.5


def oct_decode(uv):
    """Inverse of oct_encode (torch), for texel-centre directions."""
    import torch

    p = uv * 2 - 1
    z = 1 - p.abs().sum(-1)
    t = (-z).clamp(min=0)
    x = p[:, 0] + torch.where(p[:, 0] >= 0, -t, t)
    y = p[:, 1] + torch.where(p[:, 1] >= 0, -t, t)
    v = torch.stack([x, y, z], -1)
    return v / v.norm(dim=-1, keepdim=True)


# ----------------------------------------------------------------------------- anchors (units)

def build_anchors(h: float, samples_per_cell: float = 24.0, seed: int = 0) -> dict:
    """Units: per part, one anchor per (voxel of size h, dominant-normal bin) occupied by the surface.

    Returns numpy arrays: part index, canonical position, normal; and per-part voxel grids
    (origin, dims, cell -> up to ORIENT_BINS anchor ids)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    pos, nrm, prt, areas = [], [], [], []
    grids = {}
    base = 0
    for pi, part in enumerate(PARTS):
        v, f, fn, area = part_geometry(part)
        n_s = int(math.ceil(area.sum() / h ** 2 * samples_per_cell))
        x, n, _ = sample_surface(part, n_s, rng)
        lo = v.min(0) - 2 * h
        dims = np.ceil((v.max(0) + 2 * h - lo) / h).astype(int) + 1
        cell = np.floor((x - lo) / h).astype(np.int64)
        flat = (cell[:, 0] * dims[1] + cell[:, 1]) * dims[2] + cell[:, 2]
        ax = np.abs(n).argmax(1)
        sgn = (n[np.arange(len(n)), ax] < 0).astype(np.int64)
        obin = ax * 2 + sgn
        key = flat * ORIENT_BINS + obin
        order = np.argsort(key, kind="stable")
        key_s = key[order]
        starts = np.r_[0, np.flatnonzero(np.diff(key_s)) + 1]
        ends = np.r_[starts[1:], len(key_s)]
        a_pos, a_nrm, a_key, a_area = [], [], [], []
        per_sample = area.sum() / n_s
        for s, e in zip(starts, ends):
            idx = order[s:e]
            m = x[idx].mean(0)
            j = idx[np.argmin(np.linalg.norm(x[idx] - m, axis=1))]
            nn = n[idx].mean(0)
            nn /= max(np.linalg.norm(nn), 1e-12)
            a_pos.append(x[j]); a_nrm.append(nn); a_key.append(key_s[s]); a_area.append((e - s) * per_sample)
        a_pos, a_nrm, a_key = np.array(a_pos), np.array(a_nrm), np.array(a_key)
        areas.append(np.array(a_area))
        table = -np.ones((int(np.prod(dims)), ORIENT_BINS), np.int64)
        table[a_key // ORIENT_BINS, a_key % ORIENT_BINS] = base + np.arange(len(a_key))
        grids[part] = {"origin": lo, "dims": dims, "cells": table, "first": base, "count": len(a_key)}
        pos.append(a_pos); nrm.append(a_nrm); prt.append(np.full(len(a_key), pi))
        base += len(a_key)
    return {"h": h, "pos": np.concatenate(pos), "nrm": np.concatenate(nrm), "part": np.concatenate(prt),
            "area": np.concatenate(areas), "grids": grids, "samples_per_cell": samples_per_cell, "seed": seed}


def save_units(path, a: dict) -> None:
    import numpy as np

    extra = {}
    for part, g in a["grids"].items():
        for k in ("origin", "dims", "cells"):
            extra[f"grid_{part}_{k}"] = g[k]
        extra[f"grid_{part}_first"] = g["first"]; extra[f"grid_{part}_count"] = g["count"]
    np.savez_compressed(path, h=a["h"], pos=a["pos"], nrm=a["nrm"], part=a["part"], area=a["area"],
                        samples_per_cell=a["samples_per_cell"], seed=a["seed"], **extra)


def load_units(path) -> dict:
    import numpy as np

    z = np.load(path)
    grids = {p: {"origin": z[f"grid_{p}_origin"], "dims": z[f"grid_{p}_dims"], "cells": z[f"grid_{p}_cells"],
                 "first": int(z[f"grid_{p}_first"]), "count": int(z[f"grid_{p}_count"])} for p in PARTS}
    return {"h": float(z["h"]), "pos": z["pos"], "nrm": z["nrm"], "part": z["part"], "area": z["area"],
            "samples_per_cell": float(z["samples_per_cell"]), "seed": int(z["seed"]), "grids": grids}


def unit_areas(a: dict):
    return a["area"]


class SurfaceRadiance:
    """Explicit surface-owned radiance units with octahedral tables (torch, CUDA)."""

    def __init__(self, anchors: dict, res: int, device="cuda"):
        import torch

        self.h = anchors["h"]
        self.res = res
        self.device = device
        self.pos = torch.tensor(anchors["pos"], dtype=torch.float32, device=device)
        self.nrm = torch.tensor(anchors["nrm"], dtype=torch.float32, device=device)
        self.part = torch.tensor(anchors["part"], dtype=torch.long, device=device)
        self.t, self.b = frames(self.nrm)
        self.grids = {}
        for part, g in anchors["grids"].items():
            self.grids[part] = {"origin": torch.tensor(g["origin"], dtype=torch.float32, device=device),
                                "dims": torch.tensor(g["dims"], dtype=torch.long, device=device),
                                "cells": torch.tensor(g["cells"], dtype=torch.long, device=device)}
        self.n_units = len(anchors["pos"])
        self.table = torch.zeros(self.n_units * res * res, 3, device=device)

    # -- stencil: which units a query reads, with weights (fixed computation, no learned params)
    def stencil(self, x_local, part_idx, n_geo):
        """x_local (N,3) canonical part-frame positions; part_idx (N,) index into PARTS;
        n_geo (N,3) oriented geometric normals.  Returns ids (N,K_MAX) (-1 = none), w (N,K_MAX)."""
        import torch

        N = len(x_local)
        ids = torch.full((N, K_MAX), -1, dtype=torch.long, device=self.device)
        wts = torch.zeros((N, K_MAX), device=self.device)
        r = RADIUS_H * self.h
        offs = torch.stack(torch.meshgrid(*(torch.arange(-1, 2, device=self.device),) * 3, indexing="ij"), -1).reshape(-1, 3)
        for pi, part in enumerate(PARTS):
            sel = torch.nonzero(part_idx == pi).squeeze(1)
            if len(sel) == 0:
                continue
            g = self.grids[part]
            c = torch.floor((x_local[sel] - g["origin"]) / self.h).long()
            cand = []
            for o in offs:
                cc = c + o
                ok = ((cc >= 0) & (cc < g["dims"])).all(1)
                flat = (cc[:, 0] * g["dims"][1] + cc[:, 1]) * g["dims"][2] + cc[:, 2]
                flat = torch.where(ok, flat, torch.zeros_like(flat))
                a = g["cells"][flat]
                a = torch.where(ok[:, None], a, torch.full_like(a, -1))
                cand.append(a)
            cand = torch.cat(cand, 1)                                   # (n, 27*bins)
            valid = cand >= 0
            ci = cand.clamp(min=0)
            d = (self.pos[ci] - x_local[sel, None, :]).norm(dim=-1)
            q = (d / r).clamp(max=1.0)
            w = (1 - q) ** 4 * (4 * q + 1)
            gate = (self.nrm[ci] * n_geo[sel, None, :]).sum(-1) > NORMAL_GATE
            w = torch.where(valid & gate & (d < r), w, torch.zeros_like(w))
            top_w, top_i = torch.topk(w, K_MAX, dim=1)
            top_id = torch.gather(cand, 1, top_i)
            top_id = torch.where(top_w > 0, top_id, torch.full_like(top_id, -1))
            ids[sel] = top_id
            wts[sel] = top_w
        s = wts.sum(1, keepdim=True)
        wts = torch.where(s > 0, wts / s.clamp(min=1e-30), wts)
        return ids, wts

    def texel_stencil(self, ids, wts, w_world):
        """Flat table rows (N, K*4) and bilinear x kernel weights (N, K*4) for directions w_world."""
        import torch

        N, K = ids.shape
        R = self.res
        ci = ids.clamp(min=0)
        t, b, n = self.t[ci], self.b[ci], self.nrm[ci]                  # (N,K,3)
        wl = torch.stack([(w_world[:, None, :] * t).sum(-1), (w_world[:, None, :] * b).sum(-1),
                          (w_world[:, None, :] * n).sum(-1)], -1).reshape(-1, 3)
        uv = oct_encode(wl).reshape(N, K, 2) * R - 0.5
        i0 = torch.floor(uv).long()
        f = uv - i0
        rows, ws = [], []
        for dy in (0, 1):
            for dx in (0, 1):
                ix = (i0[..., 0] + dx).clamp(0, R - 1)
                iy = (i0[..., 1] + dy).clamp(0, R - 1)
                bw = (f[..., 0] if dx else 1 - f[..., 0]) * (f[..., 1] if dy else 1 - f[..., 1])
                rows.append(ci * R * R + iy * R + ix)
                ws.append(wts * bw)
        rows = torch.stack(rows, -1).reshape(N, K * 4)
        ws = torch.stack(ws, -1).reshape(N, K * 4)
        ws = torch.where(ids.repeat_interleave(4, 1) >= 0, ws, torch.zeros_like(ws))
        return rows, ws

    def query(self, rows, ws, table=None):
        tab = self.table if table is None else table
        return (tab[rows] * ws[..., None]).sum(1)

    def state(self) -> dict:
        return {"table": self.table.detach().cpu(), "res": self.res, "h": self.h, "n_units": self.n_units}
