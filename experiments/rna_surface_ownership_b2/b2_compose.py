"""P16-PU: explicit partition-of-unity boundary composition over unchanged owned units (worklog 34).

Contract (protocol/b2_v1.json, amendment 1)
  * learned state: unchanged - every unit k keeps exactly its own Theta_k (an rso_model.OwnedTriplaneSet);
    this module adds NO trainable and NO learned shared state;
  * composition data: the partition's dense surface samples (fixed geometry) and one length h;
  * output(x) = (1 - w) f_own(x) + w f_nbr(x), own = owner(x) (unchanged ownership rule),
    nbr = nearest OTHER unit of the same part, w = clip(0.5 - (d_nbr - d_own) / (2h), 0, 0.5)
    with d_u = distance from x to unit u's nearest dense sample; w = 0 (output = f_own exactly) once
    d_nbr - d_own >= h, and w -> 0.5 on both sides of the seam, so the composite is continuous there;
  * output influence: unit i affects S_i and B_i = {x not in S_i : nbr(x) = i, w(x) > 0}.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rna_surface_ownership"))


def pu_weight(d_own, d_nbr, h):
    return np.clip(0.5 - (np.asarray(d_nbr) - np.asarray(d_own)) / (2.0 * h), 0.0, 0.5)


class PUComposer:
    """Composition over an OwnedTriplaneSet. samples (S,3), labels (S,), unit_part (K,) define the
    partition geometry; h is the band half-width in world units."""

    def __init__(self, owned_set, samples, labels, unit_part, h):
        from scipy.spatial import cKDTree

        self.model = owned_set
        self.h = float(h)
        self.unit_part = np.asarray(unit_part)
        self.K = len(self.unit_part)
        self.trees = {u: cKDTree(np.asarray(samples)[np.asarray(labels) == u]) for u in range(self.K)}

    @classmethod
    def from_partition(cls, owned_set, partition_npz, h):
        z = np.load(partition_npz)
        return cls(owned_set, z["x"], z["label"], z["unit_part"], h)

    def influence(self, pos_w, owner):
        """-> (nbr, w, d_own, d_nbr); nbr = -1 and w = 0 where the owner's part has no other unit."""
        n = len(pos_w)
        d_own = np.zeros(n); d_nbr = np.full(n, np.inf); nbr = -np.ones(n, np.int64)
        for u in np.unique(owner):
            s = np.flatnonzero(owner == u)
            d_own[s] = self.trees[int(u)].query(pos_w[s])[0]
            for v in range(self.K):
                if v == u or self.unit_part[v] != self.unit_part[u]:
                    continue
                d = self.trees[v].query(pos_w[s])[0]
                better = d < d_nbr[s]
                d_nbr[s[better]] = d[better]; nbr[s[better]] = v
        w = np.where(nbr >= 0, pu_weight(d_own, np.where(np.isfinite(d_nbr), d_nbr, d_own + 1e9), self.h), 0.0)
        return nbr, w, d_own, d_nbr

    def eval_unit(self, k, pos_w, cam, light, nrm):
        """Unit k's raw 6-channel output at world points (local frame clamped exactly as OwnedTriplaneSet.forward)."""
        import torch

        if len(pos_w) == 0:
            return np.zeros((0, 6), np.float32)
        dev = self.model.lo.device
        lo, hi = self.model.lo[k].cpu().numpy(), self.model.hi[k].cpu().numpy()
        local = np.clip((pos_w - lo) / (hi - lo), 0, 1)
        a = [local, cam, light, nrm]
        dup = len(local) == 1
        if dup:
            a = [np.repeat(x, 2, 0) for x in a]
        T = lambda x: torch.tensor(np.asarray(x, np.float32), device=dev).unsqueeze(0)
        with torch.no_grad():
            o = self.model.units[k](*(T(x) for x in a)).squeeze(0).cpu().numpy()
        return o[:1] if dup else o

    def __call__(self, pos_w, cam, light, nrm, vis, owner, return_parts=False):
        """Composite RGB (lit/shadowed branch chosen by vis, as RNA) at world points with their owners."""
        n = len(pos_w)
        raw_own = np.zeros((n, 6), np.float32)
        for u in np.unique(owner):
            s = np.flatnonzero(owner == u)
            raw_own[s] = self.eval_unit(int(u), pos_w[s], cam[s], light[s], nrm[s])
        nbr, w, _, _ = self.influence(pos_w, owner)
        raw_nbr = np.zeros((n, 6), np.float32)
        band = np.flatnonzero(w > 0)
        for v in np.unique(nbr[band]):
            s = band[nbr[band] == v]
            raw_nbr[s] = self.eval_unit(int(v), pos_w[s], cam[s], light[s], nrm[s])
        sel = lambda r: np.where(vis[:, None], r[:, :3], r[:, 3:])
        hard = sel(raw_own)
        out = hard.copy()
        out[band] = ((1 - w[band, None]) * hard[band] + w[band, None] * sel(raw_nbr)[band]).astype(np.float32)
        if return_parts:
            return out, {"hard": hard, "nbr": nbr, "w": w}
        return out

    def influenced_by(self, k, pos_w, owner):
        """Boolean mask of query points whose composite output can change when unit k changes."""
        nbr, w, _, _ = self.influence(pos_w, owner)
        return (owner == k) | ((nbr == k) & (w > 0))
