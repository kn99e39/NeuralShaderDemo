"""Arm C on Lego: worklog-34 partition-of-unity composition over unchanged owned units (worklog 36).

Contract (protocol/b3_v1.json):
  * learned state unchanged (an rso_model.OwnedTriplaneSet); no trainable or learned shared state added;
  * output(x) = (1 - w) f_own(x) + w f_nbr(x), w = clip(0.5 - s / (2h), 0, 0.5), where own = the point's
    unit, nbr = the other unit whose cut is nearest, considering only units present in the point's
    connected component (blending only across an ownership cut through continuous surface), and
    s = exact Euclidean distance from the on-surface point to the Voronoi bisector of the two units'
    k-means centres (transfer adaptation: worklog 34 approximated s with dense samples, which cannot
    resolve h = 4.5 mm at Lego's ~5 mm sample spacing);
  * outside the band (s >= h, or no other unit in the component) the output is f_own exactly;
  * output influence of unit i: S_i and B_i = {x not in S_i : nbr(x) = i, w(x) > 0}.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rna_surface_ownership_b2"))
from b2_compose import pu_weight  # noqa: E402  (the worklog-34 weight function, unchanged)


class LegoPUComposer:
    def __init__(self, owned_set, centres, sample_label, sample_comp, h):
        self.model = owned_set
        self.c = np.asarray(centres, np.float64)
        self.K = len(self.c)
        self.h = float(h)
        ncomp = int(sample_comp.max()) + 1
        self.comp_units = np.zeros((ncomp + 1, self.K), bool)  # last row: unknown component -> no blending
        self.comp_units[sample_comp, sample_label] = True

    @classmethod
    def from_partition(cls, owned_set, partition_npz, h):
        z = np.load(partition_npz)
        return cls(owned_set, z["centres"], z["label"], z["comp"], h)

    def influence(self, onsurf, owner, comp):
        """-> (nbr, w, s): nearest same-component ownership cut; nbr = -1, w = 0 where none exists."""
        comp = np.where((comp >= 0) & (comp < len(self.comp_units) - 1), comp, len(self.comp_units) - 1)
        cu = self.c[owner]
        d2 = ((onsurf[:, None, :] - self.c[None]) ** 2).sum(-1)                  # (N, K)
        sep = np.linalg.norm(self.c[None] - cu[:, None, :], axis=-1)              # (N, K)
        s = (d2 - d2[np.arange(len(owner)), owner][:, None]) / np.maximum(2 * sep, 1e-12)
        allowed = self.comp_units[comp].copy(); allowed[np.arange(len(owner)), owner] = False
        s = np.where(allowed, s, np.inf)
        nbr = s.argmin(1); smin = s[np.arange(len(owner)), nbr]
        has = np.isfinite(smin)
        nbr = np.where(has, nbr, -1)
        w = np.where(has, pu_weight(np.zeros(len(owner)), np.where(has, smin, 1e9), self.h), 0.0)
        return nbr, w, np.where(has, smin, np.inf)

    def eval_unit(self, k, pos_w, cam, light, nrm):
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
        out = []
        with torch.no_grad():
            for s in range(0, len(a[0]), 1 << 20):
                out.append(self.model.units[k](*(T(x[s:s + (1 << 20)]) for x in a)).squeeze(0).cpu().numpy())
        o = np.concatenate(out)
        return o[:1] if dup else o

    def __call__(self, pos_w, cam, light, nrm, vis, owner, comp, onsurf, return_parts=False):
        n = len(pos_w)
        raw_own = np.zeros((n, 6), np.float32)
        for u in np.unique(owner):
            s = np.flatnonzero(owner == u)
            raw_own[s] = self.eval_unit(int(u), pos_w[s], cam[s], light[s], nrm[s])
        nbr, w, _ = self.influence(onsurf, owner, comp)
        band = np.flatnonzero(w > 0)
        raw_nbr = np.zeros((n, 6), np.float32)
        for v in np.unique(nbr[band]):
            s = band[nbr[band] == v]
            raw_nbr[s] = self.eval_unit(int(v), pos_w[s], cam[s], light[s], nrm[s])
        sel = lambda r: np.where(vis[:, None], r[:, :3], r[:, 3:])
        hard = sel(raw_own)
        out = hard.copy()
        out[band] = ((1 - w[band, None]) * hard[band] + w[band, None] * sel(raw_nbr)[band]).astype(np.float32)
        return (out, {"hard": hard, "nbr": nbr, "w": w}) if return_parts else out

    def influenced_by(self, k, owner, comp, onsurf):
        nbr, w, _ = self.influence(onsurf, owner, comp)
        return (owner == k) | ((nbr == k) & (w > 0))
