"""Surface-owned RNA: K complete upstream RNA models, one per surface domain (runs in the RNA venv).

Ownership contract
  * unit k owns ONE complete upstream `rna.models.NeuralSurfaceTriPlane` (its own triplane, framed
    on the unit's domain AABB, and its own MLP); nothing learned is shared between units;
  * the owner of a surface point is fixed by the partition (rso_partition.Owner);
  * evaluation: every query is evaluated by its owner only (selection, never a sum);
  * update boundary: a unit's update touches only its own module's parameters.
Upstream RNA training/validation steps are reused unchanged: the dataset appends the owner id as a
fourth position channel and `OwnedTriplaneSet.forward` routes on it.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
RNA = ROOT / "external" / "relightable-neural-assets"
if str(RNA) not in sys.path:
    sys.path.insert(1, str(RNA))
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from rna import datasets as rdatasets  # noqa: E402
from rna import interfaces as rinterfaces  # noqa: E402
from rna import models as rmodels  # noqa: E402

AABB_MARGIN = 0.02  # fraction of the unit extent added on each side of a unit's AABB


def unit_specs(partition_npz, base_model_params: dict, dataset_aabb, matched: bool = True):
    """Per-unit model params.  matched=True: triplane texel size of the original (res scaled by the
    unit's largest extent relative to the dataset AABB's largest extent, clamped to [64, base res]),
    MLP width base/sqrt(K) so the summed MLP parameters match the original."""
    z = np.load(partition_npz)
    K = int(z["units"])
    lo, hi = z["unit_lo"].astype(np.float64), z["unit_hi"].astype(np.float64)
    ext = np.maximum(hi - lo, 1e-3)
    lo, hi = lo - AABB_MARGIN * ext, hi + AABB_MARGIN * ext
    dmin, dmax = (np.asarray(a, np.float64) for a in dataset_aabb)
    base_res = int(base_model_params["resolution"][0])
    base_w = int(base_model_params["mlp_hidden_layer_neurons"])
    specs = []
    for k in range(K):
        p = dict(base_model_params)
        if matched and K > 1:
            r = int(np.clip(round(base_res * (hi[k] - lo[k]).max() / (dmax - dmin).max()), 64, base_res))
            p["resolution"] = [r, r]
            p["mlp_hidden_layer_neurons"] = int(round(base_w / math.sqrt(K)))
        specs.append({"params": p, "lo": lo[k], "hi": hi[k]})
    return specs


class OwnedTriplaneSet(nn.Module):
    """K independently owned upstream RNA surface models + their domain frames."""

    def __init__(self, specs, dataset_aabb):
        super().__init__()
        self.units = nn.ModuleList([rmodels.NeuralSurfaceTriPlane(**s["params"]) for s in specs])
        self.register_buffer("lo", torch.tensor(np.stack([s["lo"] for s in specs]), dtype=torch.float32))
        self.register_buffer("hi", torch.tensor(np.stack([s["hi"] for s in specs]), dtype=torch.float32))
        self.register_buffer("dmin", torch.tensor(np.asarray(dataset_aabb[0]), dtype=torch.float32))
        self.register_buffer("dmax", torch.tensor(np.asarray(dataset_aabb[1]), dtype=torch.float32))
        self.iteration_idx = 0

    def set_iteration_index(self, idx):
        self.iteration_idx = idx
        for u in self.units:
            u.set_iteration_index(idx)

    def forward(self, position, camera_dir, light_dir, normal):
        """position: (B, N, 4) = dataset-normalised xyz + owner id; returns (B, N, 6)."""
        pos, owner = position[..., :3], position[..., 3].round().long()
        world = pos * (self.dmax - self.dmin) + self.dmin
        out = None
        for k, unit in enumerate(self.units):
            sel = owner == k
            if not bool(sel.any()):
                continue
            local = ((world[sel] - self.lo[k]) / (self.hi[k] - self.lo[k])).clamp(0.0, 1.0)
            o = unit(local.unsqueeze(0), camera_dir[sel].unsqueeze(0), light_dir[sel].unsqueeze(0), normal[sel].unsqueeze(0))
            if out is None:
                out = torch.zeros(*pos.shape[:-1], o.shape[-1], dtype=o.dtype, device=o.device)
            out[sel] = o.squeeze(0)
        return out

    def export(self):
        return {f"unit{k}.{n}": v for k, u in enumerate(self.units) for n, v in u.export().items()}

    def unit_parameters(self, k):
        return list(self.units[k].parameters())


class OwnedSurfaceDataset(rdatasets.NeuralSurfaceDataset):
    """Upstream dataset + owner id appended to every (normalised) position."""

    def __init__(self, *args, owner=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.owner_counts = np.zeros(owner.units, np.int64)
        self.owner_max_dist = 0.0
        for i in range(self.total_slices):
            p = self.positions[i]
            world = (p.to(self.ops_device) * (self.aabb_max - self.aabb_min) + self.aabb_min).cpu().numpy()
            lab, d = owner(world)
            self.owner_counts += np.bincount(lab, minlength=owner.units)
            self.owner_max_dist = max(self.owner_max_dist, float(d.max()) if len(d) else 0.0)
            self.positions[i] = torch.cat([p, torch.tensor(lab, dtype=p.dtype, device=p.device)[:, None]], -1)


class OwnedSurfaceDataModule(rinterfaces.NeuralSurfaceDataModule):
    def __init__(self, conf, owner=None, **kw):
        super().__init__(conf, **kw)
        self.owner = owner

    def setup(self, stage=None):
        mk = lambda f: OwnedSurfaceDataset(datafile=f, use_for_visibility=self.use_for_visibility, use_positions=True,
                                           device=self.device, ops_device=self.ops_device, owner=self.owner)
        self.trainset = mk(self.conf.training_dataset)
        self.valset = mk(self.conf.validation_dataset) if self.conf.validation_dataset else None


class OwnedSurfaceTriplaneModule(rinterfaces.NeuralSurfaceTriplaneModule):
    """Upstream Lightning module (training/validation steps, loss, optimiser, schedule unchanged)
    whose model is an OwnedTriplaneSet."""

    def __init__(self, conf, specs=None, dataset_aabb=None, **interface_params):
        super().__init__(conf, **interface_params)
        self.model = OwnedTriplaneSet(specs, dataset_aabb)
        self.model.units.apply(rmodels.init_gaussian)
