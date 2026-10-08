"""Shared definitions for the oracle selective-refit feasibility batch (worklog 30).

Substrate: released 8DNA teaset asset (external/8dna26 @ 4a2157c, unmodified).
The only position-indexed learned state is the triplane `model.encode_xi.feature`
of shape (3, C, N, N); one *state unit* is one (plane, row, col) cell with its C
channels.  Everything else in the model is shared (protocol `state_units`).

Cell support follows upstream `models.mlps.fetch_2d` exactly:

    plane 0 (xy): col <- x, row <- y      feature[0][:, row, col]
    plane 1 (yz): col <- y, row <- z
    plane 2 (zx): col <- z, row <- x
    u = clamp((c * 0.5 + 0.5) * (N - 1), 0, N - 1); stencil floor/ceil with
    weights (1-fx)(1-fy), fx(1-fy), (1-fx)fy, fx fy

Import ednalib (via `init()`) before anything that touches Mitsuba.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPL = ROOT / "experiments" / "8dna_deformation_replication"
NRC = ROOT / "experiments" / "neural_recompute_cost"
for p in (HERE, REPL, NRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

RESULTS = ROOT / "results" / "selective_refit_oracle"
PROTOCOL_PATH = HERE / "protocol" / "sro_v1.json"
STATES_PROTOCOL = REPL / "protocol" / "teaset_frozen_locked.json"

TRIPLANE_KEY = "model.encode_xi.feature"  # state_dict key (released-layout prefix 'model.')
TRIPLANE_ATTR = "encode_xi.feature"       # attribute path on EightDNA
PLANE_AXES = ((0, 1), (1, 2), (2, 0))      # (col axis, row axis) per plane, as fetch_2d


def protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def states() -> dict:
    return json.loads(STATES_PROTOCOL.read_text(encoding="utf-8"))["states"]


def init():
    """Upstream runtime (ednalib.init_upstream): cuda_ad_rgb, LoopRecord off, VCallRecord on."""
    import ednalib as L

    return L.init_upstream()


# ----------------------------------------------------------------------------- state units

def stencil(xi, N: int):
    """Bilinear stencil of upstream fetch_2d for every plane.

    xi: (B, 3) torch tensor in the asset frame.  Returns (idx, w) with
    idx: (B, 3 planes, 4) long flat cell index  plane * N * N + row * N + col
    w:   (B, 3, 4) float weights (exactly upstream's).
    """
    import torch

    idxs, ws = [], []
    for p, (a, b) in enumerate(PLANE_AXES):
        xy = xi[:, [a, b]]
        xy = (xy * 0.5 + 0.5).mul(N - 1).clamp(0, N - 1)
        xy0 = xy.floor().long().clamp(0, N - 1)
        x1, y1 = xy.ceil().long().clamp(0, N - 1).unbind(-1)
        fx, fy = (xy - xy0).unbind(-1)
        x0, y0 = xy0.unbind(-1)
        base = p * N * N
        idxs.append(torch.stack([base + y0 * N + x0, base + y0 * N + x1, base + y1 * N + x0, base + y1 * N + x1], -1))
        ws.append(torch.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], -1))
    return torch.stack(idxs, 1), torch.stack(ws, 1)


def touches(xi, cell_mask_flat, N: int):
    """True for samples whose stencil has a selected cell with nonzero weight."""
    idx, w = stencil(xi, N)
    return ((cell_mask_flat[idx]) & (w > 0)).flatten(1).any(1)


def cell_mask_to_param_mask(cell_mask_flat, C: int, N: int):
    """(3*N*N,) bool -> (3, C, N, N) bool, the layout of encode_xi.feature."""
    return cell_mask_flat.view(3, 1, N, N).expand(3, C, N, N).contiguous()


def cell_centres(N: int):
    """(3*N*N, 2) plane coordinates in [-1, 1] of every cell centre (col coord, row coord)."""
    import numpy as np

    g = np.linspace(-1.0, 1.0, N)
    row, col = np.meshgrid(g, g, indexing="ij")
    one = np.stack([col.ravel(), row.ravel()], -1)
    return np.concatenate([one, one, one], 0)


# ----------------------------------------------------------------------------- model / checkpoints

def released_checkpoint() -> Path:
    return ROOT / "external" / "8dna26" / "checkpoints" / "teaset.ckpt"


def load_model(ckpt_path, device="cuda"):
    """EightDNA in training form (pure-PyTorch flows, as upstream train.py), weights from a
    released-layout checkpoint ({'state_dict': {'model.<k>': ...}, 'hyper_parameters': {'model': ...}})."""
    import torch
    from models.eight_dna import EightDNA

    from omegaconf import OmegaConf

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = EightDNA(**OmegaConf.create(ck["hyper_parameters"]).model)  # as upstream load_asset
    sd = {k[len("model."):]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(sd)
    return model.to(device), ck["hyper_parameters"]


def scratch_model(hparams: dict, seed: int, device="cuda"):
    """Upstream initialisation (torch default inits + randn grids) under a fixed seed."""
    import torch
    from models.eight_dna import EightDNA

    from omegaconf import OmegaConf

    torch.manual_seed(seed)
    return EightDNA(**OmegaConf.create(hparams).model).to(device)


def released_layout(model, hparams: dict) -> dict:
    sd = {f"model.{k}": v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    return {"state_dict": sd, "hyper_parameters": {"model": hparams["model"]}}


def param_digest(state_dict: dict, keys=None) -> str:
    """SHA-256 over the raw bytes of the given tensors (sorted keys)."""
    h = hashlib.sha256()
    for k in sorted(keys if keys is not None else state_dict):
        t = state_dict[k].detach().to("cpu").contiguous()
        h.update(k.encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def register_state_scene(state: str, translations: dict) -> str:
    """Register scenes.teaset_<state> exactly as train_8dna_state.py does; return the name."""
    import teaset_parts as T

    name = f"teaset_{state}"
    module = types.ModuleType(f"scenes.{name}")
    module.get_scene = lambda res, *_, **__: T.scene_dict(res, translations)
    sys.modules[f"scenes.{name}"] = module
    import scenes

    setattr(scenes, name, module)
    return name


def write_json(path, payload) -> None:
    import nrc_records as R

    R.write_json(path, payload)
