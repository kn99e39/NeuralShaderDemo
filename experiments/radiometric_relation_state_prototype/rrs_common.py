"""Shared definitions for the radiometric relation-state prototype (worklog 29), numpy only.

The worklog-28 probe structure is reused unchanged (rrp_common); this module only
adds the three proxy channels, their encoding and the three input semantics:

    real      aligned runtime radiometric proxy (frozen RNA rendered at the hit toward the query)
    zero      proxy channels set to 0 everywhere          (matched control)
    shuffled  hit-probe proxy values permuted within the state (matched control)
    oracle    path-traced incident radiance along the same directions (diagnostic only)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "relational_residual_prototype"))

import rrp_common as C28  # noqa: E402

PROTOCOL = HERE / "protocol" / "rrs_v1.json"
PROXY_FEATURES = ("proxy_log_r", "proxy_log_g", "proxy_log_b")
PROBE_FEATURES = C28.PROBE_FEATURES + PROXY_FEATURES
VARIANTS = ("zero", "shuffled", "real")
SHUFFLE_SEED = 29029


def protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def encode(rgb: np.ndarray) -> np.ndarray:
    """Fixed encoding of a radiometric value: log1p(max(rgb, 0))."""
    return np.log1p(np.maximum(rgb, 0.0)).astype(np.float32)


def shuffle_within_state(values: np.ndarray, hit: np.ndarray, seed: int = SHUFFLE_SEED) -> np.ndarray:
    """Permute the hit probes' values among themselves (one state only); misses stay 0.

    values: (Q, K, 3) encoded proxy (0 on misses); hit: (Q, K) bool.
    Keeps the marginal distribution of hit values, destroys their query/probe alignment.
    """
    out = np.zeros_like(values)
    v = values[hit]
    out[hit] = v[np.random.default_rng(seed).permutation(len(v))]
    return out


def assemble(probes20: np.ndarray, proxy_enc: np.ndarray, hit: np.ndarray, variant: str) -> np.ndarray:
    """Worklog-28 20-channel descriptor + 3 proxy channels under one input semantics."""
    if probes20.shape[-1] != len(C28.PROBE_FEATURES):
        raise ValueError("expected the worklog-28 20-channel descriptor")
    if variant in ("real", "oracle"):
        extra = proxy_enc
    elif variant == "zero":
        extra = np.zeros_like(proxy_enc)
    elif variant == "shuffled":
        extra = shuffle_within_state(proxy_enc, hit)
    else:
        raise ValueError(variant)
    return np.concatenate([probes20, extra.astype(np.float32)], -1)
