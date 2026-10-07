"""Shared definitions for the relational-residual prototype (worklog 28), numpy only.

Imported by the Windows/Mitsuba probe stage and by the WSL/RNA training stage,
so nothing here touches Mitsuba, DrJit or torch.

Contract (protocol/rrp_v1.json):

    L_candidate(i, t) = L_base(i, t) + dL_dynamic(i, t)

L_base is the frozen historical RNA prediction; dL_dynamic comes from a small
shared operator reading (a) persistent/local query inputs and (b) for the
relational branch only, a sparse current-geometry state of K fixed probes per
query.  Queries are the sub-pixel samples of the established interaction ROI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP8 = ROOT / "experiments" / "8dna_deformation_replication"
RES8 = ROOT / "results" / "8dna_replication"
PROTOCOL = HERE / "protocol" / "rrp_v1.json"
QUERY_PART = "teapot4"  # the stationary tea pot carrying the interaction ROI

# Per-probe descriptor channels (relational branch).  Everything is either a
# current-geometry relation expressed in the query's local frame or a
# persistent surface-attached feature of the remote hit; no state label, no
# mover transform, no world-space position.
PROBE_FEATURES = (
    "hit",                      # 1 if the probe ray hits the asset
    "near_dist",                # exp(-t / 0.1) (0 on a miss); t = hit distance
    "far_dist",                 # exp(-t / 1.0) (0 on a miss)
    "remote_n_s", "remote_n_n", "remote_n_t",  # remote shading normal (flipped to face the query), query frame
    "remote_facing",            # -dot(probe dir, remote normal) (0 on a miss)
    *(f"remote_feat_{c}" for c in range(8)),   # frozen RNA triplane feature at the remote canonical point
    "remote_light_vis",         # remote direct visibility toward the light centre (rna_bridge.visibility)
    "remote_light_cos",         # max(0, remote normal . to_light) (0 on a miss)
    "dir_s", "dir_n", "dir_t",  # the fixed probe direction in the query frame
)
# Persistent/local query inputs (both branches).
LOCAL_FEATURES = (
    *(f"query_feat_{c}" for c in range(8)),    # frozen RNA triplane feature at the query's canonical point
    "view_s", "view_n", "view_t",              # camera direction, query frame
    "light_s", "light_n", "light_t",           # mean of the 16 area-light sample directions, query frame
    "direct_vis_fraction",                     # mean of the 16 current direct-visibility bits (already an RNA input)
    "direct_irradiance",                       # mean(vis * weight) / training light intensity
    "base_log_r", "base_log_g", "base_log_b",  # log1p of the frozen RNA prediction for this sample
)
FORBIDDEN_TOKENS = ("state", "T0", "T1", "T2", "T3", "mover", "translation", "transform", "world", "gt", "reference")


def protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def probe_dirs(k: int) -> np.ndarray:
    """K fixed hemisphere directions (local frame, +y = surface normal).

    The upper half of the project's existing Fibonacci sphere
    (teaset_parts._fibonacci_dirs(2K)): its first K points have y > 0.
    """
    sys.path.insert(0, str(EXP8))
    import teaset_parts as T

    d = T._fibonacci_dirs(2 * k)[:k]
    if not np.all(d[:, 1] > 0):
        raise AssertionError("upper Fibonacci half is not a hemisphere")
    return d.astype(np.float64)


def onb(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic orthonormal tangents (s, t) for unit normals n (Duff et al. 2017)."""
    n = np.asarray(n, np.float64)
    sign = np.where(n[:, 2] >= 0, 1.0, -1.0)
    a = -1.0 / (sign + n[:, 2])
    b = n[:, 0] * n[:, 1] * a
    s = np.stack([1.0 + sign * n[:, 0] ** 2 * a, sign * b, -sign * n[:, 0]], -1)
    t = np.stack([b, sign + n[:, 1] ** 2 * a, -n[:, 1]], -1)
    return s, t


def to_local(v: np.ndarray, s: np.ndarray, n: np.ndarray, t: np.ndarray) -> np.ndarray:
    """World vectors -> (s, n, t) components of the query frame (y = normal, as probe_dirs)."""
    return np.stack([np.sum(v * s, -1), np.sum(v * n, -1), np.sum(v * t, -1)], -1)


def to_world(d_local: np.ndarray, s: np.ndarray, n: np.ndarray, t: np.ndarray) -> np.ndarray:
    """(Q, 3) frames x (K, 3) local dirs -> (Q, K, 3) world dirs."""
    return d_local[None, :, 0:1] * s[:, None] + d_local[None, :, 1:2] * n[:, None] + d_local[None, :, 2:3] * t[:, None]


def roi_lanes(pix: np.ndarray, k: int) -> np.ndarray:
    """Feature-buffer lanes of ROI pixels (lane = pixel * K + sub-pixel sample)."""
    return (np.asarray(pix)[:, None] * k + np.arange(k)[None]).reshape(-1)


def stable_pixels(f_ref: dict, f_state: dict, pix_state: np.ndarray, pix_ref: np.ndarray, k: int, part_index: int,
                  require_identity: bool) -> np.ndarray:
    """Per ROI pixel: every sub-pixel sample hits the query part in both buffers,
    and (require_identity) its canonical position and normal equal the reference's.

    For states whose ROI pixels are the T0 pixels (the stationary tea pot, T1b/T2/T3)
    require_identity=True enforces the exact stationary-query condition.  T1 moves the
    whole asset, so its pixels map to other T0 pixels and only the part condition applies.
    """
    ls, lr = roi_lanes(pix_state, k), roi_lanes(pix_ref, k)
    ok = (f_state["hit"][ls].astype(bool) & f_ref["hit"][lr].astype(bool)
          & (f_state["part"][ls] == part_index) & (f_ref["part"][lr] == part_index))
    if require_identity:
        for key in ("canonical_position", "normal", "camera_dir"):
            ok &= np.all(f_state[key][ls] == f_ref[key][lr], axis=-1)
    return ok.reshape(-1, k).all(1)


def validation_pixels(pix: np.ndarray, res: int, block: int, modulus: int) -> np.ndarray:
    """Deterministic spatial hold-out inside the training states: block (bx, by) is
    validation iff (bx + 2*by) % modulus == 0.  Same pixels in every training state."""
    y, x = np.divmod(np.asarray(pix), res)
    return ((x // block) + 2 * (y // block)) % modulus == 0


def check_feature_names() -> None:
    for name in PROBE_FEATURES + LOCAL_FEATURES:
        low = name.lower()
        for tok in FORBIDDEN_TOKENS:
            if tok.lower() in low.split("_") or (len(tok) > 2 and tok.lower() in low):
                raise AssertionError(f"feature {name!r} carries forbidden token {tok!r}")


def hit_order_index(hit: np.ndarray, lanes: np.ndarray) -> np.ndarray:
    """Index of each lane in the hit-only light arrays (stored in hit order)."""
    order = np.flatnonzero(np.asarray(hit).astype(bool))
    pos = np.searchsorted(order, lanes)
    if np.any(order[np.minimum(pos, len(order) - 1)] != lanes):
        raise ValueError("lane without a hit has no light samples")
    return pos
