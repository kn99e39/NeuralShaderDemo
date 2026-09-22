"""Numerical surface-identity contract for the high-quality Batch 1 only."""

from __future__ import annotations

import numpy as np


def barycentric_coordinates(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Return weights for paired non-degenerate triangles."""

    points = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.float64)
    a, b, c = triangles[..., 0, :], triangles[..., 1, :], triangles[..., 2, :]
    ab, ac, ap = b - a, c - a, points - a
    d00 = np.sum(ab * ab, axis=-1)
    d01 = np.sum(ab * ac, axis=-1)
    d11 = np.sum(ac * ac, axis=-1)
    d20 = np.sum(ap * ab, axis=-1)
    d21 = np.sum(ap * ac, axis=-1)
    denominator = d00 * d11 - d01 * d01
    if np.any(np.isclose(denominator, 0.0)):
        raise ValueError("cannot map a degenerate triangle")
    wb = (d11 * d20 - d01 * d21) / denominator
    wc = (d00 * d21 - d01 * d20) / denominator
    return np.stack((1.0 - wb - wc, wb, wc), axis=-1)


def interpolate(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.shape[-2] != 3 or weights.shape[-1] != 3:
        raise ValueError("expected a three-vertex triangle and three barycentric weights")
    return np.sum(values * weights[..., :, None], axis=-2)


def canonical_lookup_positions(
    current_points: np.ndarray,
    current_triangles: np.ndarray,
    canonical_triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map current hit points to their same-triangle canonical identity."""

    weights = barycentric_coordinates(current_points, current_triangles)
    return interpolate(canonical_triangles, weights), weights


def current_triangle_normals(triangles: np.ndarray) -> np.ndarray:
    """Current normals, deliberately never canonical normals."""

    triangles = np.asarray(triangles, dtype=np.float64)
    normal = np.cross(triangles[..., 1, :] - triangles[..., 0, :], triangles[..., 2, :] - triangles[..., 0, :])
    length = np.linalg.norm(normal, axis=-1, keepdims=True)
    if np.any(np.isclose(length, 0.0)):
        raise ValueError("cannot compute a normal for a degenerate triangle")
    return normal / length
