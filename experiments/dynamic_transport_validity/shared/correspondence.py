"""Topology-preserving deformed-to-canonical surface correspondence.

The functions here deliberately have no Blender or RNA dependency.  They define
the numerical contract used by the renderer and by focused validation tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


Array = np.ndarray


def _as_points(value: Iterable[float] | Array, width: int = 3) -> Array:
    array = np.asarray(value, dtype=np.float64)
    if array.shape[-1] != width:
        raise ValueError(f"expected final dimension {width}, got {array.shape}")
    return array


def barycentric_coordinates(points: Array, triangles: Array, eps: float = 1e-12) -> Array:
    """Return barycentric weights for points on their paired 3D triangles.

    `points` is `(..., 3)` and `triangles` is `(..., 3, 3)`.  Leading dimensions
    broadcast.  Degenerate triangles are rejected rather than silently producing
    an unrelated lookup.
    """

    p = _as_points(points)
    tri = _as_points(triangles)
    if tri.shape[-2] != 3:
        raise ValueError(f"expected three vertices per triangle, got {tri.shape}")

    a, b, c = tri[..., 0, :], tri[..., 1, :], tri[..., 2, :]
    v0, v1, v2 = b - a, c - a, p - a
    d00 = np.sum(v0 * v0, axis=-1)
    d01 = np.sum(v0 * v1, axis=-1)
    d11 = np.sum(v1 * v1, axis=-1)
    d20 = np.sum(v2 * v0, axis=-1)
    d21 = np.sum(v2 * v1, axis=-1)
    denominator = d00 * d11 - d01 * d01
    if np.any(np.abs(denominator) <= eps):
        raise ValueError("degenerate triangle in barycentric query")

    weight_b = (d11 * d20 - d01 * d21) / denominator
    weight_c = (d00 * d21 - d01 * d20) / denominator
    weight_a = 1.0 - weight_b - weight_c
    return np.stack((weight_a, weight_b, weight_c), axis=-1)


def interpolate(vertices: Array, barycentrics: Array) -> Array:
    """Barycentrically interpolate paired triangle vertex values."""

    values = np.asarray(vertices, dtype=np.float64)
    weights = np.asarray(barycentrics, dtype=np.float64)
    if values.shape[-2] != 3 or weights.shape[-1] != 3:
        raise ValueError("expected triangle values (...,3,C) and weights (...,3)")
    return np.sum(values * weights[..., :, None], axis=-2)


def map_deformed_to_canonical(
    deformed_points: Array,
    deformed_triangles: Array,
    canonical_triangles: Array,
) -> tuple[Array, Array]:
    """Map deformed surface samples to canonical positions by triangle identity."""

    weights = barycentric_coordinates(deformed_points, deformed_triangles)
    return interpolate(canonical_triangles, weights), weights


def transform_points(points: Array, matrix: Array) -> Array:
    """Apply a homogeneous 4x4 transform to 3D points."""

    pts = _as_points(points)
    mtx = np.asarray(matrix, dtype=np.float64)
    if mtx.shape != (4, 4):
        raise ValueError(f"expected a 4x4 matrix, got {mtx.shape}")
    homogeneous = np.concatenate((pts, np.ones((*pts.shape[:-1], 1))), axis=-1)
    transformed = homogeneous @ mtx.T
    return transformed[..., :3] / transformed[..., 3:4]


def triangle_normals(triangles: Array, eps: float = 1e-12) -> Array:
    """Compute unit geometric normals from the current triangle positions."""

    tri = _as_points(triangles)
    normals = np.cross(tri[..., 1, :] - tri[..., 0, :], tri[..., 2, :] - tri[..., 0, :])
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    if np.any(lengths <= eps):
        raise ValueError("degenerate triangle in normal computation")
    return normals / lengths


@dataclass(frozen=True)
class ValidationSummary:
    samples: int
    max_position_error: float
    max_barycentric_error: float
    max_normal_error: float = 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "samples": self.samples,
            "max_position_error": self.max_position_error,
            "max_barycentric_error": self.max_barycentric_error,
            "max_normal_error": self.max_normal_error,
        }


def validate_barycentric_mapping(
    canonical_triangles: Array,
    deformed_triangles: Array,
    barycentrics: Array,
) -> ValidationSummary:
    """Round-trip known surface samples and summarize numerical error."""

    canonical = _as_points(canonical_triangles)
    deformed = _as_points(deformed_triangles)
    expected_weights = np.asarray(barycentrics, dtype=np.float64)
    current_points = interpolate(deformed, expected_weights)
    mapped_points, measured_weights = map_deformed_to_canonical(
        current_points, deformed, canonical
    )
    expected_points = interpolate(canonical, expected_weights)
    return ValidationSummary(
        samples=int(np.prod(expected_weights.shape[:-1])),
        max_position_error=float(np.max(np.abs(mapped_points - expected_points))),
        max_barycentric_error=float(np.max(np.abs(measured_weights - expected_weights))),
    )

