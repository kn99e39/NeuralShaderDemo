from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np


EXPERIMENT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from shared.correspondence import (  # noqa: E402
    interpolate,
    map_deformed_to_canonical,
    transform_points,
    triangle_normals,
    validate_barycentric_mapping,
)


class CorrespondenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.canonical = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            ]
        )
        self.weights = np.array([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]])

    def test_identity_deformation_matches_ordinary_lookup(self) -> None:
        points = interpolate(self.canonical, self.weights)
        mapped, measured = map_deformed_to_canonical(
            points, self.canonical, self.canonical
        )
        np.testing.assert_allclose(mapped, points, atol=1e-12)
        np.testing.assert_allclose(measured, self.weights, atol=1e-12)

    def test_rigid_transform_does_not_change_canonical_identity(self) -> None:
        angle = np.deg2rad(37.0)
        matrix = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0, 2.5],
                [np.sin(angle), np.cos(angle), 0.0, -1.25],
                [0.0, 0.0, 1.0, 0.75],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        deformed = transform_points(self.canonical, matrix)
        current_points = interpolate(deformed, self.weights)
        mapped, measured = map_deformed_to_canonical(
            current_points, deformed, self.canonical
        )
        np.testing.assert_allclose(
            mapped, interpolate(self.canonical, self.weights), atol=1e-12
        )
        np.testing.assert_allclose(measured, self.weights, atol=1e-12)

    def test_barycentric_correspondence_under_intrinsic_deformation(self) -> None:
        deformed = self.canonical.copy()
        deformed[0, 2] = [-0.1, 0.7, 0.8]
        deformed[1, 1] = [1.1, 0.8, -0.4]
        summary = validate_barycentric_mapping(
            self.canonical, deformed, self.weights
        )
        self.assertLess(summary.max_position_error, 1e-12)
        self.assertLess(summary.max_barycentric_error, 1e-12)

    def test_normals_are_recomputed_from_current_geometry(self) -> None:
        canonical_normals = triangle_normals(self.canonical)
        deformed = self.canonical.copy()
        deformed[:, 2, 2] = 0.75
        current_normals = triangle_normals(deformed)
        self.assertGreater(
            float(np.max(np.linalg.norm(current_normals - canonical_normals, axis=-1))),
            0.25,
        )
        expected = np.cross(
            deformed[:, 1] - deformed[:, 0], deformed[:, 2] - deformed[:, 0]
        )
        expected /= np.linalg.norm(expected, axis=-1, keepdims=True)
        np.testing.assert_allclose(current_normals, expected, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
