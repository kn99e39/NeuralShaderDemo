from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np


PACKAGE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

from correspondence import (  # noqa: E402
    canonical_lookup_positions,
    current_triangle_normals,
    interpolate,
)


class CorrespondenceContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.canonical = np.array(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
        )
        self.weights = np.array([[0.2, 0.3, 0.5]])

    def test_identity_lookup_is_the_current_surface_point(self) -> None:
        point = interpolate(self.canonical, self.weights)
        lookup, recovered = canonical_lookup_positions(point, self.canonical, self.canonical)
        np.testing.assert_allclose(lookup, point, atol=1e-12)
        np.testing.assert_allclose(recovered, self.weights, atol=1e-12)

    def test_rigid_translation_keeps_feature_identity(self) -> None:
        translated = self.canonical + np.array([4.5, -2.0, 0.75])
        current_point = interpolate(translated, self.weights)
        lookup, recovered = canonical_lookup_positions(current_point, translated, self.canonical)
        np.testing.assert_allclose(lookup, interpolate(self.canonical, self.weights), atol=1e-12)
        np.testing.assert_allclose(recovered, self.weights, atol=1e-12)

    def test_intrinsic_deformation_keeps_feature_identity_but_updates_normal(self) -> None:
        current = self.canonical.copy()
        current[0, 2] = [-0.25, 0.70, 0.80]
        current_point = interpolate(current, self.weights)
        lookup, _ = canonical_lookup_positions(current_point, current, self.canonical)
        np.testing.assert_allclose(lookup, interpolate(self.canonical, self.weights), atol=1e-12)
        self.assertGreater(
            float(np.linalg.norm(current_triangle_normals(current) - current_triangle_normals(self.canonical))),
            0.2,
        )


if __name__ == "__main__":
    unittest.main()
