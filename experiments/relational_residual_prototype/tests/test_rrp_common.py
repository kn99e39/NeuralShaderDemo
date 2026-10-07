"""Focused tests for rrp_common (numpy only).  Run with the 8DNA venv python:
    external/8dna26/.venv-cu128/Scripts/python.exe experiments/relational_residual_prototype/tests/test_rrp_common.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rrp_common as C  # noqa: E402


class TestProbeDirections(unittest.TestCase):
    def test_deterministic_hemisphere(self):
        k = C.protocol()["probes"]["K"]
        a, b = C.probe_dirs(k), C.probe_dirs(k)
        self.assertEqual(a.shape, (32, 3))
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-12)
        self.assertTrue(np.all(a[:, 1] > 0))
        # it is the existing project utility's upper half, not a new pattern
        sys.path.insert(0, str(C.EXP8))
        import teaset_parts as T

        np.testing.assert_array_equal(a, T._fibonacci_dirs(64)[:32])
        # roughly uniform over the hemisphere: mean direction ~ (0, 1/2, 0)
        m = a.mean(0)
        self.assertLess(abs(m[0]) + abs(m[2]), 0.1)
        self.assertAlmostEqual(m[1], 0.5, delta=0.02)


class TestFrames(unittest.TestCase):
    def test_onb(self):
        rng = np.random.default_rng(0)
        n = rng.normal(size=(1000, 3))
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        n[0] = [0, 0, -1]
        n[1] = [0, 0, 1]
        s, t = C.onb(n)
        for u, v in ((s, t), (s, n), (t, n)):
            np.testing.assert_allclose(np.sum(u * v, 1), 0.0, atol=1e-12)
        for u in (s, t):
            np.testing.assert_allclose(np.linalg.norm(u, axis=1), 1.0, atol=1e-12)
        s2, t2 = C.onb(n.copy())
        np.testing.assert_array_equal(s, s2)
        np.testing.assert_array_equal(t, t2)

    def test_local_world_roundtrip(self):
        rng = np.random.default_rng(1)
        n = rng.normal(size=(50, 3))
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        s, t = C.onb(n)
        d = C.probe_dirs(32)
        w = C.to_world(d, s, n, t)  # (50, 32, 3)
        back = np.stack([C.to_local(w[:, j], s, n, t) for j in range(32)], 1)
        np.testing.assert_allclose(back, np.broadcast_to(d, back.shape), atol=1e-12)
        # every probe leaves the surface on the normal side
        self.assertTrue(np.all(np.sum(w * n[:, None], -1) > 0))


class TestSplitAndStable(unittest.TestCase):
    def test_validation_pixels(self):
        pix = np.arange(512 * 512)
        v = C.validation_pixels(pix, 512, 8, 5)
        np.testing.assert_array_equal(v, C.validation_pixels(pix, 512, 8, 5))
        self.assertAlmostEqual(v.mean(), 0.2, delta=0.01)
        # whole 8x8 blocks
        y, x = np.divmod(pix, 512)
        blk = (y // 8) * 64 + (x // 8)
        for b in np.unique(blk[:5000]):
            self.assertEqual(len(np.unique(v[blk == b])), 1)

    def test_split_disjoint_in_protocol(self):
        sp = C.protocol()["split"]
        self.assertFalse(set(sp["train_states"]) & set(sp["holdout_states"]))
        self.assertIn("T3", sp["holdout_states"])
        self.assertIn("T1", sp["holdout_states"])

    def test_stable_pixels(self):
        k = 4
        n = 3 * k
        base = {"hit": np.ones(n, bool), "part": np.full(n, 3, np.int8), "canonical_position": np.zeros((n, 3), np.float32),
                "normal": np.tile([0, 1, 0], (n, 1)).astype(np.float32), "camera_dir": np.tile([0, 0, 1], (n, 1)).astype(np.float32)}
        st = {k_: v.copy() for k_, v in base.items()}
        st["part"][1 * k + 2] = 1           # pixel 1: one sample now sees another part
        st["canonical_position"][2 * k] += 1e-7  # pixel 2: one sample's canonical point moved
        pix = np.arange(3)
        ok = C.stable_pixels(base, st, pix, pix, k, 3, require_identity=True)
        np.testing.assert_array_equal(ok, [True, False, False])
        ok2 = C.stable_pixels(base, st, pix, pix, k, 3, require_identity=False)
        np.testing.assert_array_equal(ok2, [True, False, True])

    def test_hit_order_index(self):
        hit = np.array([0, 1, 1, 0, 1], bool)
        np.testing.assert_array_equal(C.hit_order_index(hit, np.array([1, 4])), [0, 2])
        with self.assertRaises(ValueError):
            C.hit_order_index(hit, np.array([3]))


class TestNoLeakage(unittest.TestCase):
    def test_feature_names(self):
        C.check_feature_names()
        self.assertEqual(len(C.PROBE_FEATURES), 20)
        self.assertEqual(len(C.LOCAL_FEATURES), 19)
        saved = C.PROBE_FEATURES
        try:
            C.PROBE_FEATURES = saved + ("mover_distance",)
            with self.assertRaises(AssertionError):
                C.check_feature_names()
            C.PROBE_FEATURES = saved + ("state_id",)
            with self.assertRaises(AssertionError):
                C.check_feature_names()
        finally:
            C.PROBE_FEATURES = saved


if __name__ == "__main__":
    unittest.main(verbosity=2)
