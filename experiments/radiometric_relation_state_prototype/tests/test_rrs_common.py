"""Focused tests for the worklog-29 state semantics (numpy only).
    external/8dna26/.venv-cu128/Scripts/python.exe experiments/radiometric_relation_state_prototype/tests/test_rrs_common.py
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import rrs_common as C  # noqa: E402
import rrp_common as C28  # noqa: E402


def fixture(q=200, k=32, seed=0):
    rng = np.random.default_rng(seed)
    p20 = rng.normal(size=(q, k, 20)).astype(np.float32)
    hit = rng.random((q, k)) < 0.55
    p20[..., C28.PROBE_FEATURES.index("hit")] = hit
    prox = np.where(hit[..., None], rng.random((q, k, 3)) * 3, 0.0)
    return p20, C.encode(prox), hit


class TestChannels(unittest.TestCase):
    def test_layout(self):
        self.assertEqual(len(C.PROBE_FEATURES), 23)
        self.assertEqual(C.PROBE_FEATURES[:20], C28.PROBE_FEATURES)
        saved = C28.PROBE_FEATURES
        try:
            C28.PROBE_FEATURES = C.PROBE_FEATURES  # the leakage guard accepts the new names
            C28.check_feature_names()
        finally:
            C28.PROBE_FEATURES = saved

    def test_encoding(self):
        v = np.array([[0.0, 1.0, -2.0]])
        np.testing.assert_allclose(C.encode(v), [[0.0, np.log(2.0), 0.0]], rtol=1e-6)


class TestVariants(unittest.TestCase):
    def test_zero_only_zeros_radiometric(self):
        p20, e, hit = fixture()
        z = C.assemble(p20, e, hit, "zero")
        np.testing.assert_array_equal(z[..., :20], p20)
        self.assertTrue(np.all(z[..., 20:] == 0))

    def test_real_passes_aligned_values(self):
        p20, e, hit = fixture()
        r = C.assemble(p20, e, hit, "real")
        np.testing.assert_array_equal(r[..., :20], p20)
        np.testing.assert_array_equal(r[..., 20:], e)

    def test_shuffle_preserves_marginals_destroys_alignment(self):
        p20, e, hit = fixture()
        s = C.assemble(p20, e, hit, "shuffled")
        np.testing.assert_array_equal(s[..., :20], p20)
        self.assertTrue(np.all(s[~hit][:, 20:] == 0))                  # misses stay 0
        a, b = np.sort(e[hit].reshape(-1)), np.sort(s[..., 20:][hit].reshape(-1))
        np.testing.assert_array_equal(a, b)                            # same multiset of values
        same = np.all(s[..., 20:][hit] == e[hit], -1).mean()
        self.assertLess(same, 0.02)                                    # alignment destroyed
        np.testing.assert_array_equal(s, C.assemble(p20, e, hit, "shuffled"))  # deterministic
        self.assertIn(f"default_rng({C.SHUFFLE_SEED})", C.protocol()["branches"]["C_shuffled"])  # the predeclared seed

    def test_bad_variant(self):
        p20, e, hit = fixture()
        with self.assertRaises(ValueError):
            C.assemble(p20, e, hit, "t3")


class TestNoLeakageInStateBuilders(unittest.TestCase):
    def test_primary_builders_do_not_read_references_or_labels(self):
        forbidden = [r"gt_A", r"gt_B", r"gt_pixels", r"transport_decomposition", r"roi_\w+_[AB]\.npz", r"\boracle\b", r"frozen_output",
                     r"\"T3\"\s*:", r"teapot2"]
        for f in ("rrs_remote_light.py", "wsl/rrs_proxy.py"):
            src = (HERE / f).read_text(encoding="utf-8")
            code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
            code = re.sub(r'"""[\s\S]*?"""', "", code)
            for pat in forbidden:
                self.assertIsNone(re.search(pat, code), f"{f} references {pat}")


class TestProtocol(unittest.TestCase):
    def test_contract_equals_worklog28(self):
        p, p28 = C.protocol(), C28.protocol()
        self.assertEqual(p["split"]["train_states"], p28["split"]["train_states"])
        self.assertEqual(p["split"]["holdout_states"], p28["split"]["holdout_states"])
        for key in ("lr", "steps", "batch_pixels", "val_every", "seeds", "loss_scale_offset"):
            self.assertEqual(p["training"][key], p28["training"][key], key)
        for key in ("probe_encoder", "state_dim", "decoder"):
            self.assertEqual(p["model"][key], p28["model"][key], key)
        self.assertEqual(p28["probes"]["K"], 32)


if __name__ == "__main__":
    unittest.main(verbosity=2)
