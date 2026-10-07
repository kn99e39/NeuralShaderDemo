"""Proxy / training-contract tests (WSL, RNA venv):
    cd external/relightable-neural-assets
    .venv/bin/python ../../experiments/radiometric_relation_state_prototype/tests/test_rrs_wsl.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "wsl"))
sys.path.insert(0, str(HERE.parents[0] / "relational_residual_prototype"))
sys.path.insert(0, str(HERE.parents[0] / "relational_residual_prototype" / "wsl"))
sys.path.insert(0, ".")

import rrs_common as C  # noqa: E402
import rrp_common as C28  # noqa: E402
import rrs_proxy  # noqa: E402
import rrp_train  # noqa: E402
from rrp_model import RelationalResidual, n_params  # noqa: E402


class TestProxy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p28 = C28.protocol()
        cb = json.loads((C28.EXP8 / p28["base_protocol"]).read_text(encoding="utf-8"))
        cls.dev = torch.device("cuda")
        cls.module, lo, hi = rrs_proxy.load_frozen(p28, cb, cls.dev)
        cls.d0 = rrs_proxy.digest(cls.module)
        cls.render = staticmethod(rrs_proxy.make_renderer(cls.module, lo, hi, float(cb["rna_inference"]["training_light_intensity"]), cls.dev))
        z = np.load(C.ROOT / C.protocol()["worklog28_reference"]["run"] / "probes" / "T0.npz")
        sel = slice(0, 64)
        cls.inp = [z["query_canonical"][sel], z["query_camera_dir"][sel], z["query_normal"][sel], z["light_dir"][sel],
                   z["light_weight"][sel], z["light_vis"][sel]]
        cls.base = np.load(C.ROOT / C.protocol()["worklog28_reference"]["run"] / "dataset" / "T0.npz")["base_samples"][sel]

    def test_reproduces_wl28_base(self):
        # 64 samples here vs worklog 28's 65 536-sample batches: cuBLAS may pick another kernel,
        # so equality is to float tolerance; the stage itself (same batching) checks <= 1e-6 abs
        np.testing.assert_allclose(self.render(*self.inp), self.base, rtol=1e-4, atol=1e-7)

    def test_uses_current_view_normal_visibility(self):
        ref = self.render(*self.inp)
        pos, cam, nrm, ld, lw, lv = self.inp
        cam2 = cam.copy()
        cam2[:, [0, 2]] = cam2[:, [2, 0]]
        self.assertGreater(np.abs(self.render(pos, cam2, nrm, ld, lw, lv) - ref).max(), 1e-4, "view direction ignored")
        nrm2 = -nrm
        self.assertGreater(np.abs(self.render(pos, cam, nrm2, ld, lw, lv) - ref).max(), 1e-4, "normal ignored")
        self.assertGreater(np.abs(self.render(pos, cam, nrm, ld, lw, ~lv) - ref).max(), 1e-4, "visibility ignored")
        np.testing.assert_array_equal(self.render(*self.inp), ref)  # deterministic

    def test_frozen_unchanged_and_no_grad(self):
        self.render(*self.inp)
        self.assertEqual(rrs_proxy.digest(self.module), self.d0)
        self.assertFalse(any(p.requires_grad for p in self.module.parameters()))


class TestContract(unittest.TestCase):
    def test_holdouts_refused(self):
        for s in ("T1", "T3", "T0_B"):
            with self.assertRaises(rrp_train.SplitError):
                rrp_train.load_training_state(Path("/nonexistent"), s, C28.protocol())

    def test_matched_shapes(self):
        mc = C28.protocol()["model"]
        m23 = RelationalResidual(len(C.PROBE_FEATURES), len(C28.LOCAL_FEATURES), mc)
        m20 = RelationalResidual(len(C28.PROBE_FEATURES), len(C28.LOCAL_FEATURES), mc)
        self.assertEqual(n_params(m23) - n_params(m20), 3 * mc["probe_encoder"][0])  # only the first layer's inputs differ
        widths = lambda m: [p.shape for n, p in m.named_parameters() if "phi.0" not in n]
        self.assertEqual(widths(m23), widths(m20))

    def test_state_serialisation(self):
        mc = C28.protocol()["model"]
        m = RelationalResidual(len(C.PROBE_FEATURES), len(C28.LOCAL_FEATURES), mc)
        g = m.state(torch.randn(10, 32, len(C.PROBE_FEATURES))).detach().numpy()
        with tempfile.TemporaryDirectory() as d:
            np.save(Path(d) / "g.npy", g)
            np.testing.assert_array_equal(np.load(Path(d) / "g.npy"), g)


if __name__ == "__main__":
    unittest.main(verbosity=2)
