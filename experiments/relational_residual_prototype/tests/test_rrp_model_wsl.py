"""Model / training-contract tests (WSL, RNA venv):
    cd external/relightable-neural-assets
    .venv/bin/python ../../experiments/relational_residual_prototype/tests/test_rrp_model_wsl.py
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "wsl"))

import rrp_common as C  # noqa: E402
import rrp_train  # noqa: E402
from rrp_model import LocalResidual, RelationalResidual, n_params  # noqa: E402

MC = C.protocol()["model"]
P, LD = len(C.PROBE_FEATURES), len(C.LOCAL_FEATURES)


class TestBranches(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.rel = RelationalResidual(P, LD, MC)
        self.loc = LocalResidual(LD, MC)
        self.local = torch.randn(64, LD)
        self.probes = torch.randn(64, 32, P)

    def test_local_only_has_no_probe_path(self):
        with self.assertRaises(TypeError):
            self.loc(self.local, self.probes)
        out = self.loc(self.local)
        self.assertEqual(tuple(out.shape), (64, 3))
        # no parameter of the control is shaped for probe descriptors
        self.assertFalse(any(p.shape[-1] == P for p in self.loc.parameters()))

    def test_capacity_matched(self):
        a, b = n_params(self.rel), n_params(self.loc)
        self.assertLess(abs(a - b) / a, 0.05, (a, b))

    def test_relational_permutation_stable(self):
        perm = torch.randperm(32)
        a = self.rel(self.local, self.probes)
        b = self.rel(self.local, self.probes[:, perm])
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)

    def test_relational_uses_probes(self):
        a = self.rel(self.local, self.probes)
        b = self.rel(self.local, self.probes + 1.0)
        self.assertGreater(float((a - b).abs().max()), 0.0)

    def test_state_serialisation(self):
        g = self.rel.state(self.probes).detach().numpy().astype(np.float32)
        self.assertEqual(g.shape, (64, MC["state_dim"]))
        with tempfile.TemporaryDirectory() as d:
            np.save(Path(d) / "g.npy", g)
            np.testing.assert_array_equal(np.load(Path(d) / "g.npy"), g)
        sd = {k: v.clone() for k, v in self.rel.state_dict().items()}
        r2 = RelationalResidual(P, LD, MC)
        r2.load_state_dict(sd)
        torch.testing.assert_close(r2(self.local, self.probes), self.rel(self.local, self.probes))


class TestSplit(unittest.TestCase):
    def test_holdouts_refused_for_training(self):
        proto = C.protocol()
        for s in ("T1", "T3", "T0_B"):
            with self.assertRaises(rrp_train.SplitError):
                rrp_train.load_training_state(Path("/nonexistent"), s, proto)
        bad = dict(proto, split=dict(proto["split"], train_states=["T0", "T3"]))
        with self.assertRaises(rrp_train.SplitError):
            rrp_train.load_training_state(Path("/nonexistent"), "T3", bad)


class TestFrozenRNA(unittest.TestCase):
    def test_checkpoint_hash_and_params_frozen(self):
        proto = C.protocol()
        ck = C.ROOT / proto["frozen_rna"]["checkpoint"]
        self.assertEqual(hashlib.sha256(ck.read_bytes()).hexdigest(), proto["frozen_rna"]["sha256"])
        sys.path.insert(0, ".")
        from rna import interfaces

        m = interfaces.NeuralSurfaceTriplaneModule.load_from_checkpoint(str(ck), strict=False, map_location="cpu")
        m.eval().freeze()
        self.assertFalse(any(p.requires_grad for p in m.parameters()))
        captured = {}
        m.model.triplane_grid.register_forward_hook(lambda mod, i, o: captured.__setitem__("f", o))
        m.model.initial_sigma = 1
        m.model.iterations_to_sigma_1 = 1
        pos = torch.rand(1, 10, 3)
        d = torch.tensor([0.0, 1.0, 0.0]).expand(1, 10, 3)
        with torch.no_grad():
            m(pos, d, d, d)
            f1 = captured["f"].clone()
            m(pos, -d, d, -d)  # directions do not enter the triplane feature
            f2 = captured["f"].clone()
        self.assertEqual(tuple(f1.shape), (1, 8, 10))
        torch.testing.assert_close(f1, f2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
