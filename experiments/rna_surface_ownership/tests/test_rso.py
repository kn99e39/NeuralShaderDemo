"""Synthetic contract tests for surface-owned RNA (runs in the RNA venv, GPU).

    python tests/test_rso.py --partitions <dir> --out <dir>

Contracts only; never architecture evidence.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import rso_model as M  # noqa: E402

BASE = {"resolution": [64, 64], "triplane_features_num_channels": 8, "triplane_features_min_sigma": 1.0,
        "initial_sigma": 8.0, "iterations_to_sigma_1": 20000, "mlp_num_hidden_layers": 2, "mlp_hidden_layer_neurons": 32}
AABB = ((-0.72, 0.0, -0.73), (0.72, 0.70, 0.73))


def _inputs(n, seed, owner_fn=None):
    import numpy as np
    import torch

    rng = np.random.default_rng(seed)
    world = rng.uniform(AABB[0], AABB[1], (n, 3))
    pos = (world - np.array(AABB[0])) / (np.array(AABB[1]) - np.array(AABB[0]))
    owner = owner_fn(world) if owner_fn else np.zeros(n, np.int64)
    d = rng.normal(size=(3, n, 3)); d /= np.linalg.norm(d, axis=-1, keepdims=True)
    T = lambda a: torch.tensor(a, dtype=torch.float32, device="cuda").unsqueeze(0)
    return T(np.concatenate([pos, owner[:, None]], 1)), T(d[0]), T(d[1]), T(d[2]), world, owner


def t_partition(pdir):
    import numpy as np
    from rso_partition import Owner, surface_samples

    res = {}
    ok = True
    x, part, _ = surface_samples(200_000, seed=123)  # fresh points, not the stored samples
    for name, K in (("P1", 1), ("P4", 4), ("P16", 16)):
        o = Owner(Path(pdir) / f"partition_{name}.npz")
        lab, d = o(x)
        z = np.load(Path(pdir) / f"partition_{name}.npz")
        counts = np.bincount(lab, minlength=K)
        part_consistent = True if name == "P1" else bool((z["unit_part"][lab] == part).all())
        res[name] = {"units": o.units, "all_units_nonempty": bool((counts > 0).all()), "max_nn_distance": float(d.max()),
                     "unit_part_matches_point_part": part_consistent, "single_owner": True}
        ok &= o.units == K and (counts > 0).all() and d.max() < 0.01 and part_consistent  # d: nearest-sample distance outside contact zones
    return {"pass": bool(ok), **res}


def t_routing():
    """Each pixel's output equals its owner model's output alone; other units cannot affect it."""
    import numpy as np
    import torch

    K = 4
    specs = []
    for k in range(K):
        lo = np.array(AABB[0]) + np.array([k * 0.36, 0, 0]); hi = lo + np.array([0.36, 0.70, 1.46])
        specs.append({"params": dict(BASE), "lo": lo, "hi": hi})
    torch.manual_seed(0)
    m = M.OwnedTriplaneSet(specs, AABB).cuda().eval()
    owner_fn = lambda w: np.clip(((w[:, 0] - AABB[0][0]) / 0.36).astype(int), 0, K - 1)
    pos, cam, light, nrm, world, owner = _inputs(5000, 1, owner_fn)
    with torch.no_grad():
        out = m(pos, cam, light, nrm).squeeze(0)
        ok_alone = True
        for k in range(K):
            sel = torch.tensor(owner == k, device="cuda")
            local = ((torch.tensor(world[owner == k], dtype=torch.float32, device="cuda") - m.lo[k]) / (m.hi[k] - m.lo[k])).clamp(0, 1)
            alone = m.units[k](local.unsqueeze(0), cam[0][sel].unsqueeze(0), light[0][sel].unsqueeze(0), nrm[0][sel].unsqueeze(0)).squeeze(0)
            ok_alone &= bool(torch.allclose(alone, out[sel], rtol=0, atol=1e-6))  # batch-size-dependent float order only
        m2 = copy.deepcopy(m)
        with torch.no_grad():
            for p in m2.units[2].parameters():
                p.add_(1.0)
        out2 = m2(pos, cam, light, nrm).squeeze(0)
        sel2 = torch.tensor(owner == 2, device="cuda")
        others_same = bool(torch.equal(out2[~sel2], out[~sel2]))
        own_changed = bool((out2[sel2] != out[sel2]).any())
    return {"pass": ok_alone and others_same and own_changed, "owner_output_equals_unit_alone": ok_alone,
            "perturbing_unit2_leaves_other_pixels_identical": others_same, "perturbing_unit2_changes_its_pixels": own_changed}


def t_single_unit_equivalence():
    """One unit framed on the dataset AABB with the upstream model's weights == upstream model."""
    import numpy as np
    import torch

    torch.manual_seed(1)
    up = M.rmodels.NeuralSurfaceTriPlane(**BASE).cuda().eval()
    m = M.OwnedTriplaneSet([{"params": dict(BASE), "lo": np.array(AABB[0]), "hi": np.array(AABB[1])}], AABB).cuda().eval()
    m.units[0].load_state_dict(up.state_dict())
    pos, cam, light, nrm, _, _ = _inputs(4000, 2)
    with torch.no_grad():
        a = up(pos[..., :3], cam, light, nrm)
        b = m(pos, cam, light, nrm)
    err = float((a - b).abs().max())
    return {"pass": err < 1e-5, "max_abs_diff": err}


def t_isolation_and_checkpoint(tmp):
    """Train only unit 1: every other unit bit-identical; save/load round trip identical outputs."""
    import numpy as np
    import torch

    K = 3
    specs = [{"params": dict(BASE), "lo": np.array(AABB[0]), "hi": np.array(AABB[1])} for _ in range(K)]
    torch.manual_seed(2)
    m = M.OwnedTriplaneSet(specs, AABB).cuda()
    before = copy.deepcopy(m.state_dict())
    for k, u in enumerate(m.units):
        u.requires_grad_(k == 1)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    owner_fn = lambda w: np.random.default_rng(0).integers(0, K, len(w))
    for it in range(20):
        pos, cam, light, nrm, _, _ = _inputs(3000, 10 + it, owner_fn)
        m.set_iteration_index(it)
        loss = m(pos, cam, light, nrm).square().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    after = m.state_dict()
    frozen_ok = all(torch.equal(after[n], before[n]) for n in after if not n.startswith("units.1."))
    trained = any(not torch.equal(after[n], before[n]) for n in after if n.startswith("units.1."))
    p = Path(tmp) / "state.pt"
    torch.save(m.state_dict(), p)
    m2 = M.OwnedTriplaneSet(specs, AABB).cuda(); m2.load_state_dict(torch.load(p))
    # last_sigma (the eval-time blur) is not part of an upstream checkpoint: a reloaded plane uses
    # min_sigma, which equals the trained value only after the full sigma schedule; align it here
    for u in m.units:
        for g in (u.triplane_grid.xy_feature_grid, u.triplane_grid.yz_feature_grid, u.triplane_grid.xz_feature_grid):
            g.last_sigma = g.min_sigma
    m.eval(); m2.eval()
    pos, cam, light, nrm, _, _ = _inputs(2000, 99, owner_fn)
    with torch.no_grad():
        same = bool(torch.equal(m(pos, cam, light, nrm), m2(pos, cam, light, nrm)))
    return {"pass": frozen_ok and trained and same, "frozen_units_bit_identical": frozen_ok, "trained_unit_changed": trained,
            "checkpoint_roundtrip_identical": same}


def t_composition_reproducible(tmp):
    """The composed output is a pure function of (state, inputs): repeated evaluation, a
    shuffled pixel order and a save/load round trip give bit-identical per-pixel predictions."""
    import numpy as np
    import torch

    K = 4
    specs = []
    for k in range(K):
        lo = np.array(AABB[0]) + np.array([k * 0.36, 0, 0]); hi = lo + np.array([0.36, 0.70, 1.46])
        specs.append({"params": dict(BASE), "lo": lo, "hi": hi})
    torch.manual_seed(3)
    m = M.OwnedTriplaneSet(specs, AABB).cuda().eval()
    owner_fn = lambda w: np.clip(((w[:, 0] - AABB[0][0]) / 0.36).astype(int), 0, K - 1)
    pos, cam, light, nrm, _, _ = _inputs(4000, 5, owner_fn)
    p = Path(tmp) / "compose.pt"
    torch.save(m.state_dict(), p)
    m2 = M.OwnedTriplaneSet(specs, AABB).cuda().eval(); m2.load_state_dict(torch.load(p))
    perm = torch.randperm(pos.shape[1], device="cuda"); inv = torch.argsort(perm)
    with torch.no_grad():
        a, b = m(pos, cam, light, nrm), m(pos, cam, light, nrm)
        c = m2(pos, cam, light, nrm)
        d = m(pos[:, perm], cam[:, perm], light[:, perm], nrm[:, perm])[:, inv]
    repeat = bool(torch.equal(a, b)); reload = bool(torch.equal(a, c))
    shuffled = float((a - d).abs().max())  # pixel order changes GEMM blocking only
    return {"pass": repeat and reload and shuffled < 1e-5, "repeat_bit_identical": repeat,
            "reload_bit_identical": reload, "shuffled_order_max_abs_diff": shuffled}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    import os
    os.chdir(out)  # upstream RNA writes feature-grid dumps (yz_grid.exr) into the working directory
    tests = {"partition": lambda: t_partition(args.partitions), "routing": t_routing,
             "single_unit_equivalence": t_single_unit_equivalence, "isolation_and_checkpoint": lambda: t_isolation_and_checkpoint(out),
             "composition_reproducible": lambda: t_composition_reproducible(out)}
    res = {}
    for n, fn in tests.items():
        try:
            r = fn()
        except Exception:
            r = {"pass": False, "error": traceback.format_exc()}
        res[n] = r
        print(n, "PASS" if r["pass"] else "FAIL", flush=True)
    (out / "tests.json").write_text(json.dumps(res, indent=1))
    return 0 if all(r["pass"] for r in res.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
