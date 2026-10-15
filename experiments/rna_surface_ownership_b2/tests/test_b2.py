"""Synthetic contract tests for P16-PU composition (RNA venv, GPU). Contracts only; never architecture evidence.

    python tests/test_b2.py --out <dir>
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "rna_surface_ownership"))
import b2_compose as C  # noqa: E402
import rso_model as M  # noqa: E402

BASE = {"resolution": [64, 64], "triplane_features_num_channels": 8, "triplane_features_min_sigma": 1.0,
        "initial_sigma": 8.0, "iterations_to_sigma_1": 20000, "mlp_num_hidden_layers": 2, "mlp_hidden_layer_neurons": 32}
AABB = ((-1.0, -0.1, -0.5), (1.0, 0.1, 0.5))
H = 0.05


def fixture(seed=0, K=3):
    """K units side by side along x on one part (unit k owns x in [-1 + 2k/K, -1 + 2(k+1)/K)); dense samples on the
    plane y = 0; returns (composer, owner_fn)."""
    import torch

    rng = np.random.default_rng(seed)
    xs = rng.uniform(AABB[0], AABB[1], (200000, 3)); xs[:, 1] = 0
    lab = np.clip(((xs[:, 0] + 1) / (2 / K)).astype(int), 0, K - 1)
    specs = []
    for k in range(K):
        lo = np.array([-1 + 2 * k / K, -0.1, -0.5]); hi = lo + np.array([2 / K, 0.2, 1.0])
        lo, hi = lo - 0.02 * (hi - lo), hi + 0.02 * (hi - lo)
        specs.append({"params": dict(BASE), "lo": lo, "hi": hi})
    torch.manual_seed(seed)
    m = M.OwnedTriplaneSet(specs, AABB).cuda().eval(); m.set_iteration_index(10 ** 9)
    comp = C.PUComposer(m, xs, lab, np.zeros(K, int), H)
    owner_fn = lambda p: np.clip(((p[:, 0] + 1) / (2 / K)).astype(int), 0, K - 1)
    return comp, owner_fn


def queries(n, seed, xs=None):
    rng = np.random.default_rng(seed)
    p = rng.uniform(AABB[0], AABB[1], (n, 3)); p[:, 1] = 0
    if xs is not None:
        p[:, 0] = xs; p[:, 2] = 0.1
    d = rng.normal(size=(3, n, 3)); d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return p.astype(np.float32), d[0].astype(np.float32), d[1].astype(np.float32), d[2].astype(np.float32), np.ones(n, bool)


def t_weights():
    d_own = np.zeros(5); d_nbr = np.array([0, H / 2, H, 2 * H, 10 * H])
    w = C.pu_weight(d_own, d_nbr, H)
    ok = np.allclose(w, [0.5, 0.25, 0.0, 0.0, 0.0]) and (w >= 0).all() and (w <= 0.5).all()
    return {"pass": bool(ok), "w": w.tolist()}


def t_coverage_determinism():
    comp, own = fixture()
    p, c, l, n, v = queries(20000, 1)
    o = own(p)
    a = comp(p, c, l, n, v, o); b = comp(p, c, l, n, v, o)
    nbr, w, _, _ = comp.influence(p, o)
    ok = bool(np.isfinite(a).all() and np.array_equal(a, b) and (o >= 0).all() and ((w == 0) | (nbr >= 0)).all())
    return {"pass": ok, "repeat_bit_identical": bool(np.array_equal(a, b)), "all_points_owned": bool((o >= 0).all())}


def t_hard_equivalence_outside_band():
    """Outside the band the composite equals the unit's own prediction exactly; inside, the hard part equals
    OwnedTriplaneSet.forward (routing unchanged)."""
    import torch

    comp, own = fixture()
    p, c, l, n, v = queries(20000, 2)
    o = own(p)
    out, parts = comp(p, c, l, n, v, o, return_parts=True)
    outside = parts["w"] == 0
    exact = bool(np.array_equal(out[outside], parts["hard"][outside]))
    pos = (p - np.array(AABB[0])) / (np.array(AABB[1]) - np.array(AABB[0]))
    T = lambda a: torch.tensor(np.asarray(a, np.float32), device="cuda").unsqueeze(0)
    with torch.no_grad():
        f = comp.model(T(np.concatenate([pos, o[:, None]], 1)), T(c), T(l), T(n)).squeeze(0)[:, :3].cpu().numpy()
    routing = bool(np.allclose(f, parts["hard"], rtol=0, atol=1e-5))
    return {"pass": exact and routing, "outside_band_exact": exact, "hard_matches_forward": routing,
            "band_fraction": float((~outside).mean())}


def t_known_functions():
    """Units made constant (last layer weight 0, bias c_k): hard composition jumps by |c_0 - c_1| at the seam;
    PU is continuous (max step between adjacent samples << jump) and equals hard outside the band; equal
    constants -> PU == hard everywhere."""
    import torch

    comp, own = fixture(K=2)
    consts = [0.2, 0.8]
    with torch.no_grad():
        for k, u in enumerate(comp.model.units):
            last = [mm for mm in u.rgb_network.modules() if isinstance(mm, torch.nn.Linear)][-1]
            last.weight.zero_(); last.bias.fill_(consts[k])
    xs = np.linspace(-0.3, 0.3, 6001)
    p, c, l, n, v = queries(len(xs), 3, xs)
    o = own(p)
    out, parts = comp(p, c, l, n, v, o, return_parts=True)
    hard_step = float(np.abs(np.diff(parts["hard"][:, 0])).max()); pu_step = float(np.abs(np.diff(out[:, 0])).max())
    with torch.no_grad():
        for u in comp.model.units:
            [mm for mm in u.rgb_network.modules() if isinstance(mm, torch.nn.Linear)][-1].bias.fill_(0.5)
    out2, parts2 = comp(p, c, l, n, v, o, return_parts=True)
    same = bool(np.allclose(out2, parts2["hard"], atol=1e-6))
    ok = hard_step > 0.5 and pu_step < 0.05 * hard_step and same
    return {"pass": ok, "hard_max_step": hard_step, "pu_max_step": pu_step, "equal_constants_unchanged": same}


def t_influence_accounting_and_reconstruction(tmp):
    """Re-initialise + train unit 1 only: every other unit's parameters bit-identical; composite outputs change
    exactly on S_1 U B_1 and are bit-identical elsewhere; checkpoint round trip identical."""
    import torch

    comp, own = fixture(K=3)
    p, c, l, n, v = queries(30000, 4)
    o = own(p)
    before_out = comp(p, c, l, n, v, o)
    before_sd = copy.deepcopy(comp.model.state_dict())
    m = comp.model
    m.units[1].apply(M.rmodels.init_gaussian)
    for k, u in enumerate(m.units):
        u.requires_grad_(k == 1)
    opt = torch.optim.Adam([q for q in m.parameters() if q.requires_grad], lr=1e-2)
    m.train()
    for it in range(10):
        pp, cc, ll, nn, _ = queries(4000, 100 + it)
        oo = own(pp)
        pos = (pp - np.array(AABB[0])) / (np.array(AABB[1]) - np.array(AABB[0]))
        T = lambda a: torch.tensor(np.asarray(a, np.float32), device="cuda").unsqueeze(0)
        loss = m(T(np.concatenate([pos, oo[:, None]], 1)), T(cc), T(ll), T(nn)).square().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); m.set_iteration_index(10 ** 9)
    after_sd = m.state_dict()
    frozen = all(torch.equal(after_sd[k], before_sd[k]) for k in after_sd if not k.startswith("units.1."))
    after_out = comp(p, c, l, n, v, o)
    changed = (after_out != before_out).any(1)
    infl = comp.influenced_by(1, p, o)
    exact_set = bool(np.array_equal(changed, infl))
    outside_identical = bool(np.array_equal(after_out[~infl], before_out[~infl]))
    band_nbr = int(((o != 1) & infl).sum())
    path = Path(tmp) / "b2_state.pt"; torch.save(m.state_dict(), path)
    comp2, _ = fixture(K=3); comp2.model.load_state_dict(torch.load(path)); comp2.model.eval(); comp2.model.set_iteration_index(10 ** 9)
    for u in comp.model.units:  # last_sigma is not checkpointed (worklog 32); align as test_rso does
        for g in (u.triplane_grid.xy_feature_grid, u.triplane_grid.yz_feature_grid, u.triplane_grid.xz_feature_grid):
            g.last_sigma = g.min_sigma
    roundtrip = bool(np.array_equal(comp(p, c, l, n, v, o), comp2(p, c, l, n, v, o)))
    ok = frozen and exact_set and outside_identical and band_nbr > 0 and roundtrip
    return {"pass": ok, "other_units_bit_identical": frozen, "changed_set_equals_S1_union_B1": exact_set,
            "outputs_outside_influence_bit_identical": outside_identical, "neighbour_band_points_influenced": band_nbr,
            "points_influenced": int(infl.sum()), "points_owned_by_1": int((o == 1).sum()), "checkpoint_roundtrip_identical": roundtrip}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)  # upstream RNA writes yz_grid.exr into the working directory
    tests = {"weights": t_weights, "coverage_determinism": t_coverage_determinism,
             "hard_equivalence_outside_band": t_hard_equivalence_outside_band, "known_functions": t_known_functions,
             "influence_accounting_and_reconstruction": lambda: t_influence_accounting_and_reconstruction(out)}
    res = {}
    for name, fn in tests.items():
        try:
            r = fn()
        except Exception:
            r = {"pass": False, "error": traceback.format_exc()}
        res[name] = r
        print(name, "PASS" if r["pass"] else "FAIL", flush=True)
    (out / "tests.json").write_text(json.dumps(res, indent=1))
    return 0 if all(r["pass"] for r in res.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
