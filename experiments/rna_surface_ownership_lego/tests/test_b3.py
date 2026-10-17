"""Synthetic + mesh contract tests for Batch 3 (RNA venv, GPU). Contracts only; never architecture evidence.

    python tests/test_b3.py --partition <partition_L16.npz> --mesh <lego_mesh.npz> --out <dir>
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "rna_surface_ownership"))
import b3_compose as C  # noqa: E402
import b3_model as B  # noqa: E402
import rso_model as M  # noqa: E402

BASE = {"resolution": [64, 64], "triplane_features_num_channels": 8, "triplane_features_min_sigma": 1.0,
        "initial_sigma": 8.0, "iterations_to_sigma_1": 20000, "mlp_num_hidden_layers": 2, "mlp_hidden_layer_neurons": 32}


def t_owner_on_mesh(partition, mesh):
    """Points sampled ON the official mesh (fresh seed) resolve to their own surface: distance ~0, unit = the
    k-means cell of the point, component = the sampled triangle's; deterministic."""
    z = np.load(mesh); p = np.load(partition)
    V, F, comp, area = z["V"], z["F"], z["comp"], z["area"]
    rng = np.random.default_rng(123)
    t = rng.choice(len(F), 20000, p=area / area.sum())
    u, v = rng.random(20000), rng.random(20000); fl = u + v > 1; u[fl], v[fl] = 1 - u[fl], 1 - v[fl]
    a, b, c = V[F[t, 0]], V[F[t, 1]], V[F[t, 2]]
    x = a + (b - a) * u[:, None] + (c - a) * v[:, None]
    n = np.cross(b - a, c - a); n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    own = B.LegoOwner(partition, mesh)
    lab, d, (tri, cmp_, q) = own(x, n, return_tri=True)
    lab2, _ = own(x, n)
    cell = ((x[:, None, :] - p["centres"][None]) ** 2).sum(-1).argmin(1)
    ok_d = float(np.quantile(d, 0.999)); same_comp = float((cmp_ == comp[t]).mean()); same_cell = float((lab == cell).mean())
    # coincident faces (e.g. hidden contact faces of touching bricks) are geometrically indistinguishable, so
    # component agreement is reported, not required; the contract is that the chosen triangle contains the point
    # tolerance: half the dev-view pixel footprint (2.25 mm); first set to 1e-4 m, measured 0.55 mm (recorded in the worklog)
    return {"pass": ok_d < 0.5 * 2.25e-3 and same_cell > 0.999 and np.array_equal(lab, lab2),
            "dist_q999": ok_d, "component_agreement": same_comp, "cell_agreement": same_cell, "deterministic": bool(np.array_equal(lab, lab2))}


def t_owner_stacked_surfaces():
    """A floor (z = 0, component 0) meeting a wall (x = 0, component 1) at a corner. A point 0.4 mm above the
    floor and 0.3 mm from the wall is nearer to the WALL by position; rendered on the floor (normal +z) it
    must resolve to the floor, rendered on the wall (normal +x) to the wall. The test is orientation-agnostic
    (|n_t . n|), as the owner is: mesh winding is not assumed consistent."""
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]], float)
    F = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    comp = np.array([0, 0, 1, 1]); area = np.full(4, 0.5)
    d = tempfile.mkdtemp()
    np.savez(Path(d) / "m.npz", V=V, F=F, comp=comp, area=area, material=np.zeros(4, int))
    np.savez(Path(d) / "p.npz", centres=np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]]), units=np.int64(2))
    own = B.LegoOwner(Path(d) / "p.npz", Path(d) / "m.npz")
    x = np.array([[3e-4, 0.5, 4e-4]])
    _, _, (_, c_floor, _) = own(x, np.array([[0.0, 0.0, 1.0]]), return_tri=True)
    _, _, (_, c_wall, _) = own(x, np.array([[1.0, 0.0, 0.0]]), return_tri=True)
    _, _, (_, c_none, _) = own(x, None, return_tri=True)
    ok = int(c_floor[0]) == 0 and int(c_wall[0]) == 1 and int(c_none[0]) == 1
    return {"pass": ok, "floor_normal_component": int(c_floor[0]), "wall_normal_component": int(c_wall[0]), "position_only_component": int(c_none[0])}


def composer_fixture(K=2, seed=0):
    import torch

    specs = []
    for k in range(K):
        lo = np.array([-1 + 2 * k / K, -0.1, -0.5]); hi = lo + np.array([2 / K, 0.2, 1.0])
        specs.append({"params": dict(BASE), "lo": lo - 0.02, "hi": hi + 0.02})
    torch.manual_seed(seed)
    m = M.OwnedTriplaneSet(specs, ((-1.0, -0.1, -0.5), (1.0, 0.1, 0.5))).cuda().eval(); m.set_iteration_index(10 ** 9)
    centres = np.array([[-1 + 2 * (k + 0.5) / K, 0, 0] for k in range(K)])
    xs = np.random.default_rng(seed).uniform(-1, 1, (5000, 3)); lab = ((xs[:, 0] + 1) / (2 / K)).astype(int).clip(0, K - 1)
    comp = np.zeros(5000, int)  # one continuous component spanning every unit
    return C.LegoPUComposer(m, centres, lab, comp, 0.05), centres


def queries(n, seed, xs=None):
    rng = np.random.default_rng(seed)
    p = rng.uniform((-1, -0.1, -0.5), (1, 0.1, 0.5), (n, 3)); p[:, 1] = 0
    if xs is not None:
        p[:, 0] = xs; p[:, 2] = 0.1
    d = rng.normal(size=(3, n, 3)); d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return p.astype(np.float32), d[0].astype(np.float32), d[1].astype(np.float32), d[2].astype(np.float32), np.ones(n, bool)


def t_known_functions_and_components():
    import torch

    comp_, centres = composer_fixture(K=2)
    with torch.no_grad():
        for k, u in enumerate(comp_.model.units):
            last = [mm for mm in u.rgb_network.modules() if isinstance(mm, torch.nn.Linear)][-1]
            last.weight.zero_(); last.bias.fill_([0.2, 0.8][k])
    xs = np.linspace(-0.3, 0.3, 6001)
    p, c, l, n, v = queries(len(xs), 1, xs)
    own = ((p[:, 0] + 1) / 1.0).astype(int).clip(0, 1)
    out, parts = comp_(p, c, l, n, v, own, np.zeros(len(p), int), p.astype(np.float64), return_parts=True)
    hard_step = float(np.abs(np.diff(parts["hard"][:, 0])).max()); pu_step = float(np.abs(np.diff(out[:, 0])).max())
    outside = parts["w"] == 0
    exact_outside = bool(np.array_equal(out[outside], parts["hard"][outside]))
    # same geometry but each side in its own component -> no blending anywhere
    comp2 = np.where(own == 0, 0, 1)
    comp_.comp_units[:] = False; comp_.comp_units[0, 0] = True; comp_.comp_units[1, 1] = True
    out2, parts2 = comp_(p, c, l, n, v, own, comp2, p.astype(np.float64), return_parts=True)
    no_blend = bool((parts2["w"] == 0).all() and np.array_equal(out2, parts2["hard"]))
    ok = hard_step > 0.5 and pu_step < 0.05 * hard_step and exact_outside and no_blend
    return {"pass": ok, "hard_max_step": hard_step, "pu_max_step": pu_step, "outside_band_exact": exact_outside, "different_components_never_blend": no_blend}


def t_influence_and_reconstruction(tmp):
    import torch

    comp_, _ = composer_fixture(K=3, seed=4)
    m = comp_.model
    p, c, l, n, v = queries(30000, 5)
    own = ((p[:, 0] + 1) / (2 / 3)).astype(int).clip(0, 2); cp = np.zeros(len(p), int); q = p.astype(np.float64)
    before = comp_(p, c, l, n, v, own, cp, q); sd0 = copy.deepcopy(m.state_dict())
    m.units[1].apply(M.rmodels.init_gaussian)
    for k, u in enumerate(m.units):
        u.requires_grad_(k == 1)
    opt = torch.optim.Adam([x for x in m.parameters() if x.requires_grad], lr=1e-2); m.train()
    for it in range(10):
        pp, cc, ll, nn, _ = queries(4000, 50 + it); oo = ((pp[:, 0] + 1) / (2 / 3)).astype(int).clip(0, 2)
        pos = (pp - np.array([-1, -0.1, -0.5])) / np.array([2, 0.2, 1.0])
        T = lambda a: torch.tensor(np.asarray(a, np.float32), device="cuda").unsqueeze(0)
        loss = m(T(np.concatenate([pos, oo[:, None]], 1)), T(cc), T(ll), T(nn)).square().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); m.set_iteration_index(10 ** 9)
    frozen = all(torch.equal(m.state_dict()[k], sd0[k]) for k in sd0 if not k.startswith("units.1."))
    after = comp_(p, c, l, n, v, own, cp, q)
    changed = (after != before).any(1); infl = comp_.influenced_by(1, own, cp, q)
    exact = bool(np.array_equal(changed, infl)); nbr_band = int((infl & (own != 1)).sum())
    return {"pass": frozen and exact and nbr_band > 0, "other_units_bit_identical": frozen, "changed_equals_S1_union_B1": exact,
            "neighbour_band_points": nbr_band, "influenced": int(infl.sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)
    tests = {"owner_on_mesh": lambda: t_owner_on_mesh(args.partition, args.mesh), "owner_stacked_surfaces": t_owner_stacked_surfaces,
             "known_functions_and_components": t_known_functions_and_components, "influence_and_reconstruction": lambda: t_influence_and_reconstruction(out)}
    res = {}
    for name, fn in tests.items():
        try:
            r = fn()
        except Exception:
            r = {"pass": False, "error": traceback.format_exc()}
        res[name] = r
        print(name, "PASS" if r["pass"] else "FAIL", {k: v for k, v in r.items() if k not in ("pass", "error")}, r.get("error", ""), flush=True)
    (out / "tests.json").write_text(json.dumps(res, indent=1))
    return 0 if all(r["pass"] for r in res.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
