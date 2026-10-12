"""Focused tests for the surface-owned radiance units (worklog 31).

    windows/run.ps1 ../surface_owned_radiance/tests/test_sor.py --out <dir>

Synthetic fixture = the real teaset units and fitting engine with the path tracer replaced by an
analytic radiance function, so representation/optimiser semantics are checked exactly.
Implementation semantics only; never architecture evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sor_common as S  # noqa: E402

H = 0.02
RES = 32


def _model():
    a = S.build_anchors(H, seed=0)
    return a, S.SurfaceRadiance(a, RES)


def t_octahedral():
    import torch

    R = 64
    i = torch.arange(R, device="cuda")
    uv = (torch.stack(torch.meshgrid(i, i, indexing="xy"), -1).reshape(-1, 2).float() + 0.5) / R
    v = S.oct_decode(uv)
    back = S.oct_encode(v)
    err = float((back - uv).abs().max())
    unit = float((v.norm(dim=-1) - 1).abs().max())
    return {"pass": err < 1e-5 and unit < 1e-5, "roundtrip_max_err": err, "unit_norm_err": unit}


def t_stencil():
    import torch

    a, m = _model()
    g = torch.Generator(device="cuda").manual_seed(0)
    x, n, p = S.SurfaceSampler().positions(200_000, g)
    ids, w = m.stencil(x, p, n)
    has = ids[:, 0] >= 0
    s = w.sum(1)
    same_part = bool(((m.part[ids.clamp(min=0)] == p[:, None]) | (ids < 0)).all())
    gate = bool((((m.nrm[ids.clamp(min=0)] * n[:, None, :]).sum(-1) > S.NORMAL_GATE) | (ids < 0)).all())
    pou = float((s[has] - 1).abs().max())
    return {"pass": same_part and gate and pou < 1e-5 and float(has.float().mean()) > 0.995,
            "covered_fraction": float(has.float().mean()), "partition_of_unity_err": pou,
            "same_part_only": same_part, "normal_gate_respected": gate, "units": m.n_units}


def t_object_isolation():
    """Changing one object's table entries changes only queries on that object (no cross-object coupling)."""
    import torch

    a, m = _model()
    g = torch.Generator(device="cuda").manual_seed(1)
    x, n, p = S.SurfaceSampler().positions(200_000, g)
    w = torch.nn.functional.normalize(torch.randn_like(x), dim=-1)
    ids, wt = m.stencil(x, p, n)
    rows, ws = m.texel_stencil(ids, wt, w)
    t0 = torch.rand_like(m.table)
    out0 = m.query(rows, ws, t0)
    res = {}
    ok = True
    for i, part in enumerate(S.PARTS):
        t1 = t0.clone()
        sel = (m.part == i).repeat_interleave(RES * RES)
        t1[sel] += 1.0
        out1 = m.query(rows, ws, t1)
        changed = (out1 != out0).any(1)
        other = changed & (p != i)
        res[part] = {"queries_changed_on_other_objects": int(other.sum()),
                     "own_queries_changed_fraction": float(changed[p == i].float().mean())}
        ok &= int(other.sum()) == 0 and res[part]["own_queries_changed_fraction"] > 0.99
    # linearity: query(a t1 + b t2) = a query(t1) + b query(t2)
    t2 = torch.rand_like(t0)
    lin = float((m.query(rows, ws, 2 * t0 + 3 * t2) - (2 * out0 + 3 * m.query(rows, ws, t2))).abs().max())
    return {"pass": bool(ok and lin < 1e-4), "per_object": res, "linearity_err": lin}


def _analytic(q, edit_part=None):
    """Smooth analytic 'radiance' of position and direction; optionally changed on one part."""
    import torch

    x, w, p = q["x_local"], q["w"], q["part"]
    base = 0.3 + 0.2 * torch.sin(4 * x[:, :1] + 3 * w[:, 1:2]) + 0.1 * w[:, 2:3] ** 2 + 0.05 * torch.cos(5 * x[:, 2:3])
    y = base.repeat(1, 3) * torch.tensor([1.0, 0.8, 0.6], device=x.device)
    if edit_part is not None:
        y = torch.where((p == S.PARTS.index(edit_part))[:, None], y + 0.25 * (w[:, :1] > 0).float(), y)
    return y


def t_fit_semantics():
    """Synthetic fixture: fit all units to the analytic field, then 'edit' one object and refit only
    that object's units (whole-object update); others must stay bitwise identical and the edited
    object must recover."""
    import torch
    import sor_fit as F
    import sor_targets as TG

    a, m = _model()
    edit = {"part": None}

    def fake_trace(scene, ids, q, seed, spp=1):
        y = _analytic(q, edit["part"])
        return torch.zeros_like(y), y, torch.ones(len(y), dtype=torch.bool, device=y.device)

    real = TG.trace_t
    TG.trace_t = fake_trace
    try:
        pool = F.PositionPool(m, 4_000_000, seed=2)
        tab0, info0 = F.fit(m, None, None, {}, pool, 150, 262144, 0.02, seed=1)
        g = torch.Generator(device="cuda").manual_seed(9)
        x, n, p = S.SurfaceSampler().positions(100_000, g)
        q = TG.directions(x, n, p, g, {})
        ids, wt = m.stencil(q["x_local"], q["part"], q["n"])
        rows, ws = m.texel_stencil(ids, wt, q["w"])
        e0 = float((m.query(rows, ws, tab0) - _analytic(q)).abs().mean())
        edit["part"] = "teapot4"
        mask = torch.tensor((a["part"] == S.PARTS.index("teapot4")), device="cuda")
        pool_e = F.PositionPool(m, 2_000_000, seed=3, unit_mask=mask)
        tab1, info1 = F.fit(m, None, None, {}, pool_e, 150, 262144, 0.02, seed=2, unit_mask=mask, table0=tab0)
        fr = (~mask).repeat_interleave(RES * RES)
        frozen_ok = bool(torch.equal(tab1[fr], tab0[fr]))
        on = q["part"] == S.PARTS.index("teapot4")
        y1 = _analytic(q, "teapot4")
        pred1 = m.query(rows, ws, tab1)
        e_edit_before = float((m.query(rows, ws, tab0)[on] - y1[on]).abs().mean())
        e_edit_after = float((pred1[on] - y1[on]).abs().mean())
        e_other = float((pred1[~on] - y1[~on]).abs().mean())
        out_other_identical = bool(torch.equal(pred1[~on], m.query(rows, ws, tab0)[~on]))
    finally:
        TG.trace_t = real
    # e0 bound is loose on purpose: 150 noise-free steps cover 39 M samples for 63 M entries, so the
    # global fit is budget-limited; the semantics under test are isolation, bitwise freezing and the
    # edited object's recovery (whose units get ~4x the per-unit samples of the global fit)
    ok = (e0 < 0.1 and frozen_ok and out_other_identical and e_edit_after < 0.3 * e_edit_before)
    return {"pass": bool(ok), "fit_mae_T0_field": e0, "frozen_entries_bit_identical": frozen_ok,
            "other_object_outputs_identical": out_other_identical, "edited_object_mae_before": e_edit_before,
            "edited_object_mae_after": e_edit_after, "other_objects_mae": e_other,
            "support_fraction_teapot4": pool_e.support_fraction}


def t_combined_lookup():
    import torch
    import sor_fit as F

    T = torch.rand(1000, 3, device="cuda")
    tr = torch.zeros(1000, dtype=torch.bool, device="cuda"); tr[::3] = True
    remap = torch.full((1000,), -1, dtype=torch.long, device="cuda")
    idx = torch.nonzero(tr).squeeze(1); remap[idx] = torch.arange(len(idx), device="cuda")
    sub = torch.rand(len(idx), 3, device="cuda")
    full = T.clone(); full[idx] = sub
    rows = torch.randint(0, 1000, (5000, 8), device="cuda")
    ok = torch.equal(F.combined_lookup(rows, T, sub, remap), full[rows])
    return {"pass": bool(ok)}


def t_trace_equivalence():
    """GPU query path (directions + trace_t) vs numpy path (sample_queries + trace): acceptance and mean."""
    import numpy as np
    import torch
    from sor_targets import directions, sample_queries, scene_for, trace, trace_t

    scene, ids, tr = scene_for("T0")
    g = torch.Generator(device="cuda").manual_seed(4)
    x, n, p = S.SurfaceSampler().positions(1_048_576, g)
    _, y, ok = trace_t(scene, ids, directions(x, n, p, g, tr), seed=5)
    q2 = sample_queries(1_048_576, np.random.default_rng(6), tr)
    _, y2, ok2 = trace(scene, ids, q2, seed=7)
    m1, m2 = y[ok].mean(0).cpu().numpy(), y2[ok2].mean(0)
    se = y[ok].std(0).cpu().numpy() / np.sqrt(int(ok.sum()))
    z = np.abs(m1 - m2) / (np.sqrt(2) * se)
    return {"pass": bool(abs(float(ok.float().mean()) - float(ok2.mean())) < 0.002 and (z < 4).all()),
            "accept_gpu": float(ok.float().mean()), "accept_np": float(ok2.mean()), "mean_gpu": m1.tolist(),
            "mean_np": m2.tolist(), "z": z.tolist()}


def t_data_separation():
    p = S.protocol()
    seeds = {"oracle": p["oracle"]["seeds"]["primary"], "oracle_rep": p["oracle"]["seeds"]["replicate"],
             "heldout_T0": 51050, "heldout_T3": 51051, "pretrain": p["fit_contract"]["pretrain_T0"]["seed"]}
    distinct = len(set(seeds.values())) == len(seeds)
    arm = (S.HERE / "sor_arm.py").read_text(encoding="utf-8")
    fitsrc = (S.HERE / "sor_fit.py").read_text(encoding="utf-8")
    no_refs = all(s not in arm + fitsrc for s in ("gt_A", "w21_reference_correction", "regions.npz", "image_T3"))
    return {"pass": bool(distinct and no_refs), "seeds": seeds, "fit_and_arm_read_no_reference": no_refs}


TESTS = {"octahedral": t_octahedral, "stencil": t_stencil, "object_isolation": t_object_isolation,
         "combined_lookup": t_combined_lookup, "fit_semantics_synthetic": t_fit_semantics,
         "trace_equivalence": t_trace_equivalence, "data_separation": t_data_separation}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    S.init()
    import nrc_records as R
    import ednalib as L

    res = {}
    for name in args.only or list(TESTS):
        t = time.perf_counter()
        try:
            r = TESTS[name]()
        except Exception:
            r = {"pass": False, "error": traceback.format_exc()}
        r["seconds"] = time.perf_counter() - t
        res[name] = r
        print(name, "PASS" if r["pass"] else "FAIL", flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    S.write_json(out / "tests.json", dict(res, git=R.git_state(L.ROOT)))
    return 0 if all(v["pass"] for v in res.values()) else 1


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(rc)
