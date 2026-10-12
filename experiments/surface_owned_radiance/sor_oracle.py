"""Offline reference-derived oracle over surface-owned units (protocol `oracle`).

    windows/run.ps1 ../surface_owned_radiance/sor_oracle.py --units <units.npz> --out <dir> [--seed 31031]

Dense surface queries (same distribution as training) are traced in T0 and T3 with common
random numbers (same rays in the canonical frame, same sampler seeds).  A unit owns the outgoing
>= 2-vertex radiance of its stencil, so its invalidation weight is the radiometric change it
owns:  E_u = sum_q  w_qu * mean_c |L_T3(q) - L_T0(q)|.   M95 = smallest top-E set covering 95%
of sum E.  Also: path-count weights (rays whose traces differ), per-part and distance-to-mover
decompositions, an equal-size proximity mask, and the oracle's own cost.  Not a runtime mechanism.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import sor_common as S


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    t_start = time.perf_counter()
    import nrc_records as R
    import ednalib as L

    git = R.git_state(L.ROOT)
    R.require_clean(git, args.allow_dirty)
    mi, dr = S.init()
    import numpy as np
    import torch
    from scipy.spatial import cKDTree
    from sor_targets import sample_queries, scene_for, trace
    from teaset_gt_states import surface_samples

    proto = S.protocol()
    oc = proto["oracle"]
    seed = args.seed if args.seed is not None else oc["seeds"]["primary"]
    n_total = args.n or oc["queries"]
    out = S.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    units = S.load_units(args.units)
    model = S.SurfaceRadiance(units, 1)
    U = model.n_units
    st = S.states()
    s0, ids0, tr0 = scene_for("T0")
    s3, ids3, tr3 = scene_for("T3")
    E = np.zeros(U); W = np.zeros(U); Ecount = np.zeros(U); Wcount = np.zeros(U)
    rng = np.random.default_rng(seed)
    timing = {"trace_T0_s": 0.0, "trace_T3_s": 0.0, "accumulate_s": 0.0}
    chunk = 2_000_000
    n_aff = 0; n_ok = 0
    dq_sub, d_sub = [], []
    for c0 in range(0, n_total, chunk):
        n = min(chunk, n_total - c0)
        q = sample_queries(n, rng, tr0)                       # canonical queries (T0 world = canonical)
        q3 = dict(q)
        q3["x_world"] = q["x_local"] + np.stack([np.asarray(tr3.get(S.PARTS[i], (0, 0, 0)), float) for i in q["part"]])
        t = time.perf_counter(); _, y0, ok0 = trace(s0, ids0, q, seed=seed * 7919 + c0); timing["trace_T0_s"] += time.perf_counter() - t
        t = time.perf_counter(); _, y3, ok3 = trace(s3, ids3, q3, seed=seed * 7919 + c0); timing["trace_T3_s"] += time.perf_counter() - t
        t = time.perf_counter()
        ok = ok0 & ok3
        d = np.abs(y3 - y0).mean(1)
        aff = d > 0
        sel = np.flatnonzero(ok)
        xl = torch.tensor(q["x_local"][sel], dtype=torch.float32, device="cuda")
        ids, wts = model.stencil(xl, torch.tensor(q["part"][sel], device="cuda"), torch.tensor(q["n"][sel], dtype=torch.float32, device="cuda"))
        ids, wts = ids.cpu().numpy(), wts.cpu().numpy()
        m = ids >= 0
        mag = np.abs(y3[sel]).mean(1)
        np.add.at(E, ids[m], (wts * d[sel][:, None])[m])
        np.add.at(W, ids[m], (wts * mag[:, None])[m])
        np.add.at(Ecount, ids[m], (wts * aff[sel][:, None])[m])
        np.add.at(Wcount, ids[m], wts[m])
        n_aff += int(aff[sel].sum()); n_ok += len(sel)
        pick = sel[aff[sel]][::50]
        dq_sub.append(np.c_[q["part"][pick], q["x_local"][pick], d[pick]])
        timing["accumulate_s"] += time.perf_counter() - t
        print(f"oracle {c0 + n}/{n_total} affected {n_aff}/{n_ok}", flush=True)

    t = time.perf_counter()
    order = np.lexsort((np.arange(U), -E))
    cum = np.cumsum(E[order]) / max(E.sum(), 1e-30)

    def prefix(frac):
        k = int(np.searchsorted(cum, frac) + 1)
        m = np.zeros(U, bool); m[order[:k]] = True
        return m

    masks = {"M90": prefix(0.90), "M95": prefix(0.95), "M99": prefix(0.99)}
    # equal-size proximity mask: units nearest the mover (T0 or T3 position), same count as M95
    mover_pts = np.concatenate([surface_samples(S.MOVER, st["T0"]), surface_samples(S.MOVER, st["T3"])], 0)
    pos_world3 = units["pos"] + np.stack([np.asarray(tr3.get(S.PARTS[i], (0, 0, 0)), float) for i in units["part"]])
    pos_world0 = units["pos"]
    dist = np.minimum(cKDTree(mover_pts).query(pos_world3)[0], cKDTree(mover_pts).query(pos_world0)[0])
    is_mover = units["part"] == S.PARTS.index(S.MOVER)
    dist = np.where(is_mover, 0.0, dist)
    s_order = np.lexsort((np.arange(U), dist))
    masks["S_spatial"] = np.zeros(U, bool); masks["S_spatial"][s_order[:int(masks["M95"].sum())]] = True
    timing["ranking_s"] = time.perf_counter() - t

    areas = S.unit_areas(units)
    sub = np.concatenate(dq_sub)
    dm = np.minimum(cKDTree(mover_pts).query(sub[:, 1:4] + np.stack([np.asarray(tr3.get(S.PARTS[int(i)], (0, 0, 0)), float) for i in sub[:, 0]]))[0],
                    cKDTree(mover_pts).query(sub[:, 1:4])[0])
    wsub = sub[:, 4]

    def mstats(m):
        return {"units": int(m.sum()), "fraction_units": float(m.mean()), "fraction_area": float(areas[m].sum() / areas.sum()),
                "coverage_E": float(E[m].sum() / E.sum()), "coverage_count": float(Ecount[m].sum() / max(Ecount.sum(), 1e-30)),
                "per_part_units": {p: int((m & (units["part"] == i)).sum()) for i, p in enumerate(S.PARTS)},
                "mover_units": int((m & is_mover).sum())}

    part_u = units["part"]
    obj_E = {p: float(E[part_u == i].sum()) for i, p in enumerate(S.PARTS)}
    obj_share = {p: obj_E[p] / max(E.sum(), 1e-30) for p in S.PARTS}
    obj_mean = {p: float(E[part_u == i].sum() / max(Wcount[part_u == i].sum(), 1e-30)) for i, p in enumerate(S.PARTS)}
    obj_mag = {p: float(W[part_u == i].sum() / max(Wcount[part_u == i].sum(), 1e-30)) for i, p in enumerate(S.PARTS)}
    thr = oc["object_dirty_rule_threshold"]
    dirty = {f"{t}": [p for p in S.PARTS if obj_share[p] >= t] for t in [thr] + oc["object_dirty_sensitivity_thresholds"]}
    rec = {"schema": "sor_oracle/v1", "seed": seed, "queries": n_total, "git": git,
           "object_change_energy": obj_E, "object_change_share": obj_share,
           "object_mean_abs_change": obj_mean, "object_mean_radiance_T3": obj_mag,
           "object_relative_change": {p: obj_mean[p] / max(obj_mag[p], 1e-30) for p in S.PARTS},
           "dirty_objects": dirty[f"{thr}"], "dirty_objects_sensitivity": dirty, "geometry_dirty": oc["geometry_dirty"],
           "affected_fraction_of_accepted": n_aff / max(n_ok, 1), "accepted": n_ok,
           "units": U, "units_with_any_change": int((E > 0).sum()),
           "masks": {k: mstats(v) for k, v in masks.items()},
           "overlap_M95_S": float((masks["M95"] & masks["S_spatial"]).sum() / max((masks["M95"] | masks["S_spatial"]).sum(), 1)),
           "change_weight_by_distance_to_mover": {
               b: float(wsub[(dm >= lo) & (dm < hi)].sum() / max(wsub.sum(), 1e-30))
               for b, lo, hi in (("<0.05", 0, 0.05), ("0.05-0.15", 0.05, 0.15), ("0.15-0.30", 0.15, 0.30), (">=0.30", 0.30, 1e9))},
           "change_weight_by_part": {p: float(wsub[sub[:, 0] == i].sum() / max(wsub.sum(), 1e-30)) for i, p in enumerate(S.PARTS)},
           "timing": dict(timing, total_s=time.perf_counter() - t_start), "environment": L.environment_record()}
    np.savez_compressed(out / "oracle_units.npz", E=E, W=W, Ecount=Ecount, Wcount=Wcount, dist_to_mover=dist, **masks)
    S.write_json(out / "oracle.json", rec)
    print({k: v["units"] for k, v in rec["masks"].items()}, rec["affected_fraction_of_accepted"], flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(rc)
