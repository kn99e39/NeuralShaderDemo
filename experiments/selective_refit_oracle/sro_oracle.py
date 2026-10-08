"""Offline reference-derived oracle affected-state mask (protocol `oracle`).

    windows/run.ps1 ../selective_refit_oracle/sro_oracle.py --out <dir> [--seed 30030] [--smoke]

Traces one upstream reload of 8DNA training rays in the T0 and in the T3 teaset
with common random numbers, flags every ray whose supervision fields differ,
and accumulates per triplane cell (exact upstream bilinear stencil, integer
weights so the sums are deterministic):

    W3 = support weight of valid T3 samples, A3 = of affected valid T3 samples,
    W0 / A0 = the same for T0 samples, A = A3 + A0.

M95 = smallest prefix of cells ranked by A (desc, ties by index) covering 95%
of sum(A).  Also writes the equal-size spatial control mask S, sensitivity
sizes, decompositions and the oracle's own cost.  Not a runtime mechanism.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import sro_common as S

Q = 65536  # weight quantisation (protocol oracle.cell_accumulation)
TOL = 1e-5


def affected_flags(a: dict, b: dict):
    """Per-ray: does the supervision differ between two traces of the same random inputs?"""
    import torch

    diff = (a["valid"] != b["valid"]) | (a["occluded"] != b["occluded"]) | (a["depth"] != b["depth"])
    near_nonzero = torch.zeros_like(diff)
    for k in ("xi", "xo", "wo"):
        dk = (a[k] - b[k]).abs().amax(-1)
        diff |= dk > TOL
        near_nonzero |= (dk > 0) & (dk <= TOL)
    dt = (a["throughput"] - b["throughput"]).abs()
    lim = TOL * (1 + a["throughput"].abs())
    diff |= (dt > lim).any(-1)
    near_nonzero |= ((dt > 0) & (dt <= lim)).any(-1)
    return diff, near_nonzero & ~diff


def accumulate(acc, xi, sel, N):
    import torch

    if sel.any():
        idx, w = S.stencil(xi[sel], N)
        wq = torch.round(w * Q).long()
        acc.index_add_(0, idx.reshape(-1), wq.reshape(-1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="4 of 64 chunks only; never evidence")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    t_start = time.perf_counter()
    proto = S.protocol()
    orc = proto["oracle"]
    seed = args.seed if args.seed is not None else orc["seeds"]["primary"]
    import nrc_records as R
    import ednalib as L

    git = R.git_state(L.ROOT)
    R.require_clean(git, args.allow_dirty or args.smoke)
    mi, dr = S.init()
    import numpy as np
    import torch
    import teaset_parts as T
    from sro_paths import PathGen, draw_chunk, n_chunks, strata

    out = S.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    st = S.states()
    N_ds, H_ds, SPP = 128, 128, 256
    ck = torch.load(S.released_checkpoint(), map_location="cpu", weights_only=False)
    Ngrid = ck["hyper_parameters"]["model"]["encode_xi"]["N"]
    ncell = 3 * Ngrid * Ngrid

    timing = {}
    t0 = time.perf_counter()
    gen0 = PathGen(T.scene_dict(H_ds, st["T0"]), H_ds)
    gen3 = PathGen(T.scene_dict(H_ds, st["T3"]), H_ds)
    timing["scene_load_s"] = time.perf_counter() - t0
    bb0, bb3 = gen0.bbox, gen3.bbox
    if any(abs(float(getattr(bb0, k)[i]) - float(getattr(bb3, k)[i])) > 0 for k in ("min", "max") for i in range(3)):
        raise SystemExit("T0/T3 asset bboxes differ; xo projection would differ for every ray")

    acc = {k: torch.zeros(ncell, dtype=torch.long, device="cuda") for k in ("W3", "A3", "W0", "A0")}
    thr_aff = torch.zeros(ncell, dtype=torch.float64, device="cuda")   # throughput-weighted (secondary, report only)
    thr_all = torch.zeros(ncell, dtype=torch.float64, device="cuda")
    counts = {"rays": 0, "valid_T0": 0, "valid_T3": 0, "affected_any": 0, "affected_valid_T3": 0,
              "affected_valid_T0": 0, "near_tolerance_nonzero": 0, "valid_T3_nonzero_throughput": 0}
    sub = {"xi3_aff": [], "xi0_aff": [], "xi3_unaff": []}  # deterministic stride subsamples for decompositions
    torch.manual_seed(seed)
    s_xi, s_wi = strata(N_ds, H_ds)
    nch = n_chunks(N_ds, SPP)
    traced = 0
    timing["trace_T0_s"] = timing["trace_T3_s"] = timing["compare_accumulate_s"] = 0.0
    for idx in range(nch):
        # every chunk's random numbers are drawn in upstream order; smoke traces every 16th
        # chunk only (chunk index = cosine stratum of wi, so the first chunks are grazing)
        a_in, b_in, mseed = draw_chunk(idx, N_ds, H_ds, SPP, s_xi, s_wi)
        if args.smoke and idx % 16 != 8:
            continue
        traced += 1
        torch.cuda.synchronize(); t = time.perf_counter()
        f0 = gen0.trace(a_in, b_in, mseed, SPP)
        torch.cuda.synchronize(); timing["trace_T0_s"] += time.perf_counter() - t; t = time.perf_counter()
        f3 = gen3.trace(a_in, b_in, mseed, SPP)
        torch.cuda.synchronize(); timing["trace_T3_s"] += time.perf_counter() - t; t = time.perf_counter()
        aff, near = affected_flags(f0, f3)
        v0, v3 = f0["valid"], f3["valid"]
        accumulate(acc["W3"], f3["xi"], v3, Ngrid)
        accumulate(acc["A3"], f3["xi"], v3 & aff, Ngrid)
        accumulate(acc["W0"], f0["xi"], v0, Ngrid)
        accumulate(acc["A0"], f0["xi"], v0 & aff, Ngrid)
        lum = f3["throughput"].mean(-1).double()
        if v3.any():
            idx3, w3 = S.stencil(f3["xi"][v3], Ngrid)
            wl = (w3.double() * lum[v3, None, None]).reshape(-1)
            thr_all.index_add_(0, idx3.reshape(-1), wl)
            m = aff[v3]
            if m.any():
                idx3a, w3a = S.stencil(f3["xi"][v3 & aff], Ngrid)
                thr_aff.index_add_(0, idx3a.reshape(-1), (w3a.double() * lum[v3 & aff, None, None]).reshape(-1))
        counts["rays"] += int(aff.numel())
        counts["valid_T0"] += int(v0.sum()); counts["valid_T3"] += int(v3.sum())
        counts["affected_any"] += int(aff.sum())
        counts["affected_valid_T3"] += int((aff & v3).sum()); counts["affected_valid_T0"] += int((aff & v0).sum())
        counts["near_tolerance_nonzero"] += int(near.sum())
        counts["valid_T3_nonzero_throughput"] += int((v3 & (f3["throughput"].abs().amax(-1) > 0)).sum())
        stride = 64
        sub["xi3_aff"].append(f3["xi"][v3 & aff][::stride].cpu())
        sub["xi0_aff"].append(f0["xi"][v0 & aff][::stride].cpu())
        sub["xi3_unaff"].append(f3["xi"][v3 & ~aff][::stride * 16].cpu())
        torch.cuda.synchronize(); timing["compare_accumulate_s"] += time.perf_counter() - t
        del f0, f3
        if idx % 8 == 0:
            print(f"chunk {idx + 1}/{nch} affected so far {counts['affected_any']}", flush=True)

    t = time.perf_counter()
    arr = {k: v.cpu().numpy() for k, v in acc.items()}
    A = arr["A3"] + arr["A0"]
    order = np.lexsort((np.arange(ncell), -A))  # A desc, index asc
    cum = np.cumsum(A[order]) / max(A.sum(), 1)

    def prefix(frac):
        k = int(np.searchsorted(cum, frac) + 1) if A.sum() > 0 else 0
        m = np.zeros(ncell, bool)
        m[order[:k]] = True
        return m

    m95, m90, m99 = prefix(0.95), prefix(0.90), prefix(0.99)
    frac_cells = A / np.maximum(arr["W3"] + arr["A0"], 1)
    m_half = frac_cells >= 0.5
    timing["ranking_s"] = time.perf_counter() - t

    # equal-size spatial control: cell-centre distance in its plane to the projected milk pot (T0 and T3)
    t = time.perf_counter()
    from scipy.spatial import cKDTree
    from teaset_gt_states import surface_samples

    mover = "teapot2"
    pts = np.concatenate([surface_samples(mover, st["T0"]), surface_samples(mover, st["T3"])], 0)
    centres = S.cell_centres(Ngrid)
    dist = np.empty(ncell)
    for p, (a, b) in enumerate(S.PLANE_AXES):
        tree = cKDTree(pts[:, [a, b]])
        sl = slice(p * Ngrid * Ngrid, (p + 1) * Ngrid * Ngrid)
        dist[sl] = tree.query(centres[sl])[0]
    s_order = np.lexsort((np.arange(ncell), dist))
    m_spatial = np.zeros(ncell, bool)
    m_spatial[s_order[:int(m95.sum())]] = True
    timing["spatial_mask_s"] = time.perf_counter() - t

    # decompositions (stride subsamples): first-hit part by nearest part surface, distance to the mover
    t = time.perf_counter()
    parts = list(T.PARTS)
    trees = {(s, p): cKDTree(surface_samples(p, st[s])) for s in ("T0", "T3") for p in parts}

    def classify(x, s):
        d = np.stack([trees[(s, p)].query(x)[0] for p in parts], 1)
        return np.array(parts)[d.argmin(1)], d.min(1)

    def dist_mover(x):
        return np.minimum(trees[("T0", mover)].query(x)[0], trees[("T3", mover)].query(x)[0])

    bins = [0.05, 0.15, 0.30]

    def binned(dm):
        edges = [0.0] + bins + [np.inf]
        return {f"[{edges[i]}, {edges[i + 1]})": float(np.mean((dm >= edges[i]) & (dm < edges[i + 1])))
                for i in range(len(edges) - 1)} if len(dm) else {}

    decomp = {}
    for key, s in (("xi3_aff", "T3"), ("xi0_aff", "T0"), ("xi3_unaff", "T3")):
        x = torch.cat(sub[key]).numpy().astype(np.float64) if sub[key] else np.zeros((0, 3))
        if len(x):
            part, dpart = classify(x, s)
            dm = dist_mover(x)
            decomp[key] = {"n_subsample": int(len(x)), "part_share": {p: float(np.mean(part == p)) for p in parts},
                           "max_distance_to_assigned_part": float(dpart.max()),
                           "distance_to_mover_bins": binned(dm),
                           "share_beyond_0.15_from_mover": float(np.mean(dm >= 0.15)),
                           "share_beyond_0.30_from_mover": float(np.mean(dm >= 0.30))}
    timing["decomposition_s"] = time.perf_counter() - t

    def mask_stats(m):
        W3s, A3s, A0s = arr["W3"][m].sum(), arr["A3"][m].sum(), arr["A0"][m].sum()
        return {"cells": int(m.sum()), "fraction_of_cells": float(m.mean()),
                "params": int(m.sum()) * ck["hyper_parameters"]["model"]["encode_xi"]["C"],
                "coverage_A": float(A[m].sum() / max(A.sum(), 1)),
                "coverage_A3": float(arr["A3"][m].sum() / max(arr["A3"].sum(), 1)),
                "coverage_A0": float(arr["A0"][m].sum() / max(arr["A0"].sum(), 1)),
                "coverage_throughput_weighted_T3": float(thr_aff[torch.from_numpy(m).cuda()].sum() / max(float(thr_aff.sum()), 1e-30)),
                "support_share_of_all_T3": float(W3s / max(arr["W3"].sum(), 1)),
                "collateral_unaffected_share_of_selected_T3_support": float(1 - A3s / max(W3s, 1)),
                "per_plane_cells": [int(m[p * Ngrid * Ngrid:(p + 1) * Ngrid * Ngrid].sum()) for p in range(3)]}

    occupied = (arr["W3"] > 0) | (arr["W0"] > 0)
    rec = {
        "schema": "sro_oracle/v1", "seed": seed, "smoke": args.smoke, "git": git,
        "dataset": {"N": N_ds, "H": H_ds, "SPP": SPP, "chunks": nch, "chunks_traced": traced}, "grid": {"N": Ngrid, "cells": ncell},
        "counts": counts,
        "fractions": {"valid_T3_of_rays": counts["valid_T3"] / counts["rays"],
                      "affected_of_valid_T3": counts["affected_valid_T3"] / max(counts["valid_T3"], 1),
                      "affected_of_valid_T0": counts["affected_valid_T0"] / max(counts["valid_T0"], 1),
                      "near_tolerance_nonzero_of_rays": counts["near_tolerance_nonzero"] / counts["rays"]},
        "cells": {"occupied_T0_or_T3": int(occupied.sum()), "with_any_affected": int((A > 0).sum()),
                  "with_any_affected_fraction_of_occupied": float((A > 0).sum() / max(occupied.sum(), 1))},
        "masks": {"M95": mask_stats(m95), "M90": mask_stats(m90), "M99": mask_stats(m99),
                  "half_affected": mask_stats(m_half), "S_spatial": mask_stats(m_spatial)},
        "overlap_M95_S": {"intersection": int((m95 & m_spatial).sum()), "jaccard": float((m95 & m_spatial).sum() / max((m95 | m_spatial).sum(), 1))},
        "decomposition": decomp,
        "timing": dict(timing, total_s=time.perf_counter() - t_start),
        "environment": L.environment_record(),
    }
    np.savez_compressed(out / "oracle_cells.npz", W3=arr["W3"], A3=arr["A3"], W0=arr["W0"], A0=arr["A0"],
                        thr_aff=thr_aff.cpu().numpy(), thr_all=thr_all.cpu().numpy(),
                        M95=m95, M90=m90, M99=m99, half=m_half, S_spatial=m_spatial, spatial_distance=dist)
    S.write_json(out / "oracle.json", rec)
    print({k: rec["masks"][k]["cells"] for k in rec["masks"]}, rec["fractions"], f"{rec['timing']['total_s']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(rc)
