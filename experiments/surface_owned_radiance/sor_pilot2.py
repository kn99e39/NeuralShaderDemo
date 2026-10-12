"""T0-only capacity pilot on the online fitting engine (never reads T3 data).

    windows/run.ps1 ../surface_owned_radiance/sor_pilot2.py --out <dir> --h 0.02 --res 32 --steps 400

Reports held-out (16 spp) relative MAE of the >= 2-vertex radiance and camera-view errors at T0
against the worklog-24 class images, at several points of the fit, plus memory and time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import sor_common as S


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--h", type=float, default=0.02)
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=1_048_576)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--pool", type=int, default=32_000_000)
    ap.add_argument("--evals", type=int, nargs="*", default=None)
    args = ap.parse_args()
    mi, dr = S.init()
    import numpy as np
    import torch
    import ednalib as L
    from sor_fit import PositionPool, fit, render_indirect
    from sor_targets import camera_queries, directions, scene_for, trace_t
    from teaset_frozen_eval import load_rois

    out = S.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rec = {"args": vars(args), "evals": []}
    units = S.build_anchors(args.h)
    model = S.SurfaceRadiance(units, args.res)
    rec["units"] = {"total": model.n_units, "per_part": {p: units["grids"][p]["count"] for p in S.PARTS},
                    "table_entries": model.n_units * args.res ** 2 * 3}
    print(rec["units"], flush=True)
    scene, ids, tr = scene_for("T0")
    pool = PositionPool(model, args.pool, seed=7)
    # held-out queries, 16 spp, separate generator
    gh = torch.Generator(device="cuda").manual_seed(99)
    x, n, p = S.SurfaceSampler().positions(300_000, gh)
    qh = directions(x, n, p, gh, tr)
    _, yh, okh = trace_t(scene, ids, qh, seed=424242, spp=16)
    s = torch.nonzero(okh).squeeze(1)
    ih, wh = model.stencil(qh["x_local"][s], qh["part"][s], qh["n"][s])
    rh, whh = model.texel_stencil(ih, wh, qh["w"][s])
    yh = yh[s]
    scene512, ids512, _ = scene_for("T0", 512)
    cam = camera_queries(scene512, ids512, tr, 512, 4, seed=5)
    z = np.load(S.ROOT / "results/8dna_replication/transport_decomposition/w21_envmap/image_T0.npz")
    ref_ind = z["classes"][:, :, 2:9].sum(2); ref_dir = z["classes"][:, :, 0] + z["classes"][:, :, 1]
    reg = np.load(S.ROOT / "results/selective_refit_oracle/v1_bc085ea/regions/regions.npz")
    refd = S.ROOT / "results/8dna_replication/w21_reference_correction/T0"
    g0 = 0.5 * (L.load_exr(refd / "gt_A.exr") + L.load_exr(refd / "gt_B.exr"))
    rois = load_rois(S.ROOT / "results/8dna_replication/gt_design/v2", ("T0",))
    masks = {"interaction": rois["T0"]["interaction"][0].reshape(512, 512), "R_aff_stat": reg["R_aff_stat"],
             "R_unaff": reg["R_unaff"], "asset": ~reg["background"]}
    n8 = L.load_exr(S.ROOT / "results/8dna_replication/frozen/teaset_locked/T0/8dna_upstream.exr")
    disp = L.tonemap
    rec["8dna_T0_display_mae"] = {k: float(np.abs(disp(n8) - disp(g0)).mean(-1)[m].mean()) for k, m in masks.items()}
    rec["ref_decomposition_display_mae"] = {k: float(np.abs(disp(ref_dir + ref_ind) - disp(g0)).mean(-1)[m].mean()) for k, m in masks.items()}
    evals = args.evals or sorted({max(1, args.steps // 8), args.steps // 4, args.steps // 2, args.steps})
    t_start = time.time()

    def on_snap(step, table, tim):
        with torch.no_grad():
            ph = (table[rh] * whh[..., None]).sum(1)
            rel = float((ph - yh).abs().mean() / yh.abs().mean())
        img = render_indirect(model, table, cam)
        tot = ref_dir + img
        e = {"step": step, "wall_s": time.time() - t_start, **tim, "heldout_rel_mae": rel,
             "indirect_linear_mae": {k: float(np.abs(img - ref_ind)[m].mean()) for k, m in masks.items()},
             "total_display_mae": {k: float(np.abs(disp(tot) - disp(g0)).mean(-1)[m].mean()) for k, m in masks.items()}}
        rec["evals"].append(e)
        print(json.dumps(e), flush=True)
        if step == args.steps:
            L.save_exr(out / "indirect_T0.exr", img.astype(np.float32))
            L.save_exr(out / "total_T0.exr", tot.astype(np.float32))

    table, info = fit(model, scene, ids, tr, pool, args.steps, args.batch, args.lr, seed=3,
                      snapshot_steps=evals, on_snapshot=on_snap)
    rec["fit"] = info
    rec["pool"] = {"seconds": pool.seconds}
    rec["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    S.write_json(out / "pilot.json", rec)
    print("8dna", rec["8dna_T0_display_mae"], "peak", rec["peak_mem_gib"], flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(rc)
