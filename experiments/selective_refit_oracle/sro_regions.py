"""Evaluation regions for the T3 frame (protocol `evaluation.regions`), from geometry and references only.

    windows/run.ps1 ../selective_refit_oracle/sro_regions.py --out <dir>

Writes regions.npz with 512x512 masks R_aff_stat, R_unaff, R_buffer, R_mover, R_aff,
R_unstable, stable_stationary, background, the per-pixel change c and noise n
(display, 7x7 box, channel mean), and the T3 first-hit positions on a 5x5
sub-pixel lattice (pixel edges included) for the learned-state stencil of each pixel.
Run before any training; it reads no model output.
"""

from __future__ import annotations

import argparse
import os
import sys

import sro_common as S


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    import nrc_records as R
    import ednalib as L

    git = R.git_state(L.ROOT)
    R.require_clean(git, args.allow_dirty)
    mi, dr = S.init()
    import numpy as np
    from scipy.ndimage import uniform_filter
    import teaset_parts as T
    from teaset_gt_states import part_map, primary_buffers

    proto = S.protocol()
    res = proto["scene"]["render"]["res"]
    st = S.states()
    out = S.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bufs, scenes = {}, {}
    for s in ("T0", "T3"):
        scenes[s] = mi.load_dict(T.scene_dict(res, st[s]))
        ids = T.part_shape_ids(scenes[s], st[s])
        sid, p, n = primary_buffers(scenes[s], res)
        bufs[s] = (part_map(sid, ids), p)
    pm0, p0 = bufs["T0"]
    pm3, p3 = bufs["T3"]
    mover = "teapot2"
    hit0, hit3 = pm0 != "", pm3 != ""
    stable = hit0 & hit3 & (pm0 == pm3) & (pm3 != mover) & (np.linalg.norm(p3 - p0, axis=-1) < 1e-5)

    ref = S.ROOT / "results" / "8dna_replication" / "w21_reference_correction"
    gA = {s: L.load_exr(ref / s / "gt_A.exr") for s in ("T0", "T3")}
    gB = {s: L.load_exr(ref / s / "gt_B.exr") for s in ("T0", "T3")}
    disp = L.tonemap
    g0, g3 = 0.5 * (gA["T0"] + gB["T0"]), 0.5 * (gA["T3"] + gB["T3"])
    c = uniform_filter(np.abs(disp(g3) - disp(g0)).mean(-1), 7)
    n = uniform_filter(np.abs(disp(gA["T3"]) - disp(gB["T3"])).mean(-1), 7)
    R_aff_stat = stable & (c > 3 * n)
    R_unaff = stable & (c < 1 * n)
    R_buffer = stable & ~R_aff_stat & ~R_unaff
    R_mover = pm3 == mover
    R_unstable = (hit0 | hit3) & ~stable & ~R_mover
    background = ~hit0 & ~hit3

    # T3 first hits on a 5x5 sub-pixel lattice (edges included) -> learned-state stencil per pixel
    sensor = scenes["T3"].sensors()[0]
    j, i = np.meshgrid(np.arange(res), np.arange(res), indexing="ij")
    sub = []
    for oy in (0.0, 0.25, 0.5, 0.75, 1.0):
        for ox in (0.0, 0.25, 0.5, 0.75, 1.0):
            fx = np.clip((i.ravel() + ox) / res, 0, 1 - 1e-7)
            fy = np.clip((j.ravel() + oy) / res, 0, 1 - 1e-7)
            ray, _ = sensor.sample_ray(0.0, 0.5, mi.Point2f(fx.astype(np.float32), fy.astype(np.float32)), mi.Point2f(0.5, 0.5))
            si = scenes["T3"].ray_intersect(ray)
            pos = np.stack([np.array(si.p.x), np.array(si.p.y), np.array(si.p.z)], -1)
            pos[~np.array(si.is_valid())] = np.nan
            sub.append(pos.astype(np.float32))
    sub = np.stack(sub, 1)  # (res*res, 25, 3)

    counts = {k: int(v.sum()) for k, v in dict(stable_stationary=stable, R_aff_stat=R_aff_stat, R_unaff=R_unaff,
                                                 R_buffer=R_buffer, R_mover=R_mover, R_unstable=R_unstable,
                                                 background=background).items()}
    np.savez_compressed(out / "regions.npz", stable_stationary=stable, R_aff_stat=R_aff_stat, R_unaff=R_unaff,
                        R_buffer=R_buffer, R_mover=R_mover, R_aff=R_aff_stat | R_mover, R_unstable=R_unstable,
                        background=background, change=c, noise=n, part_T0=pm0.astype(str), part_T3=pm3.astype(str),
                        x1_T3_sub=sub)
    S.write_json(out / "regions.json", {"schema": "sro_regions/v1", "git": git, "res": res, "pixels": counts,
                                        "rule": proto["evaluation"]["regions"], "environment": L.environment_record()})
    print(counts, flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(rc)
