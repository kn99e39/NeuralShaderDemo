"""Pretraining (T0) and the matched T3 arms of protocol sor_v1.json.

    windows/run.ps1 ../surface_owned_radiance/sor_arm.py --arm {T0,B,C,DG,E,U} --root <batch dir> [--seed 0]

T0  : all units, init zeros, fit on the T0 scene (the pretrained object models)
B   : all units, init zeros, fit on T3
C   : all units, init T0 tables, fit on T3
DG  : units of the geometry-dirty object(s), init T0, others frozen
E   : units of the oracle transport-dirty objects, init T0, others frozen
U   : oracle unit-level M95 (diagnostic), init T0, others frozen

Writes <root>/arms/<arm>_s<seed>/: events.jsonl (timing), snapshots.npz (camera indirect images per
snapshot step), heldout.json (angular fidelity per snapshot), table.pt (final table), arm.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import sor_common as S

ARMS = ("T0", "B", "C", "DG", "E", "E5", "U")


def object_units_mask(units, objects):
    import numpy as np

    keep = np.isin(units["part"], [S.PARTS.index(o) for o in objects])
    return keep


def dirty_objects(root: Path, proto: dict, thr=None) -> list[str]:
    rec = json.loads((root / "oracle" / "oracle.json").read_text())
    thr = proto["oracle"]["object_dirty_rule_threshold"] if thr is None else thr
    return [p for p, share in rec["object_change_share"].items() if share >= thr]


def main() -> int:
    import nrc_records as R

    created = R.process_create_time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--root", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None, help="smoke only")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    proto = S.protocol()
    fc = proto["fit_contract"]
    seed = args.seed if args.seed is not None else (fc["pretrain_T0"]["seed"] if args.arm == "T0" else fc["seeds"]["all"])
    out = root / "arms" / f"{args.arm}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    log = R.EventLog(out / "events.jsonl", f"arm_{args.arm}")
    log.emit("process_created", t_created=created)
    import ednalib as L

    git = R.git_state(L.ROOT)
    R.require_clean(git, args.smoke)
    mi, dr = S.init()
    import numpy as np
    import torch
    from sor_fit import PositionPool, fit, render_indirect
    from sor_targets import camera_queries, scene_for

    log.emit("imports_done")
    log.emit("environment", environment=L.environment_record(), git=git, argv=sys.argv[1:])
    state = "T0" if args.arm == "T0" else "T3"
    steps = args.steps or (fc["pretrain_T0"]["steps"] if args.arm == "T0" else fc["arms_budget_steps"])
    snaps = sorted({s for s in fc["snapshot_steps"] if s <= steps} | {steps}) if args.arm != "T0" else sorted({0, steps // 4, steps // 2, steps})

    log.begin("setup", key="units_scene")
    units = S.load_units(root / "units" / "units.npz")
    model = S.SurfaceRadiance(units, proto["representation"]["res"])
    scene, ids, tr = scene_for(state)
    torch.cuda.synchronize(); log.end("setup", key="units_scene")

    log.begin("setup", key="init_and_mask")
    table0 = None
    if args.arm not in ("T0", "B"):
        table0 = torch.load(root / "arms" / f"T0_s{fc['pretrain_T0']['seed']}" / "table.pt", map_location="cuda")
    U = model.n_units
    if args.arm in ("T0", "B", "C"):
        mask_np, objects = None, list(S.PARTS)
    elif args.arm == "DG":
        objects = proto["oracle"]["geometry_dirty"]
        mask_np = object_units_mask(units, objects)
    elif args.arm in ("E", "E5"):
        objects = dirty_objects(root, proto, None if args.arm == "E" else 0.05)
        mask_np = object_units_mask(units, objects)
    else:  # U
        objects = None
        mask_np = np.load(root / "oracle" / "oracle_units.npz")["M95"]
    mask = None if mask_np is None else torch.tensor(mask_np, device="cuda")
    torch.cuda.synchronize(); log.end("setup", key="init_and_mask")

    log.begin("setup", key="pool")
    pool = PositionPool(model, fc["pretrain_T0"]["pool"] if mask is None else fc["pretrain_T0"]["pool"] // 4,
                        seed=1000 + seed, unit_mask=mask)
    support = 1.0 if mask is None else pool.support_fraction
    gb = fc["pretrain_T0"]["batch"] if args.arm == "T0" else fc["global_batch"]
    batch = gb if mask is None else max(1, int(round(gb * support)))
    torch.cuda.empty_cache()  # pool building leaves large stencil temporaries cached; DrJit needs the memory
    torch.cuda.synchronize(); log.end("setup", key="pool", support_fraction=support, batch=batch)

    # evaluation inputs (excluded from adaptation latency)
    log.begin("evaluation", key="prepare")
    scene512, ids512, _ = scene_for(state, 512)
    cam = camera_queries(scene512, ids512, tr, 512, 4, seed=5)
    held = torch.load(root / "heldout" / f"heldout_{state}.pt", map_location="cuda")
    hi, hw = model.stencil(held["x_local"], held["part"], held["n"])
    hrows, hws = model.texel_stencil(hi, hw, held["w"])
    del hi, hw
    torch.cuda.empty_cache()
    torch.cuda.synchronize(); log.end("evaluation", key="prepare")

    images, held_rows = {}, []
    t_eval = [0.0]

    def on_snap(step, table, tim):
        t = time.perf_counter()
        log.emit("snapshot_ready", step=step, **tim)
        with torch.no_grad():
            ph = (table[hrows] * hws[..., None]).sum(1)
            err = (ph - held["y"]).abs().mean(1)
            ref = held["y"].abs().mean(1)
            row = {"step": step, "rel_mae_all": float(err.mean() / ref.mean())}
            for i, p in enumerate(S.PARTS):
                m = held["part"] == i
                row[f"rel_mae_{p}"] = float(err[m].mean() / ref[m].mean())
            held_rows.append(row)
        images[step] = render_indirect(model, table, cam).astype(np.float32)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        t_eval[0] += time.perf_counter() - t
        log.emit("snapshot_evaluated", step=step, eval_s=time.perf_counter() - t)
        print(f"[{args.arm} s{seed}] step {step} heldout {row['rel_mae_all']:.4f} trace {tim['trace_s']:.1f}s opt {tim['opt_s']:.1f}s", flush=True)

    log.begin("fit")
    table, info = fit(model, scene, ids, tr, pool, steps, batch, 0.02, seed=seed * 7 + 11, unit_mask=mask, table0=table0,
                      snapshot_steps=snaps, on_snapshot=on_snap)
    torch.cuda.synchronize(); log.end("fit", **info)
    log.begin("save")
    torch.save(table, out / "table.pt")
    torch.cuda.synchronize(); log.end("save")
    np.savez_compressed(out / "snapshots.npz", **{f"step_{k:05d}": v for k, v in images.items()})
    # parameter identity of frozen units vs the init table
    ident = None
    if table0 is not None and mask is not None:
        R2 = model.res * model.res
        fr = (~mask).repeat_interleave(R2)
        ident = {"frozen_units": int((~mask).sum()), "frozen_entries_bit_identical": bool(torch.equal(table[fr], table0[fr])),
                 "per_object_frozen_identical": {p: bool(torch.equal(table[fr & (model.part.repeat_interleave(R2) == i)],
                                                                     table0[fr & (model.part.repeat_interleave(R2) == i)]))
                                                 for i, p in enumerate(S.PARTS)}}
    rec = {"schema": "sor_arm/v1", "arm": args.arm, "seed": seed, "state": state, "steps": steps, "snapshots": snaps,
           "objects_trainable": objects, "trainable_units": int(U if mask is None else mask.sum()), "units_total": U,
           "support_fraction": support, "batch": batch, "fit": info, "eval_seconds": t_eval[0], "heldout": held_rows,
           "identity": ident, "pool": {"seconds": pool.seconds, "drawn": pool.drawn},
           "peak_mem_gib": torch.cuda.max_memory_allocated() / 2 ** 30, "smoke": args.smoke, "git": git}
    S.write_json(out / "arm.json", rec)
    log.emit("done"); log.close()
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(rc)
