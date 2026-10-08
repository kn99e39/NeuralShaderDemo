"""Matched T3 refits of the released 8DNA teaset asset (protocol `controls`, `training_contract`).

    windows/run.ps1 ../selective_refit_oracle/sro_train.py --arm {B,C,D,E,E_mask,S} --out <dir>
        [--seed 0] [--steps 32768] [--mask <oracle_cells.npz>] [--smoke]

Arms (one substrate, one T3 scene, one data stream per seed):
  B       scratch init (upstream EightDNA init under torch seed), all parameters trainable
  C       released T0 checkpoint, all parameters trainable
  D       released T0 checkpoint, only encode_xi.feature (all cells) trainable
  E       released T0 checkpoint, only the oracle M95 cells trainable; the batch is D's batch
          restricted to valid samples touching a selected cell, loss normalised by the batch's
          full valid count (= E_mask's update exactly, network evaluated on touching samples only)
  E_mask  E's mask on D's full batch (diagnostic: equivalence and per-step cost)
  S       equal-size spatial mask, E's implementation

The loop is upstream train.py's (ModelTrainer.training_step, Adam lr 5e-4, Lightning norm
clipping 5, NaN-loss steps skipped, a reload + resample every len(dataset) = 8192 steps with the
reload seed drawn by torch.randint as on_train_epoch_start does), without Lightning.  Training
draws no other random numbers, so every arm with the same seed sees the same batches.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import time
from pathlib import Path

import sro_common as S

ARMS = ("B", "C", "D", "E", "E_mask", "S")
VAL_BATCHES = 64  # protocol training_contract.validation: the first 64 raw batches
MASK_KEY = {"E": "M95", "E_mask": "M95", "S": "S_spatial"}


def upstream_loss(model, xi, wi, xo, wo, thr, n_norm: int):
    """ModelTrainer.training_step's loss; with n_norm = len(xi) it is upstream's mean exactly,
    with n_norm > len(xi) the same per-sample terms are summed and divided by n_norm."""
    import torch.nn.functional as NF

    logpdf, albedo = model(xi, wi, xo, wo)
    if n_norm == len(xi):
        loss_albedo = NF.mse_loss(thr, albedo)
        loss_pdf = -(thr * logpdf).mean()
    else:
        loss_albedo = ((thr - albedo) ** 2).sum() / (n_norm * thr.shape[-1])
        loss_pdf = -(thr * logpdf).sum() / (n_norm * logpdf.shape[-1])
    return loss_pdf + loss_albedo, loss_pdf, loss_albedo


def build_validation(gen_scene: dict, seed: int, keep_every: int, N: int, H: int, SPP: int):
    """Held-out T3 samples: one reload drawn with `seed`, every keep_every-th ray of each chunk."""
    import torch
    from sro_paths import PathGen, generate

    gen = PathGen(gen_scene, H)
    torch.manual_seed(seed)
    parts = {k: [] for k in ("xi", "wi", "xo", "wo", "throughput", "valid")}
    for idx, f in generate(gen, N, H, SPP):
        off = idx % keep_every
        for k in parts:
            parts[k].append(f[k][off::keep_every])
    return {k: torch.cat(v) for k, v in parts.items()}


def validation_loss(model, val: dict, batch: int, n_batches: int, cell_mask=None, Ngrid=64):
    import torch

    out = {"loss": 0.0, "loss_touching": 0.0, "loss_other": 0.0, "n": 0, "n_touching": 0}
    with torch.no_grad():
        for b in range(n_batches):
            sl = slice(b * batch, (b + 1) * batch)
            m = val["valid"][sl]
            xi, wi, xo, wo, thr = (val[k][sl][m] for k in ("xi", "wi", "xo", "wo", "throughput"))
            if len(xi) == 0:
                continue
            logpdf, albedo = model(xi, wi, xo, wo)
            per = -(thr * logpdf).mean(-1) + ((thr - albedo) ** 2).mean(-1)
            out["loss"] += float(per.sum()); out["n"] += len(per)
            if cell_mask is not None:
                t = S.touches(xi, cell_mask, Ngrid)
                out["loss_touching"] += float(per[t].sum()); out["n_touching"] += int(t.sum())
                out["loss_other"] += float(per[~t].sum())
    n, nt = max(out["n"], 1), max(out["n_touching"], 1)
    return {"loss": out["loss"] / n, "n": out["n"],
            "loss_touching_mask": out["loss_touching"] / nt if cell_mask is not None else None,
            "loss_not_touching_mask": out["loss_other"] / max(out["n"] - out["n_touching"], 1) if cell_mask is not None else None,
            "n_touching_mask": out["n_touching"]}


def main() -> int:
    created = None
    import nrc_records as R

    created = R.process_create_time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--mask", default=None, help="oracle_cells.npz (arms E, E_mask, S)")
    ap.add_argument("--smoke", action="store_true", help="tiny budget, dirty tree allowed; never evidence")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    proto = S.protocol()
    tc = proto["training_contract"]
    seed = args.seed if args.seed is not None else tc["seeds"]["all_runs"]
    budget = args.steps if args.steps is not None else tc["budget_steps"]
    snaps = sorted({s for s in tc["snapshot_steps"] if s <= budget} | {budget})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = R.EventLog(out / "events.jsonl", f"train_{args.arm}")
    log.emit("process_created", t_created=created)

    import ednalib as L

    git = R.git_state(L.ROOT)
    R.require_clean(git, args.allow_dirty or args.smoke)
    poller = R.GpuMemoryPoller()
    poller.start()
    mi, dr = S.init()
    import numpy as np
    import torch
    import teaset_parts as T
    from utils.dataset import PathSamplingDataset

    log.emit("imports_done")
    log.emit("environment", environment=L.environment_record(), git=git, argv=sys.argv[1:])
    sync = torch.cuda.synchronize
    st = S.states()
    ds_cfg = {"N": 128, "H": 128, "SPP": 256, "batch_size": 32768}

    # --- held-out validation samples (timed as validation; excluded from adaptation latency)
    log.begin("validation", key="build")

    val = build_validation(T.scene_dict(ds_cfg["H"], st["T3"]), 40040, 128, ds_cfg["N"], ds_cfg["H"], ds_cfg["SPP"])
    sync(); log.end("validation", key="build", n=int(val["valid"].numel()))

    # --- T3 supervision: upstream PathSamplingDataset on the registered T3 scene
    name = S.register_state_scene("T3", st["T3"])
    log.begin("dataset_init")
    ds = PathSamplingDataset(scene=name, bounce=2, device="cuda", **ds_cfg)
    sync(); log.end("dataset_init")

    log.begin("model_load")
    ck_rel = S.released_checkpoint()
    if args.arm == "B":
        hp = torch.load(ck_rel, map_location="cpu", weights_only=False)["hyper_parameters"]
        model = S.scratch_model(hp, seed)
    else:
        model, hp = S.load_model(ck_rel)
    model.train()
    sync(); log.end("model_load")
    Ngrid, C = hp["model"]["encode_xi"]["N"], hp["model"]["encode_xi"]["C"]
    init_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}

    log.begin("mask_load")
    cell_mask = None
    if args.arm in MASK_KEY:
        if not args.mask:
            raise SystemExit("--mask required")
        cell_mask = torch.from_numpy(np.load(args.mask)[MASK_KEY[args.arm]]).cuda()
        pmask = S.cell_mask_to_param_mask(cell_mask, C, Ngrid)
    trip = model.encode_xi.feature
    if args.arm in ("B", "C"):
        trainable = [p for p in model.parameters()]
    else:
        for p in model.parameters():
            p.requires_grad_(False)
        trip.requires_grad_(True)
        trainable = [trip]
        if cell_mask is not None:
            trip.register_hook(lambda g: g * pmask)
    sync(); log.end("mask_load", cells=None if cell_mask is None else int(cell_mask.sum()))

    log.begin("optimizer_init")
    opt = torch.optim.Adam(trainable, lr=5e-4)
    sync(); log.end("optimizer_init")
    restricted = args.arm in ("E", "S")
    n_trainable = sum(p.numel() for p in trainable) if cell_mask is None else int(cell_mask.sum()) * C

    torch.manual_seed(seed)  # the data stream (reload seeds, strata jitter, shuffles) starts here
    steps_per_reload = len(ds)
    stream = []
    snap_dir = out / "snapshots"
    snap_dir.mkdir(exist_ok=True)

    def snapshot(step):
        sync()
        log.begin("snapshot_write", key=step)
        path = snap_dir / f"step_{step:06d}.ckpt"
        torch.save(S.released_layout(model, hp), path)
        log.end("snapshot_write", key=step, file=path.name)
        model.eval()
        log.begin("validation", key=step)
        v = validation_loss(model, val, ds_cfg["batch_size"], VAL_BATCHES,
                            torch.from_numpy(np.load(args.mask)["M95"]).cuda() if args.mask else None, Ngrid)
        sync(); log.end("validation", key=step, **v)
        model.train()
        log.emit("torch_memory", step=step, max_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
                 max_reserved_mib=torch.cuda.max_memory_reserved() / 2 ** 20)
        print(f"[{args.arm} s{seed}] step {step} val {v['loss']:.5f}", flush=True)

    step, skipped, empty_steps, touched_total, valid_total = 0, 0, 0, 0, 0
    opt_open = None
    if 0 in snaps:
        snapshot(0)
    while step < budget:
        b = step % steps_per_reload
        if b == 0:
            reload_seed = torch.randint(0, torch.iinfo(torch.int32).max, (1,)).item()
            sync(); log.begin("path_generation", key=step // steps_per_reload)
            ds.reload(reload_seed)
            sync(); log.end("path_generation", key=step // steps_per_reload)
            log.begin("resample", key=step // steps_per_reload)
            ds.resample()
            sync(); log.end("resample", key=step // steps_per_reload)
            digest = hashlib.sha256(ds.inds[:4096].cpu().numpy().tobytes() + ds.xi[:4096].numpy().tobytes()).hexdigest()[:16]
            stream.append({"reload": step // steps_per_reload, "reload_seed": reload_seed, "digest": digest})
            log.emit("stream", **stream[-1])
            opt_open = step
            log.begin("optimization", key=opt_open)
        batch = ds[b]
        m = batch["mask"]
        if not m.any():
            step += 1
            skipped += 1
        else:
            xi, wi, xo, wo, thr = (batch[k][m] for k in ("xi", "wi", "xo", "wo", "throughput"))
            n_valid = len(xi)
            valid_total += n_valid
            if restricted:
                t = S.touches(xi, cell_mask, Ngrid)
                touched_total += int(t.sum())
                xi, wi, xo, wo, thr = xi[t], wi[t], xo[t], wo[t], thr[t]
            opt.zero_grad(set_to_none=False)
            if len(xi) == 0:
                empty_steps += 1  # zero gradient for every selected cell, exactly as E_mask
                loss = torch.zeros((), device="cuda")
            else:
                loss, _, _ = upstream_loss(model, xi, wi, xo, wo, thr, n_valid)
            if loss.isnan() or loss.isinf():
                skipped += 1  # upstream returns None: Lightning skips the optimiser step
            else:
                if len(xi):
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 5.0)
                opt.step()
            step += 1
        if step in snaps or step % steps_per_reload == 0 or step == budget:
            sync(); log.end("optimization", key=opt_open, end_step=step)
            if step in snaps:
                snapshot(step)
            if step < budget and step % steps_per_reload != 0:
                opt_open = step
                log.begin("optimization", key=opt_open)

    # --- identity checks: frozen parameters bit-identical
    sd = model.state_dict()
    frozen_ok, changed = True, {}
    for k, v in sd.items():
        same = torch.equal(v, init_sd[k])
        if k == "encode_xi.feature" and cell_mask is not None:
            fm = ~S.cell_mask_to_param_mask(cell_mask, C, Ngrid)
            same_frozen = torch.equal(v[fm], init_sd[k][fm])
            changed[k] = {"unselected_bit_identical": bool(same_frozen),
                          "selected_cells_changed": int((v != init_sd[k]).any(1).flatten()[cell_mask].sum())}
            frozen_ok &= same_frozen
        elif args.arm in ("D", "E", "E_mask", "S") and k != "encode_xi.feature":
            frozen_ok &= same
            if not same:
                changed[k] = "CHANGED"
        if k == "encode_xi.feature":
            changed.setdefault(k, {})
            if isinstance(changed[k], dict):
                changed[k]["cells_changed_total"] = int((v != init_sd[k]).any(1).flatten().sum())
    gpu = poller.halt()
    rec = {"schema": "sro_train/v1", "arm": args.arm, "seed": seed, "budget_steps": budget, "smoke": args.smoke,
           "snapshot_steps": snaps, "steps_per_reload": steps_per_reload, "stream": stream,
           "trainable_parameters": n_trainable, "total_parameters": sum(v.numel() for v in sd.values()),
           "mask": None if cell_mask is None else {"file": args.mask, "key": MASK_KEY[args.arm], "cells": int(cell_mask.sum())},
           "skipped_steps": skipped, "empty_restricted_steps": empty_steps,
           "restricted_fraction_of_valid": (touched_total / valid_total) if restricted and valid_total else None,
           "frozen_identity_ok": bool(frozen_ok), "parameter_changes": changed,
           "init_digest": S.param_digest(init_sd), "final_digest": S.param_digest(sd),
           "torch_memory_mib": {"max_allocated": torch.cuda.max_memory_allocated() / 2 ** 20,
                                "max_reserved": torch.cuda.max_memory_reserved() / 2 ** 20},
           "gpu_poll": gpu, "host_peak_mb": R.host_peak_memory_mb()}
    S.write_json(out / "train.json", rec)
    log.emit("done")
    log.close()
    print(f"[{args.arm} s{seed}] done frozen_ok={frozen_ok} restricted={rec['restricted_fraction_of_valid']}", flush=True)
    if args.arm in ("D", "E", "E_mask", "S") and not frozen_ok:
        return 4
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(rc)
