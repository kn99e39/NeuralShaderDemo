"""Synthetic geometry-change fixture: implementation semantics only, never architecture evidence.

    windows/run.ps1 ../selective_refit_oracle/tests/test_sro_synthetic.py --out <dir>

One asset (shapegroup instance) = floor rectangle + stationary receiver sphere + moving sphere,
rough conductors.  T0 -> T3 moves the moving sphere toward the receiver.  A small 8DNA (upstream
architecture, default config) is trained from scratch at T0 for a few hundred steps, then:
  oracle (CRN traces, same code as sro_oracle) -> M95 cells,
  E-style restricted refit at T3 for 100 steps (sro_train.upstream_loss, mask hook),
and the contracts are checked: affected rays concentrate near the mover; frozen tensors and
unselected cells are bit-identical; predictions of held-out samples whose stencil avoids every
selected cell are bit-identical; touched predictions change; the selected-sample loss drops.
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sro_common as S  # noqa: E402


def scene(res, mover_x):
    mi, _ = S.init()
    T = mi.ScalarTransform4f
    rc = lambda a: {"type": "twosided", "bsdf": {"type": "roughconductor", "material": "Ni_palik", "alpha": a}}
    return {
        "type": "scene",
        "camera": {"type": "perspective", "fov": 35, "to_world": T.look_at(origin=[0, 1.5, 2.0], target=[0, 0.1, 0], up=[0, 1, 0]),
                   "film": {"type": "hdrfilm", "width": res, "height": res, "filter": {"type": "box"}}},
        "background": {"type": "constant", "radiance": {"type": "rgb", "value": 1.0}},
        "group0": {"type": "shapegroup",
                   "floor": {"type": "rectangle", "to_world": T.rotate([1, 0, 0], -90) @ T.scale([0.6, 0.6, 1.0]), "bsdf": rc(0.1)},
                   "receiver": {"type": "sphere", "center": [0.0, 0.2, 0.3], "radius": 0.18, "bsdf": rc(0.05)},
                   "mover": {"type": "sphere", "center": [mover_x, 0.15, -0.05], "radius": 0.14, "bsdf": rc(0.1)}},
        "instance0": {"type": "instance", "shapegroup": {"type": "ref", "id": "group0"}},
        "integrator": {"type": "prb", "max_depth": -1, "rr_depth": 5},
    }


def register(name, mover_x):
    mod = types.ModuleType(f"scenes.{name}")
    mod.get_scene = lambda res, *_, **__: scene(res, mover_x)
    sys.modules[f"scenes.{name}"] = mod
    import scenes

    setattr(scenes, name, mod)
    return name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    mi, dr = S.init()
    import numpy as np
    import torch
    from sro_oracle import Q, affected_flags
    from sro_paths import PathGen, draw_chunk, n_chunks, strata
    from sro_train import upstream_loss
    from utils.dataset import PathSamplingDataset

    X0, X3 = -0.35, 0.05
    N_ds, H_ds, SPP, B = 32, 32, 32, 4096
    hp = torch.load(S.released_checkpoint(), map_location="cpu", weights_only=False)["hyper_parameters"]
    Ng, C = 64, 8
    res = {}

    # 1. T0 pretraining from scratch (all parameters), upstream loss
    ds0 = PathSamplingDataset(scene=register("syn_T0", X0), N=N_ds, H=H_ds, batch_size=B, device="cuda", SPP=SPP, bounce=2)
    model = S.scratch_model(hp, 0)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    torch.manual_seed(0)
    for ep in range(2):
        ds0.reload(0); ds0.resample()
        for b in range(len(ds0)):
            bt = ds0[b]; m = bt["mask"]
            if not m.any():
                continue
            loss, _, _ = upstream_loss(model, *(bt[k][m] for k in ("xi", "wi", "xo", "wo", "throughput")), int(m.sum()))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    res["pretrain_steps"] = 2 * len(ds0)

    # 2. oracle with common random numbers
    g0, g3 = PathGen(scene(H_ds, X0), H_ds), PathGen(scene(H_ds, X3), H_ds)
    acc = {k: torch.zeros(3 * Ng * Ng, dtype=torch.long, device="cuda") for k in ("A3", "A0", "W3")}
    torch.manual_seed(123)
    sx, sw = strata(N_ds, H_ds)
    aff_xi, unaff_xi, n_aff, n_valid = [], [], 0, 0
    for idx in range(n_chunks(N_ds, SPP)):
        a, b, sd = draw_chunk(idx, N_ds, H_ds, SPP, sx, sw)
        f0, f3 = g0.trace(a, b, sd, SPP), g3.trace(a, b, sd, SPP)
        aff, _ = affected_flags(f0, f3)
        for key, f, sel in (("A3", f3, f3["valid"] & aff), ("A0", f0, f0["valid"] & aff), ("W3", f3, f3["valid"])):
            if sel.any():
                i_, w_ = S.stencil(f["xi"][sel], Ng)
                acc[key].index_add_(0, i_.reshape(-1), torch.round(w_ * Q).long().reshape(-1))
        aff_xi.append(f3["xi"][f3["valid"] & aff].cpu()); unaff_xi.append(f3["xi"][f3["valid"] & ~aff].cpu())
        n_aff += int((f3["valid"] & aff).sum()); n_valid += int(f3["valid"].sum())
    A = (acc["A3"] + acc["A0"]).cpu().numpy()
    order = np.lexsort((np.arange(len(A)), -A))
    k = int(np.searchsorted(np.cumsum(A[order]) / A.sum(), 0.95) + 1)
    m95 = np.zeros(len(A), bool); m95[order[:k]] = True
    ax, ux = torch.cat(aff_xi).numpy(), torch.cat(unaff_xi).numpy()
    c0, c3 = np.array([X0, 0.15, -0.05]), np.array([X3, 0.15, -0.05])
    dmov = lambda x: np.minimum(np.linalg.norm(x - c0, axis=1), np.linalg.norm(x - c3, axis=1)) - 0.14
    res["oracle"] = {"affected_of_valid_T3": n_aff / max(n_valid, 1), "M95_cells": k, "M95_fraction": k / len(A),
                     "affected_median_distance_to_mover": float(np.median(dmov(ax))),
                     "unaffected_median_distance_to_mover": float(np.median(dmov(ux)))}

    # 3. E-style restricted refit at T3
    cm = torch.from_numpy(m95).cuda()
    pm = S.cell_mask_to_param_mask(cm, C, Ng)
    init = {kk: v.clone() for kk, v in model.state_dict().items()}
    ds3 = PathSamplingDataset(scene=register("syn_T3", X3), N=N_ds, H=H_ds, batch_size=B, device="cuda", SPP=SPP, bounce=2)
    torch.manual_seed(7)
    ds3.reload(0); ds3.resample()  # held-out samples: their own reload, shuffled
    vbs = [ds3[i] for i in range(4)]
    vx = {kk: torch.cat([vb[kk][vb["mask"]] for vb in vbs]) for kk in ("xi", "wi", "xo", "wo", "throughput")}
    vt = S.touches(vx["xi"], cm, Ng)
    model.eval()
    with torch.no_grad():
        lp0, al0 = model(vx["xi"], vx["wi"], vx["xo"], vx["wo"])
        l0_touch = float((-(vx["throughput"][vt] * lp0[vt]).mean(-1) + ((vx["throughput"][vt] - al0[vt]) ** 2).mean(-1)).mean())
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)
    trip = model.encode_xi.feature
    trip.requires_grad_(True)
    trip.register_hook(lambda g: g * pm)
    opt = torch.optim.Adam([trip], lr=5e-4)
    ds3.reload(0); ds3.resample()  # training samples: a fresh reload
    touched, valid = 0, 0
    for b in range(1, 101):
        bt = ds3[b % len(ds3)]; m = bt["mask"]
        xi, wi, xo, wo, thr = (bt[kk][m] for kk in ("xi", "wi", "xo", "wo", "throughput"))
        n = len(xi); t = S.touches(xi, cm, Ng); touched += int(t.sum()); valid += n
        opt.zero_grad(set_to_none=False)
        if t.any():  # as sro_train: an empty restricted batch is a zero-gradient step
            loss, _, _ = upstream_loss(model, xi[t], wi[t], xo[t], wo[t], thr[t], n)
            loss.backward()
        torch.nn.utils.clip_grad_norm_([trip], 5.0); opt.step()
    model.eval()
    with torch.no_grad():
        lp1, al1 = model(vx["xi"], vx["wi"], vx["xo"], vx["wo"])
        l1_touch = float((-(vx["throughput"][vt] * lp1[vt]).mean(-1) + ((vx["throughput"][vt] - al1[vt]) ** 2).mean(-1)).mean())
    sd = model.state_dict()
    res["refit"] = {
        "frozen_shared_identical": all(torch.equal(sd[kk], init[kk]) for kk in sd if kk != "encode_xi.feature"),
        "unselected_cells_identical": bool(torch.equal(sd["encode_xi.feature"][~pm], init["encode_xi.feature"][~pm])),
        "selected_cells_changed": int((sd["encode_xi.feature"] != init["encode_xi.feature"]).any(1).flatten()[cm].sum()),
        "untouched_predictions_bit_identical": bool(torch.equal(lp0[~vt], lp1[~vt]) and torch.equal(al0[~vt], al1[~vt])),
        "untouched_samples": int((~vt).sum()),
        "touched_predictions_changed_share": float(((lp0[vt] != lp1[vt]).any(-1) | (al0[vt] != al1[vt]).any(-1)).float().mean()),
        "touched_loss_before_after": [l0_touch, l1_touch],
        "restricted_fraction_of_valid": touched / max(valid, 1),
    }
    r = res["refit"]; o = res["oracle"]
    res["pass"] = bool(r["frozen_shared_identical"] and r["unselected_cells_identical"] and r["selected_cells_changed"] > 0
                       and r["untouched_predictions_bit_identical"] and r["touched_predictions_changed_share"] > 0.5
                       and l1_touch < l0_touch and o["M95_fraction"] < 1.0
                       and o["affected_median_distance_to_mover"] < o["unaffected_median_distance_to_mover"])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    S.write_json(out / "synthetic.json", res)
    print("synthetic", "PASS" if res["pass"] else "FAIL", res, flush=True)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(rc)
