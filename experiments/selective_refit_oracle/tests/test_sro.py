"""Focused tests for the oracle selective-refit batch (protocol section 12 of the batch).

    windows/run.ps1 ../selective_refit_oracle/tests/test_sro.py --out <dir> [--only name ...]

Each test returns a dict with 'pass'; the runner writes <out>/tests.json and exits 1 on any failure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sro_common as S  # noqa: E402


def _model():
    import torch

    model, hp = S.load_model(S.released_checkpoint())
    return model, hp


def t_state_units():
    """Stencil + weights reproduce upstream Triplane.forward exactly; cell layout = parameter layout."""
    import torch

    model, hp = _model()
    N, C = 64, 8
    g = torch.Generator(device="cuda").manual_seed(1)
    xi = torch.rand(20000, 3, device="cuda", generator=g) * 2 - 1
    xi[:50] = torch.tensor([1.0, -1.0, 0.0], device="cuda")   # clamped corners
    up = model.encode_xi(xi)
    idx, w = S.stencil(xi, N)
    feat = model.encode_xi.feature.detach().permute(0, 2, 3, 1).reshape(3 * N * N, C)
    mine = torch.cat([(feat[idx[:, p]] * w[:, p, :, None]).sum(1) for p in range(3)], -1)
    err = float((mine - up).abs().max())
    one = torch.zeros(3 * N * N, dtype=torch.bool, device="cuda"); one[N * N + 5 * N + 7] = True
    pm = S.cell_mask_to_param_mask(one, C, N)
    layout_ok = bool(pm[1, :, 5, 7].all() and pm.sum() == C)
    return {"pass": err < 1e-6 and layout_ok, "max_abs_diff": err, "layout_ok": layout_ok}


def _fake_batch(n, seed):
    import torch
    import torch.nn.functional as NF

    g = torch.Generator(device="cuda").manual_seed(seed)
    xi = (torch.rand(n, 3, device="cuda", generator=g) * 1.4 - 0.7)
    wi = NF.normalize(torch.randn(n, 3, device="cuda", generator=g), dim=-1)
    xo = NF.normalize(torch.randn(n, 3, device="cuda", generator=g), dim=-1)
    wo = NF.normalize(torch.randn(n, 3, device="cuda", generator=g), dim=-1)
    thr = torch.rand(n, 3, device="cuda", generator=g) * 0.5
    return xi, wi, xo, wo, thr


def t_masking_and_equivalence():
    """Masked full batch (E_mask) vs restricted batch (E): identical masked gradients on the same
    batch (the update E computes); frozen tensors bit-identical after 20 Adam steps in both modes.
    The 20-step trajectory difference is reported (Adam normalises per element, so float
    reduction-order noise on near-zero gradients can grow up to ~lr per step)."""
    import torch
    from sro_train import upstream_loss

    N, C = 64, 8
    cell_mask = torch.zeros(3 * N * N, dtype=torch.bool, device="cuda")
    g = torch.Generator(device="cuda").manual_seed(3)
    cell_mask[torch.randperm(3 * N * N, device="cuda", generator=g)[:600]] = True
    pm = S.cell_mask_to_param_mask(cell_mask, C, N)

    def setup():
        model, hp = _model()
        for p in model.parameters():
            p.requires_grad_(False)
        trip = model.encode_xi.feature
        trip.requires_grad_(True)
        trip.register_hook(lambda gr: gr * pm)
        return model, trip

    def batch(mode, it):
        xi, wi, xo, wo, thr = _fake_batch(8192, 100 + it)
        n_valid = len(xi)
        if mode == "restricted":
            t = S.touches(xi, cell_mask, N)
            xi, wi, xo, wo, thr = xi[t], wi[t], xo[t], wo[t], thr[t]
        return (xi, wi, xo, wo, thr), n_valid

    grads = {}
    for mode in ("mask", "zeroed", "restricted"):
        model, trip = setup()
        if mode == "zeroed":  # full batch, non-touching samples' loss terms multiplied by 0
            (xi, wi, xo, wo, thr), n = batch("mask", 0)
            t = S.touches(xi, cell_mask, N).float()[:, None]
            logpdf, albedo = model(xi, wi, xo, wo)
            loss = (-(thr * logpdf) * t).sum() / (n * 3) + (((thr - albedo) ** 2) * t).sum() / (n * 3)
        else:
            b_, n = batch(mode, 0)
            loss, _, _ = upstream_loss(model, *b_, n)
        loss.backward()
        grads[mode] = trip.grad.clone()
    rel = lambda x, y: float((grads[x] - grads[y]).norm() / grads[x].norm())
    rel_zeroed = rel("mask", "zeroed")
    rel_restricted = rel("mask", "restricted")
    unsel_grad_zero = all(bool((g_[~pm] == 0).all()) for g_ in grads.values())
    finals = {}
    for mode in ("mask", "restricted"):
        model, trip = setup()
        init = {k: v.clone() for k, v in model.state_dict().items()}
        opt = torch.optim.Adam([trip], lr=5e-4)
        for it in range(20):
            b, n = batch(mode, it)
            opt.zero_grad(set_to_none=False)
            loss, _, _ = upstream_loss(model, *b, n)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([trip], 5.0)
            opt.step()
        sd = model.state_dict()
        frozen_ok = all(torch.equal(sd[k], init[k]) for k in sd if k != "encode_xi.feature")
        unsel_ok = torch.equal(sd["encode_xi.feature"][~pm], init["encode_xi.feature"][~pm])
        upd = (sd["encode_xi.feature"] - init["encode_xi.feature"])
        finals[mode] = (sd["encode_xi.feature"].clone(), frozen_ok, unsel_ok, upd)
    tdiff = float((finals["mask"][0] - finals["restricted"][0]).abs().max())
    upd_norm = float(finals["mask"][3].norm())
    rel_traj = float((finals["mask"][0] - finals["restricted"][0]).norm()) / max(upd_norm, 1e-30)
    # zeroed vs mask: same kernels and shapes, so only the dropped samples' contribution can differ;
    # restricted vs mask: smaller GEMM shapes (different float reduction order), bounded loosely
    ok = rel_zeroed <= 1e-6 and rel_restricted <= 1e-3 and unsel_grad_zero and all(f[1] and f[2] for f in finals.values())
    return {"pass": bool(ok), "grad_rel_l2_mask_vs_zeroed_nontouching": rel_zeroed,
            "grad_rel_l2_mask_vs_restricted": rel_restricted, "unselected_grad_zero": unsel_grad_zero,
            "frozen_shared_identical": [f[1] for f in finals.values()],
            "unselected_cells_identical": [f[2] for f in finals.values()],
            "trajectory_20_steps_max_abs_diff": tdiff, "trajectory_20_steps_rel_l2_to_update": rel_traj}


def t_upstream_loss_equivalence():
    """upstream_loss with n_norm = len equals ModelTrainer.training_step's loss on the same batch."""
    import torch
    import torch.nn.functional as NF
    from sro_train import upstream_loss

    model, hp = _model()
    xi, wi, xo, wo, thr = _fake_batch(4096, 7)
    with torch.no_grad():
        mine, _, _ = upstream_loss(model, xi, wi, xo, wo, thr, len(xi))
        logpdf, albedo = model(xi, wi, xo, wo)  # train.py ModelTrainer.training_step, verbatim
        loss_albedo = NF.mse_loss(thr, albedo)
        loss_pdf = -(thr * logpdf).mean()
        ref = loss_pdf + loss_albedo
        # normalised-sum form on a subset == mean form scaled by subset share
        sub = slice(0, 1000)
        part, _, _ = upstream_loss(model, xi[sub], wi[sub], xo[sub], wo[sub], thr[sub], len(xi))
        lp, al = model(xi[sub], wi[sub], xo[sub], wo[sub])
        expect = (-(thr[sub] * lp).sum() / (len(xi) * 3)) + ((thr[sub] - al) ** 2).sum() / (len(xi) * 3)
    src = (S.ROOT / "external" / "8dna26" / "train.py").read_text()
    verbatim = "loss_albedo = NF.mse_loss(throughput,albedo)" in src and "loss_pdf = -(throughput*logpdf).mean()" in src
    ok = float(abs(mine - ref)) == 0.0 and float(abs(part - expect)) < 1e-6 and verbatim
    return {"pass": bool(ok), "diff": float(abs(mine - ref)), "subset_diff": float(abs(part - expect)), "upstream_source_matches": verbatim}


def t_dataset_equivalence():
    """sro_paths.generate reproduces upstream PathSamplingDataset.reload bit for bit (small config)."""
    import torch
    import teaset_parts as T
    from sro_paths import PathGen, generate
    from utils.dataset import PathSamplingDataset

    N, H, SPP = 16, 16, 16
    name = S.register_state_scene("T3", S.states()["T3"])
    ds = PathSamplingDataset(scene=name, N=N, H=H, batch_size=256, device="cuda", SPP=SPP, bounce=2)
    torch.manual_seed(11)
    ds.reload(0)
    gen = PathGen(T.scene_dict(H, S.states()["T3"]), H)
    torch.manual_seed(11)
    parts = {k: [] for k in ("xi", "wi", "xo", "wo", "throughput", "valid")}
    for _, f in generate(gen, N, H, SPP):
        for k in parts:
            parts[k].append(f[k].cpu())
    mine = {k: torch.cat(v) for k, v in parts.items()}
    diffs = {k: float((mine[k].float() - getattr(ds, k if k != "valid" else "mask").float()).abs().max())
             for k in ("xi", "wi", "xo", "wo", "throughput", "valid")}
    return {"pass": all(v == 0.0 for v in diffs.values()) and float(ds.mask.float().mean()) > 0.05,
            "max_abs_diff": diffs, "valid_fraction": float(ds.mask.float().mean())}


def t_crn_determinism_and_correspondence():
    """Same scene traced twice with the same draws -> zero affected; T0 vs T3: unaffected rays have
    identical supervision and their first hits lie on stationary parts."""
    import torch
    import teaset_parts as T
    from sro_oracle import affected_flags
    from sro_paths import PathGen, draw_chunk, strata

    H, N, SPP = 128, 128, 256
    st = S.states()
    g0, g0b, g3 = (PathGen(T.scene_dict(H, st[s]), H) for s in ("T0", "T0", "T3"))
    torch.manual_seed(5)
    sx, sw = strata(N, H)
    for idx in range(41):  # chunk 40: a steep stratum with many valid rays
        a, b, seed = draw_chunk(idx, N, H, SPP, sx, sw)
    f0, f0b, f3 = g0.trace(a, b, seed, SPP), g0b.trace(a, b, seed, SPP), g3.trace(a, b, seed, SPP)
    same, near_same = affected_flags(f0, f0b)
    aff, near = affected_flags(f0, f3)
    un = ~aff & f3["valid"]
    eq = all(torch.equal(f0[k][un], f3[k][un]) for k in ("xi", "wo", "xo", "throughput"))
    # unaffected T3 first hits are not on the moved milk pot (distance check on a subsample)
    import numpy as np
    from scipy.spatial import cKDTree
    from teaset_gt_states import surface_samples

    x = f3["xi"][un][::50].cpu().numpy().astype(np.float64)
    d_mover3 = cKDTree(surface_samples("teapot2", st["T3"])).query(x)[0]
    on_mover = float(np.mean(d_mover3 < 2e-3))
    return {"pass": bool(int(same.sum()) == 0 and eq and on_mover < 0.01 and int(aff.sum()) > 0),
            "affected_same_scene": int(same.sum()), "affected_T0_T3": int(aff.sum()), "valid_T3": int(f3["valid"].sum()),
            "unaffected_fields_bit_identical": bool(eq), "unaffected_share_on_mover_T3": on_mover,
            "near_tolerance_T0_T3": int(near.sum())}


def t_mask_determinism(oracle_dir_a, oracle_dir_b):
    import numpy as np

    a, b = np.load(Path(oracle_dir_a) / "oracle_cells.npz"), np.load(Path(oracle_dir_b) / "oracle_cells.npz")
    keys = ("W3", "A3", "W0", "A0", "M95", "S_spatial")
    eq = {k: bool(np.array_equal(a[k], b[k])) for k in keys}
    return {"pass": all(eq.values()), "equal": eq}


def t_timing_accounting():
    import nrc_records as R
    from sro_metrics import adaptation_times

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "events.jsonl"
        rows = [
            {"t": 100.0, "pc": 0.0, "stage": "s", "event": "process_created", "kind": "point", "t_created": 98.0},
            {"t": 101.0, "pc": 1.0, "stage": "s", "event": "validation", "kind": "begin", "key": "build"},
            {"t": 111.0, "pc": 11.0, "stage": "s", "event": "validation", "kind": "end", "key": "build"},
            {"t": 111.0, "pc": 11.0, "stage": "s", "event": "dataset_init", "kind": "begin"},
            {"t": 112.0, "pc": 12.0, "stage": "s", "event": "dataset_init", "kind": "end"},
            {"t": 112.0, "pc": 12.0, "stage": "s", "event": "snapshot_write", "kind": "begin", "key": 0},
            {"t": 113.0, "pc": 13.0, "stage": "s", "event": "snapshot_write", "kind": "end", "key": 0},
            {"t": 113.0, "pc": 13.0, "stage": "s", "event": "validation", "kind": "begin", "key": 0},
            {"t": 116.0, "pc": 16.0, "stage": "s", "event": "validation", "kind": "end", "key": 0},
            {"t": 116.0, "pc": 16.0, "stage": "s", "event": "path_generation", "kind": "begin", "key": 0},
            {"t": 120.0, "pc": 20.0, "stage": "s", "event": "path_generation", "kind": "end", "key": 0},
            {"t": 120.0, "pc": 20.0, "stage": "s", "event": "optimization", "kind": "begin", "key": 0},
            {"t": 130.0, "pc": 30.0, "stage": "s", "event": "optimization", "kind": "end", "key": 0},
            {"t": 130.0, "pc": 30.0, "stage": "s", "event": "snapshot_write", "kind": "begin", "key": 128},
            {"t": 131.0, "pc": 31.0, "stage": "s", "event": "snapshot_write", "kind": "end", "key": 128},
        ]
        p.write_text("\n".join(json.dumps(r) for r in rows))
        t, _ = adaptation_times(p)
    r = t[128]
    # elapsed 131-98 = 33; minus validation 10+3, minus earlier snapshot 1 -> 19
    ok = abs(r["elapsed_s"] - 33) < 1e-9 and abs(r["adaptation_latency_s"] - 19) < 1e-9 and abs(r["optimization_s"] - 10) < 1e-9 \
        and abs(r["path_generation_s"] - 4) < 1e-9 and abs(r["process_setup_and_other_s"] - (19 - 10 - 4 - 1 - 1)) < 1e-9
    return {"pass": bool(ok), "row": r}


def t_checkpoint_roundtrip():
    import torch
    from models.integrator import load_asset

    model, hp = _model()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.ckpt"
        torch.save(S.released_layout(model, hp), p)
        m2, _ = S.load_model(p)
        a = load_asset("8dna", str(p))
        sd_up = {k.replace("_orig_mod.", ""): v for k, v in a.state_dict().items()}
        same_train = S.param_digest(model.state_dict()) == S.param_digest(m2.state_dict())
        same_up = S.param_digest(model.state_dict()) == S.param_digest(sd_up)
    rel = torch.load(S.released_checkpoint(), map_location="cpu", weights_only=False)["state_dict"]
    same_rel = S.param_digest({k[6:]: v for k, v in rel.items()}) == S.param_digest(model.state_dict())
    return {"pass": bool(same_train and same_up and same_rel), "training_load": same_train, "upstream_load_asset": same_up,
            "equals_released": same_rel}


def t_data_separation():
    """Seeds distinct; training reads only the mask arrays of the oracle; regions read no run output."""
    p = S.protocol()
    seeds = [p["oracle"]["seeds"]["primary"], p["oracle"]["seeds"]["stability_replicate"], 40040]
    distinct = len(set(seeds)) == 3
    tr = (S.HERE / "sro_train.py").read_text(encoding="utf-8")
    rg = (S.HERE / "sro_regions.py").read_text(encoding="utf-8")
    train_reads = all(s not in tr for s in ("gt_A", "gt_B", "w21_reference_correction", "regions.npz", "renders"))
    regions_reads = all(s not in rg for s in ("runs", "snapshots", "oracle_cells", ".ckpt"))
    return {"pass": bool(distinct and train_reads and regions_reads), "seeds": seeds,
            "train_reads_no_reference": train_reads, "regions_read_no_model": regions_reads}


TESTS = {
    "state_units": t_state_units,
    "masking_and_equivalence": t_masking_and_equivalence,
    "upstream_loss_equivalence": t_upstream_loss_equivalence,
    "dataset_equivalence": t_dataset_equivalence,
    "crn_determinism_and_correspondence": t_crn_determinism_and_correspondence,
    "timing_accounting": t_timing_accounting,
    "checkpoint_roundtrip": t_checkpoint_roundtrip,
    "data_separation": t_data_separation,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--mask-dirs", nargs=2, help="two oracle outputs of the same seed (mask determinism)")
    args = ap.parse_args()
    S.init()
    import nrc_records as R
    import ednalib as L

    res = {}
    names = args.only or list(TESTS)
    if args.mask_dirs:
        names = names + ["mask_determinism"]
    for n in names:
        t = time.perf_counter()
        try:
            r = t_mask_determinism(*args.mask_dirs) if n == "mask_determinism" else TESTS[n]()
        except Exception:
            r = {"pass": False, "error": traceback.format_exc()}
        r["seconds"] = time.perf_counter() - t
        res[n] = r
        print(n, "PASS" if r["pass"] else "FAIL", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prev = json.loads((out / "tests.json").read_text()) if (out / "tests.json").exists() else {}
    prev.update(res)
    S.write_json(out / "tests.json", dict(prev, git=R.git_state(L.ROOT)))
    return 0 if all(r["pass"] for r in res.values()) else 1


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(rc)
