"""T0-only capacity pilot (never reads T3): units, target throughput, LS fit, camera indirect error.

    windows/run.ps1 ../surface_owned_radiance/sor_pilot.py --out <dir> --h 0.02 --res 32 --n-train 20000000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import sor_common as S


def fit(model, data, steps, batch, lr, seed, log_every=200, table0=None):
    import torch

    g = torch.Generator(device="cuda").manual_seed(seed)
    tab = torch.nn.Parameter(model.table.clone() if table0 is None else table0.clone())
    opt = torch.optim.Adam([tab], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.05)
    N = len(data["y"])
    hist = []
    for it in range(steps):
        idx = torch.randint(0, N, (batch,), device="cuda", generator=g)
        ids, w = data["ids"][idx].long(), data["wts"][idx].float()
        rows, ws = model.texel_stencil(ids, w, data["w"][idx])
        pred = model.query(rows, ws, tab)
        loss = ((pred - data["y"][idx]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        if it % log_every == 0 or it == steps - 1:
            hist.append((it, float(loss)))
    model.table = tab.detach()
    return hist


def fit_online(model, scene, ids, tr, steps, batch, lr, seed, parts=None, table0=None, log_every=250, heldout=None):
    """Online LS fit: every step draws fresh queries, traces 1-spp MC targets, takes one Adam step."""
    import numpy as np
    import torch
    from sor_targets import sample_queries, trace

    rng = np.random.default_rng(seed)
    tab = torch.nn.Parameter(model.table.clone() if table0 is None else table0.clone())
    opt = torch.optim.Adam([tab], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.05)
    hist, t_trace, t_opt = [], 0.0, 0.0
    for it in range(steps):
        t0 = time.perf_counter()
        q = sample_queries(batch, rng, tr, parts=parts)
        _, y, ok = trace(scene, ids, q, seed=seed * 100_000 + it)
        sel = np.flatnonzero(ok)
        torch.cuda.synchronize(); t1 = time.perf_counter(); t_trace += t1 - t0
        xl = torch.tensor(q["x_local"][sel], dtype=torch.float32, device="cuda")
        pi = torch.tensor(q["part"][sel], dtype=torch.long, device="cuda")
        nn = torch.tensor(q["n"][sel], dtype=torch.float32, device="cuda")
        ww = torch.tensor(q["w"][sel], dtype=torch.float32, device="cuda")
        yy = torch.tensor(y[sel], dtype=torch.float32, device="cuda")
        i_, w_ = model.stencil(xl, pi, nn)
        rows, ws = model.texel_stencil(i_, w_, ww)
        pred = model.query(rows, ws, tab)
        loss = ((pred - yy) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        torch.cuda.synchronize(); t_opt += time.perf_counter() - t1
        if it % log_every == 0 or it == steps - 1:
            h = None
            if heldout is not None:
                with torch.no_grad():
                    ph = model.query(heldout["rows"], heldout["ws"], tab)
                    h = float((ph - heldout["y"]).abs().mean() / heldout["y"].abs().mean())
            hist.append({"step": it, "loss": float(loss.detach()), "heldout_rel": h, "trace_s": t_trace, "opt_s": t_opt})
            print(hist[-1], flush=True)
    model.table = tab.detach()
    return hist


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--h", type=float, default=0.02)
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--n-train", type=int, default=20_000_000)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=262144)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--sub", type=int, default=4)
    ap.add_argument("--online", action="store_true")
    args = ap.parse_args()
    mi, dr = S.init()
    import numpy as np
    import torch
    import ednalib as L
    from sor_targets import camera_queries, sample_queries, scene_for, trace

    out = S.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rec = {"args": vars(args)}
    t = time.time()
    anchors = S.build_anchors(args.h)
    model = S.SurfaceRadiance(anchors, args.res)
    rec["units"] = {"total": model.n_units, "per_part": {p: anchors["grids"][p]["count"] for p in S.PARTS},
                    "table_entries": model.n_units * args.res ** 2, "build_s": time.time() - t}
    print(rec["units"], flush=True)

    scene, ids, tr = scene_for("T0")
    if args.online:
        qh = sample_queries(200_000, np.random.default_rng(2), tr)
        _, yh, okh = trace(scene, ids, qh, seed=22, spp=16)
        sel = np.flatnonzero(okh)
        xl = torch.tensor(qh["x_local"][sel], dtype=torch.float32, device="cuda")
        i_, w_ = model.stencil(xl, torch.tensor(qh["part"][sel], device="cuda"), torch.tensor(qh["n"][sel], dtype=torch.float32, device="cuda"))
        rows, ws = model.texel_stencil(i_, w_, torch.tensor(qh["w"][sel], dtype=torch.float32, device="cuda"))
        hd = {"rows": rows, "ws": ws, "y": torch.tensor(yh[sel], dtype=torch.float32, device="cuda")}
        t = time.time()
        rec["fit_hist"] = fit_online(model, scene, ids, tr, args.steps, args.batch, args.lr, 3, heldout=hd)
        rec["fit_s"] = time.time() - t
        return finish(args, rec, model, scene, ids, tr, out)
    rng = np.random.default_rng(1)
    q = sample_queries(args.n_train, rng, tr)
    t = time.time()
    _, y, ok = trace(scene, ids, q, seed=11)
    torch.cuda.synchronize()
    rec["targets"] = {"n": args.n_train, "accepted": float(ok.mean()), "seconds": time.time() - t,
                      "per_s": args.n_train / (time.time() - t),
                      "y_mean": y[ok].mean(0).tolist(), "y_p99": np.percentile(y[ok].mean(1), 99).tolist(),
                      "y_max": float(y[ok].max())}
    print(rec["targets"], flush=True)

    def to_gpu(q, y, ok):
        sel = np.flatnonzero(ok)
        xl = torch.tensor(q["x_local"][sel], dtype=torch.float32, device="cuda")
        pi = torch.tensor(q["part"][sel], dtype=torch.long, device="cuda")
        n = torch.tensor(q["n"][sel], dtype=torch.float32, device="cuda")
        idl, wl = [], []
        for s0 in range(0, len(sel), 500_000):
            i_, w_ = model.stencil(xl[s0:s0 + 500_000], pi[s0:s0 + 500_000], n[s0:s0 + 500_000])
            idl.append(i_.int()); wl.append(w_.half())
        return {"ids": torch.cat(idl), "wts": torch.cat(wl), "w": torch.tensor(q["w"][sel], dtype=torch.float32, device="cuda"),
                "y": torch.tensor(y[sel], dtype=torch.float32, device="cuda")}

    t = time.time()
    data = to_gpu(q, y, ok)
    rec["stencil_s"] = time.time() - t
    rec["empty_stencil_fraction"] = float((data["ids"][:, 0] < 0).float().mean())
    t = time.time()
    rec["fit_hist"] = fit(model, data, args.steps, args.batch, args.lr, 0)
    torch.cuda.synchronize(); rec["fit_s"] = time.time() - t
    del data; torch.cuda.empty_cache()

    # held-out queries (T0, separate seed), 16 spp each
    qh = sample_queries(200_000, np.random.default_rng(2), tr)
    _, yh, okh = trace(scene, ids, qh, seed=22, spp=16)
    dh = to_gpu(qh, yh, okh)
    with torch.no_grad():
        rows, ws = model.texel_stencil(dh["ids"].long(), dh["wts"].float(), dh["w"])
        ph = model.query(rows, ws)
    rec["heldout"] = {"mae": float((ph - dh["y"]).abs().mean()), "ref_mean_abs": float(dh["y"].abs().mean()),
                      "rel": float((ph - dh["y"]).abs().mean() / dh["y"].abs().mean())}
    print(rec["heldout"], flush=True)

    return finish(args, rec, model, scene, ids, tr, out)


def finish(args, rec, model, scene, ids, tr, out):
    import numpy as np
    import torch
    import ednalib as L
    from sor_targets import camera_queries, scene_for
    # camera indirect at T0 vs worklog-24 class image (1024 spp, seed A)
    scene512, ids512, _ = scene_for("T0", 512)
    cq = camera_queries(scene512, ids512, tr, 512, args.sub, seed=5)
    pred = np.zeros((len(cq["pix"]), 3))
    sel = np.flatnonzero(cq["valid"])
    with torch.no_grad():
        for s0 in range(0, len(sel), 400_000):
            j = sel[s0:s0 + 400_000]
            xl = torch.tensor(cq["x_local"][j], dtype=torch.float32, device="cuda")
            pi = torch.tensor(cq["part"][j], dtype=torch.long, device="cuda")
            nn = torch.tensor(cq["n"][j], dtype=torch.float32, device="cuda")
            i_, w_ = model.stencil(xl, pi, nn)
            rows, ws = model.texel_stencil(i_, w_, torch.tensor(cq["w"][j], dtype=torch.float32, device="cuda"))
            pred[j] = model.query(rows, ws).cpu().numpy()
    img = pred.reshape(512 * 512, args.sub ** 2, 3).mean(1).reshape(512, 512, 3)
    z = np.load(S.ROOT / "results/8dna_replication/transport_decomposition/w21_envmap/image_T0.npz")
    ref_ind = z["classes"][:, :, 2:9].sum(2)
    ref_dir = z["classes"][:, :, 0] + z["classes"][:, :, 1]
    reg = np.load(S.ROOT / "results/selective_refit_oracle/v1_bc085ea/regions/regions.npz")
    ref = S.ROOT / "results/8dna_replication/w21_reference_correction/T0"
    g0 = 0.5 * (L.load_exr(ref / "gt_A.exr") + L.load_exr(ref / "gt_B.exr"))
    tot = ref_dir + img
    disp = L.tonemap
    from teaset_frozen_eval import load_rois
    rois = load_rois(S.ROOT / "results/8dna_replication/gt_design/v2", ("T0",))
    m_int = rois["T0"]["interaction"][0].reshape(512, 512)
    n8 = L.load_exr(S.ROOT / "results/8dna_replication/frozen/teaset_locked/T0/8dna_upstream.exr")
    rec["camera_T0"] = {}
    for name, m in (("interaction", m_int), ("R_aff_stat", reg["R_aff_stat"]), ("R_unaff", reg["R_unaff"]),
                    ("asset", ~reg["background"])):
        rec["camera_T0"][name] = {
            "indirect_linear_mae": float(np.abs(img - ref_ind)[m].mean()), "ref_indirect_mean": float(ref_ind[m].mean()),
            "total_display_mae": float(np.abs(disp(tot) - disp(g0)).mean(-1)[m].mean()),
            "refdirect_plus_refindirect_display_mae": float(np.abs(disp(ref_dir + ref_ind) - disp(g0)).mean(-1)[m].mean()),
            "8dna_T0_display_mae": float(np.abs(disp(n8) - disp(g0)).mean(-1)[m].mean())}
    print(json.dumps(rec["camera_T0"], indent=1), flush=True)
    L.save_exr(out / "pred_indirect_T0.exr", img.astype(np.float32))
    L.save_exr(out / "total_T0.exr", tot.astype(np.float32))
    rec["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    S.write_json(out / "pilot.json", rec)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(rc)
