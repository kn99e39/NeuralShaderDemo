"""Online least-squares fitting of surface-owned units (shared by the pilot and every arm).

Every step draws `batch` fresh surface queries from a position pool, traces 1-spp Monte-Carlo
>= 2-vertex targets in the current geometry (sor_targets.trace_t), and takes one Adam step on the
TRAINABLE units only.  Trainable units are held in a sub-table parameter; every other unit is
read from a constant table, so frozen units are bit-identical by construction and the optimiser
cost scales with the trainable set.  Loss: mean squared error in linear radiance over accepted
queries (the least-squares estimate of the mean outgoing radiance).
"""

from __future__ import annotations

import math
import time

import sor_common as S


class PositionPool:
    """Area-uniform surface positions (canonical frame); optionally restricted to queries whose
    stencil reads at least one unit of `unit_mask` (the support of a unit set)."""

    def __init__(self, model, size: int, seed: int, unit_mask=None, chunk: int = 2_000_000):
        import torch

        smp = S.SurfaceSampler()
        g = torch.Generator(device="cuda").manual_seed(seed)
        xs, ns, ps = [], [], []
        drawn = 0
        t0 = time.perf_counter()
        while sum(len(x) for x in xs) < size:
            x, n, p = smp.positions(chunk, g)
            drawn += chunk
            if unit_mask is not None:
                ids, _ = model.stencil(x, p, n)
                keep = (unit_mask[ids.clamp(min=0)] & (ids >= 0)).any(1)
                x, n, p = x[keep], n[keep], p[keep]
            xs.append(x); ns.append(n); ps.append(p)
            if unit_mask is not None and drawn > 400 * size:
                break
        self.x = torch.cat(xs)[:size]; self.n = torch.cat(ns)[:size]; self.p = torch.cat(ps)[:size]
        self.drawn = drawn
        self.support_fraction = len(torch.cat(xs)) / drawn
        self.seconds = time.perf_counter() - t0

    def draw(self, k: int, gen):
        import torch

        i = torch.randint(0, len(self.x), (k,), device="cuda", generator=gen)
        return self.x[i], self.n[i], self.p[i]


def combined_lookup(rows, frozen, sub, remap):
    """Values of table rows: trainable rows from `sub` (via remap >= 0), the rest from `frozen`."""
    import torch

    r = remap[rows]
    tr = r >= 0
    v = frozen[rows]
    if sub is not None and tr.any():
        v = torch.where(tr[..., None], sub[r.clamp(min=0)], v)
    return v


def fit(model, scene, ids, translations, pool, steps: int, batch: int, lr: float, seed: int,
        unit_mask=None, table0=None, snapshot_steps=(), on_snapshot=None, log=None, lr_min_frac=0.05):
    """Fit; returns (final full table, history).  unit_mask None = all units trainable."""
    import torch
    from sor_targets import directions, trace_t

    _, dr = S.init()

    def sync():
        dr.sync_thread()
        torch.cuda.synchronize()

    R2 = model.res * model.res
    U = model.n_units
    frozen = (torch.zeros_like(model.table) if table0 is None else table0.clone()).contiguous()
    if unit_mask is None:
        unit_mask = torch.ones(U, dtype=torch.bool, device="cuda")
    train_units = torch.nonzero(unit_mask).squeeze(1)
    row_is_tr = unit_mask.repeat_interleave(R2)
    remap = torch.full((U * R2,), -1, dtype=torch.long, device="cuda")
    tr_rows = torch.nonzero(row_is_tr).squeeze(1)
    remap[tr_rows] = torch.arange(len(tr_rows), device="cuda")
    sub = torch.nn.Parameter(frozen[tr_rows].clone())
    opt = torch.optim.Adam([sub], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(steps, 1), eta_min=lr * lr_min_frac)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    hist = []
    t_trace = t_opt = 0.0
    n_acc = 0

    def snap(step):
        if on_snapshot is not None:
            full = frozen.clone()
            full[tr_rows] = sub.detach()
            on_snapshot(step, full, {"trace_s": t_trace, "opt_s": t_opt, "accepted": n_acc})

    if 0 in snapshot_steps:
        snap(0)
    for it in range(steps):
        sync(); t0 = time.perf_counter()
        x, n, p = pool.draw(batch, gen)
        q = directions(x, n, p, gen, translations)
        _, y, ok = trace_t(scene, ids, q, seed=(seed * 1_000_003 + it) & 0x7FFFFFFF)
        sync(); t1 = time.perf_counter(); t_trace += t1 - t0
        sel = torch.nonzero(ok).squeeze(1)
        n_acc += len(sel)
        i_, w_ = model.stencil(q["x_local"][sel], q["part"][sel], q["n"][sel])
        rows, ws = model.texel_stencil(i_, w_, q["w"][sel])
        vals = combined_lookup(rows, frozen, sub, remap)
        pred = (vals * ws[..., None]).sum(1)
        loss = ((pred - y[sel]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        sync(); t_opt += time.perf_counter() - t1
        if log is not None and (it % 100 == 0 or it == steps - 1):
            log(it, float(loss.detach()), t_trace, t_opt)
        if (it + 1) in snapshot_steps:
            snap(it + 1)
    full = frozen.clone()
    full[tr_rows] = sub.detach()
    return full, {"trace_s": t_trace, "opt_s": t_opt, "accepted": n_acc, "trainable_units": int(len(train_units)),
                  "trainable_entries": int(len(tr_rows)) * 3}


def render_indirect(model, table, cam, chunk: int = 400_000):
    """Indirect image (res, res, 3) from camera queries (sor_targets.camera_queries output, numpy)."""
    import numpy as np
    import torch

    res = int(round(math.sqrt(cam["pix"].max() + 1)))
    sub2 = len(cam["pix"]) // (res * res)
    pred = torch.zeros(len(cam["pix"]), 3, device="cuda")
    sel = np.flatnonzero(cam["valid"])
    with torch.no_grad():
        for s0 in range(0, len(sel), chunk):
            j = torch.tensor(sel[s0:s0 + chunk], device="cuda")
            xl = torch.tensor(cam["x_local"][sel[s0:s0 + chunk]], dtype=torch.float32, device="cuda")
            pi = torch.tensor(cam["part"][sel[s0:s0 + chunk]], dtype=torch.long, device="cuda")
            nn = torch.tensor(cam["n"][sel[s0:s0 + chunk]], dtype=torch.float32, device="cuda")
            ww = torch.tensor(cam["w"][sel[s0:s0 + chunk]], dtype=torch.float32, device="cuda")
            i_, w_ = model.stencil(xl, pi, nn)
            rows, ws = model.texel_stencil(i_, w_, ww)
            pred[j] = (table[rows] * ws[..., None]).sum(1)
    return pred.reshape(res * res, sub2, 3).mean(1).reshape(res, res, 3).cpu().numpy()


def camera_stencil_units(model, cam, chunk: int = 400_000):
    """Per pixel, the set of units its sub-pixel queries read (list of unique unit ids per pixel is
    not stored; returns (P*sub2, K) ids array) for frozen-output identity checks."""
    import numpy as np
    import torch

    ids_all = np.full((len(cam["pix"]), S.K_MAX), -1, np.int64)
    sel = np.flatnonzero(cam["valid"])
    with torch.no_grad():
        for s0 in range(0, len(sel), chunk):
            j = sel[s0:s0 + chunk]
            i_, w_ = model.stencil(torch.tensor(cam["x_local"][j], dtype=torch.float32, device="cuda"),
                                   torch.tensor(cam["part"][j], dtype=torch.long, device="cuda"),
                                   torch.tensor(cam["n"][j], dtype=torch.float32, device="cuda"))
            ids_all[j] = torch.where(w_ > 0, i_, torch.full_like(i_, -1)).cpu().numpy()
    return ids_all
