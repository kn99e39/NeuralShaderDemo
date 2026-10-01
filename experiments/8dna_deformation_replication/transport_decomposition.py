"""Reference-only transport decomposition of the teaset interaction ROI (worklog 24).

    windows/run.ps1 transport_decomposition.py roi   --regime common_light --state T3 --seed A
    windows/run.ps1 transport_decomposition.py image --regime common_light --state T3
    windows/run.ps1 transport_decomposition.py analyze
    windows/run.ps1 transport_decomposition.py exports --worklog 24

A project-owned diagnostic path tracer: a primal re-implementation of Mitsuba
3.5.1's prb/path estimator (emitter sampling with visibility + BSDF sampling,
power-heuristic MIS, Russian roulette from depth 5, unbounded depth) that adds
every radiance contribution to exactly one path class (protocol/
teaset_transport_decomposition.json).  The classes therefore partition the
estimate; whether the estimator itself is right is checked against the
canonical references (Mitsuba C++ path) of the accepted experiment.  The
canonical reference renderer is not touched.

Classes, with v = number of scattering vertices x1..xv between camera and
emitter (x1 = visible point) and part identity from the hit mesh pointer:
  camera_emitter (v=0), direct (v=1), x2_<part> (v=2, by the part of x2),
  higher_mover / higher_other (v>=3, whether the mover is among x2..xv).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

import ednalib as L
import teaset_parts as T

PROTO = "protocol/teaset_transport_decomposition.json"
PARTS = ("plate", "teapot2", "teapot3", "teapot4")
CLASSES = ("camera_emitter", "direct", "x2_plate", "x2_teapot2", "x2_teapot3", "x2_teapot4", "x2_other",
           "higher_mover", "higher_other")
MOVER = "teapot2"
OUT = "transport_decomposition"


def regime_setup(regime: str, state: str):
    """(scene, translations, part ids, roi npz, reference paths) for one regime/state."""
    mi, _ = L.init_upstream()
    locked = json.loads(open(L.EXPERIMENT / "protocol/teaset_frozen_locked.json", encoding="utf-8").read())
    tr = locked["states"][state]
    if regime == "common_light":
        cb = json.loads(open(L.EXPERIMENT / "protocol/teaset_cross_backbone_locked.json", encoding="utf-8").read())
        scene = mi.load_dict(T.scene_dict(512, tr, cb["lighting"]))
        roi_dir, ref_dir = L.RESULTS / "gt_design/common_light", L.RESULTS / cb["frozen_output"] / state
    elif regime == "w21_envmap":
        scene = mi.load_dict(T.scene_dict(512, tr, None))
        roi_dir, ref_dir = L.RESULTS / "gt_design/v2", L.RESULTS / "w21_reference_correction" / state
    else:
        raise ValueError(regime)
    ids = T.part_shape_ids(scene, tr)
    return scene, tr, ids, roi_dir / f"rois_{state}.npz", (ref_dir / "gt_A.exr", ref_dir / "gt_B.exr")


def pixel_rays(sensor, res: int, pix: np.ndarray, offsets: np.ndarray):
    """Camera rays for pixels `pix` (flat indices), offsets (len(pix)*k, 2) in [0,1); lane = i*k + j."""
    mi, _ = L.init_upstream()
    k = len(offsets) // len(pix)
    xy = np.stack([pix % res, pix // res], -1).astype(np.float64)
    film = (np.repeat(xy, k, 0) + offsets) / res
    ray, _ = sensor.sample_ray(0.0, 0.5, mi.Point2f(*film.T.astype(np.float32)), mi.Point2f(0.5, 0.5))
    return ray


def mis_weight(a, b):
    _, dr = L.init_upstream()
    a2, b2 = dr.sqr(a), dr.sqr(b)
    w = a2 / (a2 + b2)
    return dr.select(dr.isfinite(w), w, 0.0)


def decompose(scene, ray, sampler, ids: dict[str, int], rr_depth: int = 5):
    """Per-lane radiance per class (dict name -> (n,3) numpy) and the plain total."""
    mi, dr = L.init_upstream()
    if dr.flag(dr.JitFlag.LoopRecord):
        raise RuntimeError("the decomposition relies on wavefront mode (LoopRecord off)")
    ctx = mi.BSDFContext()
    n = dr.width(ray.o)
    ray = mi.Ray3f(ray)
    acc = [mi.Spectrum(0.0) for _ in CLASSES]
    total = mi.Spectrum(0.0)
    beta = mi.Spectrum(1.0)
    eta = mi.Float(1.0)
    depth = mi.UInt32(0)
    active = mi.Bool(True)
    prev_si = dr.zeros(mi.SurfaceInteraction3f, n)
    prev_pdf = mi.Float(1.0)
    prev_delta = mi.Bool(True)
    mover_seen = mi.Bool(False)
    x2 = mi.UInt32(CLASSES.index("x2_other"))
    pid = {p: mi.UInt32(ids[p]) for p in PARTS}
    mover = pid[MOVER]
    it = [0]  # number of completed iterations; wavefront mode runs the body once per iteration

    def add(contrib, v: int):
        nonlocal total
        S = mi.Spectrum
        contrib = S(contrib)
        total = S(total + contrib)
        if v == 0:
            acc[0] = S(acc[0] + contrib)
        elif v == 1:
            acc[1] = S(acc[1] + contrib)
        elif v == 2:
            for c in range(2, 7):
                acc[c] = S(acc[c] + dr.select(dr.eq(x2, c), contrib, S(0.0)))
        else:
            acc[7] = S(acc[7] + dr.select(mover_seen, contrib, S(0.0)))
            acc[8] = S(acc[8] + dr.select(mover_seen, S(0.0), contrib))

    loop = mi.Loop(name="transport decomposition",
                   state=lambda: (sampler, ray, beta, eta, depth, active, prev_si, prev_pdf, prev_delta,
                                  mover_seen, x2, total, *acc))
    while loop(active):
        d = it[0]
        si = scene.ray_intersect(ray, active)
        # emission reached by the previous BSDF sample: path x1..x_d -> emitter
        ds = mi.DirectionSample3f(scene, si, prev_si)
        em_pdf = scene.pdf_emitter_direction(prev_si, ds, active & ~prev_delta)
        le = beta * mis_weight(prev_pdf, em_pdf) * ds.emitter.eval(si, active)
        add(dr.select(active, le, mi.Spectrum(0.0)), d)
        # x_{d+1} joins the path
        sid = T.shape_ids(si)
        hit = active & si.is_valid()
        if d + 1 >= 2:
            mover_seen |= hit & dr.eq(sid, mover)
        if d + 1 == 2:
            x2 = mi.UInt32(CLASSES.index("x2_other"))
            for p in PARTS:
                x2 = dr.select(hit & dr.eq(sid, pid[p]), CLASSES.index(f"x2_{p}"), x2)
        active_next = hit
        bsdf = si.bsdf(ray)
        # emitter sampling at x_{d+1}: path x1..x_{d+1} -> emitter
        active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
        dse, em_w = scene.sample_emitter_direction(si, sampler.next_2d(), True, active_em)
        active_em &= dr.neq(dse.pdf, 0.0)
        wo = si.to_local(dse.d)
        s1, s2 = sampler.next_1d(), sampler.next_2d()
        bval, bpdf, bs, bweight = bsdf.eval_pdf_sample(ctx, si, wo, s1, s2, active_next)
        mis_em = dr.select(dse.delta, 1.0, mis_weight(dse.pdf, bpdf))
        add(dr.select(active_em, beta * mis_em * bval * em_w, mi.Spectrum(0.0)), d + 1)
        ray = si.spawn_ray(si.to_world(bs.wo))
        beta = mi.Spectrum(dr.select(active_next, beta * bweight, beta))
        eta = mi.Float(dr.select(active_next, eta * bs.eta, eta))
        prev_si = dr.detach(si, True)
        prev_pdf = bs.pdf
        prev_delta = mi.has_flag(bs.sampled_type, mi.BSDFFlags.Delta)
        depth = dr.select(hit, depth + 1, depth)
        bmax = dr.max(beta)
        active_next &= dr.neq(bmax, 0.0)
        rr_prob = dr.minimum(bmax * dr.sqr(eta), 0.95)
        rr_active = depth >= rr_depth
        beta = mi.Spectrum(dr.select(rr_active, beta * dr.rcp(rr_prob), beta))
        rr_continue = sampler.next_1d() < rr_prob
        active = active_next & (~rr_active | rr_continue)
        it[0] += 1
    vec = lambda s: np.stack([np.array(s.x), np.array(s.y), np.array(s.z)], -1)
    return {c: vec(a) for c, a in zip(CLASSES, acc)}, vec(total), it[0]


def render_pixels(regime: str, state: str, pix: np.ndarray, spp: int, seed: int, max_lanes: int = 4_000_000):
    """Per-pixel class radiance (n_pix, n_classes, 3), plain total and the x1 part histogram."""
    mi, dr = L.init_upstream()
    scene, tr, ids, _, _ = regime_setup(regime, state)
    sensor = scene.sensors()[0]
    res = sensor.film().crop_size()[0]
    k = max(1, min(spp, max_lanes // len(pix)))
    while spp % k:
        k -= 1
    cls = np.zeros((len(pix), len(CLASSES), 3))
    tot = np.zeros((len(pix), 3))
    max_err = 0.0
    iters = []
    x1_parts = np.zeros(len(PARTS) + 1)
    t = time.time()
    for c in range(spp // k):
        rng = np.random.default_rng(seed * 100003 + c)
        ray = pixel_rays(sensor, res, pix, rng.random((len(pix) * k, 2)))
        if c == 0:
            si = scene.ray_intersect(ray)
            sid = np.array(T.shape_ids(si))
            for i, p in enumerate(PARTS):
                x1_parts[i] = np.mean(sid == ids[p])
            x1_parts[-1] = 1 - x1_parts[:-1].sum()
        sampler = mi.load_dict({"type": "independent"})
        sampler.seed((seed * 65536 + c) & 0xFFFFFFFF, len(pix) * k)
        per, total, n_it = decompose(scene, ray, sampler, ids)
        stack = np.stack([per[name] for name in CLASSES], 1)
        max_err = max(max_err, float(np.abs(stack.sum(1) - total).max()))
        cls += stack.reshape(len(pix), k, len(CLASSES), 3).sum(1)
        tot += total.reshape(len(pix), k, 3).sum(1)
        iters.append(n_it)
        if c % 8 == 0:
            print(f"  {regime} {state} chunk {c + 1}/{spp // k}  {time.time() - t:.0f}s", flush=True)
    return {"classes": cls / spp, "total": tot / spp, "per_sample_partition_max_abs_err": max_err,
            "iterations": iters, "x1_parts": x1_parts, "part_ids": ids, "translations": tr,
            "seconds": time.time() - t}


def cmd_roi(args) -> int:
    proto = json.loads(open(L.EXPERIMENT / PROTO, encoding="utf-8").read())
    _, tr, _, roi_path, _ = regime_setup(args.regime, args.state)
    z = np.load(roi_path)
    pix = np.flatnonzero(z["mask_interaction"].reshape(-1))
    spp = proto["sampling"]["roi_spp"][args.regime]
    seed = proto["sampling"]["seeds"][args.seed]
    r = render_pixels(args.regime, args.state, pix, spp, seed)
    out = L.RESULTS / OUT / args.regime / f"roi_{args.state}_{args.seed}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, pix=pix, classes=r["classes"].astype(np.float32), total=r["total"].astype(np.float32),
                        class_names=np.array(CLASSES), x1_parts=r["x1_parts"], spp=spp, seed=seed,
                        partition_max_abs_err=r["per_sample_partition_max_abs_err"],
                        provenance=json.dumps({"environment": L.environment_record(), "regime": args.regime,
                                               "state": args.state, "translations": tr,
                                               "part_ids": r["part_ids"], "seconds": r["seconds"],
                                               "iterations_per_chunk": r["iterations"]}, default=str))
    print("wrote", L.rel(out), f"{r['seconds']:.0f}s", "partition err", r["per_sample_partition_max_abs_err"],
          "x1 parts", np.round(r["x1_parts"], 4).tolist())
    return 0


def cmd_image(args) -> int:
    proto = json.loads(open(L.EXPERIMENT / PROTO, encoding="utf-8").read())
    pix = np.arange(512 * 512)
    r = render_pixels(args.regime, args.state, pix, args.spp, proto["sampling"]["seeds"]["A"])
    out = L.RESULTS / OUT / args.regime / f"image_{args.state}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, classes=r["classes"].reshape(512, 512, len(CLASSES), 3).astype(np.float32),
                        total=r["total"].reshape(512, 512, 3).astype(np.float32), class_names=np.array(CLASSES),
                        spp=args.spp, provenance=json.dumps({"environment": L.environment_record()}, default=str))
    print("wrote", L.rel(out), f"{r['seconds']:.0f}s")
    return 0


# ---------------------------------------------------------------- analysis
def groups(proto) -> dict[str, list[int]]:
    return {g: [CLASSES.index(c) for c in members] for g, members in proto["groups"].items()}


def analyze_regime(proto, regime: str) -> dict:
    states = ("T0", "T1", "T1b", "T2", "T3")
    G = groups(proto)
    load = lambda s, sd: np.load(L.RESULTS / OUT / regime / f"roi_{s}_{sd}.npz")
    data = {(s, sd): load(s, sd) for s in states for sd in ("A", "B")}
    rec = {"validity": {}, "states": {}}
    # reference agreement
    for s in states:
        _, _, _, roi_path, refs = regime_setup(regime, s)
        pix = data[(s, "A")]["pix"]
        ra, rb = (L.load_exr(p).reshape(-1, 3)[pix] for p in refs)
        diag = 0.5 * (data[(s, "A")]["total"] + data[(s, "B")]["total"])
        ref = 0.5 * (ra + rb)
        tol = max(0.01, 3 * abs(ra.mean() / rb.mean() - 1))
        mae_d = float(np.abs(L.tonemap(data[(s, "A")]["total"]) - L.tonemap(ra)).mean())
        mae_r = float(np.abs(L.tonemap(rb) - L.tonemap(ra)).mean())
        rec["validity"][s] = {"diag_mean": float(diag.mean()), "ref_mean": float(ref.mean()),
                              "mean_ratio": float(diag.mean() / ref.mean()), "mean_tolerance": tol,
                              "display_mae_diagA_vs_refA": mae_d, "display_mae_refB_vs_refA": mae_r,
                              "partition_max_abs_err": max(float(data[(s, sd)]["partition_max_abs_err"]) for sd in "AB"),
                              "x1_parts": data[(s, "A")]["x1_parts"].tolist(),
                              "pass": abs(diag.mean() / ref.mean() - 1) <= tol and mae_d <= 1.5 * mae_r}
    # component repeat noise at T0 (single-render A vs B)
    noise = {g: float(np.abs(data[("T0", "A")]["classes"][:, idx].sum(1) - data[("T0", "B")]["classes"][:, idx].sum(1)).mean())
             for g, idx in G.items()}
    noise["total"] = float(np.abs(data[("T0", "A")]["total"] - data[("T0", "B")]["total"]).mean())
    rec["repeat_noise_T0"] = noise
    for s in states:
        z = np.load(regime_setup(regime, s)[3])
        m, i0 = z["mask_interaction"].reshape(-1), z["idx0_interaction"]
        pix_s = data[(s, "A")]["pix"]
        pos0 = {p: i for i, p in enumerate(data[("T0", "A")]["pix"])}
        row0 = np.array([pos0[p] for p in i0])
        assert np.array_equal(pix_s, np.flatnonzero(m))

        def comp(sd_s, sd_0):
            cs, c0 = data[(s, sd_s)]["classes"], data[("T0", sd_0)]["classes"][row0]
            return cs.astype(np.float64) - c0

        dC = {sd: comp(sd, sd) for sd in ("A", "B")}
        dCm = 0.5 * (dC["A"] + dC["B"])

        def shares(dc):
            dg = dc.sum(1)
            den = float((dg * dg).sum())
            return {g: (float((dc[:, idx].sum(1) * dg).sum() / den) if den > 0 else None) for g, idx in G.items()}

        sh, sa, sb = shares(dCm), shares(dC["A"]), shares(dC["B"])
        tot_s = 0.5 * (data[(s, "A")]["total"] + data[(s, "B")]["total"])
        tot_0 = 0.5 * (data[("T0", "A")]["total"] + data[("T0", "B")]["total"])[row0]
        dg = dCm.sum(1)
        st = {"total_mean": float(tot_s.mean()), "dG_mean_signed": float(dg.mean()), "dG_mean_abs": float(np.abs(dg).mean()),
              "dG_display_mae": float(np.abs(L.tonemap(tot_s) - L.tonemap(tot_0)).mean()), "groups": {}, "classes": {}}
        for g, idx in G.items():
            dcg = dCm[:, idx].sum(1)
            mag = float(np.abs(dcg).mean())
            st["groups"][g] = {"mean_T0": float(data[("T0", "A")]["classes"][:, idx].sum(1).mean()),
                               "mean_state": float(0.5 * (data[(s, "A")]["classes"][:, idx].sum(1) + data[(s, "B")]["classes"][:, idx].sum(1)).mean()),
                               "change_signed": float(dcg.mean()), "change_abs": mag,
                               "change_over_noise": mag / noise[g] if noise[g] > 0 else None,
                               "significant": bool(noise[g] > 0 and mag > 3 * noise[g]) or bool(noise[g] == 0 and mag > 0),
                               "share": sh[g], "share_A": sa[g], "share_B": sb[g],
                               "share_seed_gap": abs(sa[g] - sb[g]) if sa[g] is not None else None}
        for ci, c in enumerate(CLASSES):
            st["classes"][c] = {"change_signed": float(dCm[:, ci].mean()), "change_abs": float(np.abs(dCm[:, ci]).mean()),
                                "share": float((dCm[:, ci] * dg).sum() / max((dg * dg).sum(), 1e-30))}
        st["share_sum_check"] = float(sum(st["classes"][c]["share"] for c in CLASSES))
        rec["states"][s] = st
    # decision
    dec = proto["decision"]
    used = ("direct", "mover_mediated", "higher_order", "mover_single", "higher_mover")
    valid = all(v["pass"] for v in rec["validity"].values())
    out = {}
    for s in ("T1b", "T2", "T3"):
        gs = rec["states"][s]["groups"]
        stable = all(gs[g]["share_seed_gap"] is not None and gs[g]["share_seed_gap"] <= 0.10 for g in used)
        if not (valid and stable):
            label = "E"
        elif gs["mover_mediated"]["share"] >= 0.6 and gs["mover_mediated"]["significant"]:
            label = "A (single-bounce)" if gs["mover_single"]["share"] >= gs["higher_mover"]["share"] else "A (higher-order)"
        elif gs["direct"]["share"] >= 0.6 and gs["direct"]["significant"]:
            label = "B"
        elif gs["higher_order"]["share"] >= 0.6 and gs["higher_order"]["significant"]:
            label = "C"
        else:
            top = sorted(("direct", "mover_single", "stationary_single", "higher_mover", "higher_other"),
                         key=lambda g: -abs(gs[g]["share"]))[:2]
            label = "D (" + ", ".join(top) + ")"
        out[s] = {"label": label, "validity_pass": valid, "share_stability_pass": stable}
    rec["decision"] = out
    rec["batch_answer"] = out["T3"]["label"]
    return rec


def cmd_analyze(args) -> int:
    proto = json.loads(open(L.EXPERIMENT / PROTO, encoding="utf-8").read())
    L.init_upstream()
    rec = {"protocol": proto, "environment": L.environment_record(), "regimes": {}}
    for regime in proto["regimes"]:
        rec["regimes"][regime] = analyze_regime(proto, regime)
    L.write_json(L.RESULTS / OUT / "decomposition.json", rec)
    for regime, r in rec["regimes"].items():
        print(f"== {regime}: answer {r['batch_answer']}  validity", {s: v["pass"] for s, v in r["validity"].items()})
        for s, st in r["states"].items():
            print(f"  {s:4s} dG|.| {st['dG_mean_abs']:.4f} " + "  ".join(
                f"{g} {st['groups'][g]['share']:+.2f}" for g in ("direct", "mover_single", "stationary_single", "higher_mover", "higher_other")))
        print("  decision", r["decision"])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("roi")
    a.add_argument("--regime", required=True)
    a.add_argument("--state", required=True)
    a.add_argument("--seed", choices=("A", "B"), required=True)
    b = sub.add_parser("image")
    b.add_argument("--regime", required=True)
    b.add_argument("--state", required=True)
    b.add_argument("--spp", type=int, default=1024)
    sub.add_parser("analyze")
    e = sub.add_parser("exports")
    e.add_argument("--worklog", required=True)
    args = ap.parse_args()
    if args.cmd == "exports":
        import decomposition_exports
        return decomposition_exports.main(args.worklog)
    return {"roi": cmd_roi, "image": cmd_image, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    rc = main()
    # Outputs are complete here; skip the interpreter teardown, which crashes with
    # an access violation on this Windows setup once DrJit and torch are loaded.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
