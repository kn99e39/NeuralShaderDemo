"""Architecture Batch 2 diagnostics A/C/D (worklog 34; RNA venv, GPU; no training).

    python b2_diag.py --cfg C2 --val <val.h5> --partitions <dir> --ref-dir <dir> --model-config <rna.yml> \
        --arms A=<dir> P4=<dir> P16=<dir> P16w=<dir> --out <dir>

Definitions are fixed in protocol/b2_v1.json ("diagnostics"); this script implements them on the
dev views 0/10/20/30 (worklog-33 8x-spp references). All radiance comparisons use the RNA tonemap
ldr(x) = (max(x,0)/(1+max(x,0)))^(1/2.2), channel-averaged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rna_surface_ownership"))
import rso_model as M  # noqa: E402  (puts upstream RNA on sys.path)

DEV_VIEWS = (0, 10, 20, 30)
BINS = [0, 1, 2, 4, 8, 16, 32, np.inf]
COS20 = float(np.cos(np.deg2rad(20.0)))


def ldr(x):
    x = np.clip(x, 0, None)
    return ((x / (1 + x)) ** (1 / 2.2)).mean(-1)


def wpearson(a, b, w):
    w = w / w.sum()
    ma, mb = (w * a).sum(), (w * b).sum()
    cov = (w * (a - ma) * (b - mb)).sum()
    return float(cov / np.sqrt((w * (a - ma) ** 2).sum() * (w * (b - mb) ** 2).sum() + 1e-30))


def load_arm(d, conf, aabb, dev):
    import torch
    from rna import models

    sd = torch.load(Path(d) / "state.pt", map_location=dev)
    if (Path(d) / "specs.json").exists():
        specs = [{"params": s["params"], "lo": np.array(s["lo"]), "hi": np.array(s["hi"])} for s in json.loads((Path(d) / "specs.json").read_text())]
        m = M.OwnedTriplaneSet(specs, aabb).to(dev); K = len(specs)
    else:
        m = models.NeuralSurfaceTriPlane(**dict(conf.model_params)).to(dev); K = 0
    m.load_state_dict(sd); m.eval(); m.set_iteration_index(10 ** 9)
    return m, K


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--partitions", required=True)
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import h5py
    import torch
    from rna import config
    from rso_partition import Owner
    from scipy.spatial import cKDTree

    dev = "cuda"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    conf = config.get_neumat_config(config_filename=str(Path(args.model_config).resolve()))
    f = h5py.File(args.val, "r")
    res = int(f.attrs["resolution"][0])
    aabb = (np.array(f.attrs["aabb_min"], np.float64), np.array(f.attrs["aabb_max"], np.float64))
    pz = np.load(Path(args.partitions) / "partition_P16.npz")
    X, LAB, unit_part = pz["x"], pz["label"], pz["unit_part"]
    trees = {u: cKDTree(X[LAB == u]) for u in range(16)}
    own16 = Owner(Path(args.partitions) / "partition_P16.npz")
    own4 = Owner(Path(args.partitions) / "partition_P4.npz")
    arms = {}
    for spec in args.arms:
        name, d = spec.split("=", 1)
        arms[name] = load_arm(d, conf, aabb, dev)
    T = lambda a: torch.tensor(np.asarray(a, np.float32), device=dev).unsqueeze(0)

    def run(model, K, pos_w, cam, light, nrm, vis, owner):
        pos = (pos_w - aabb[0]) / (aabb[1] - aabb[0])
        if K:
            o = {16: owner[16], 4: owner[4], 1: np.zeros(len(pos_w), np.int64)}[K]
            pos = np.concatenate([pos, o[:, None]], 1)
        with torch.no_grad():
            pr = model(T(pos), T(cam), T(light), T(nrm)).squeeze(0)
        return torch.where(torch.tensor(vis, device=dev)[:, None], pr[:, :3], pr[:, 3:]).cpu().numpy()

    def run_unit(model, k, pos_w, cam, light, nrm, vis):
        """Unit k of an OwnedTriplaneSet evaluated at arbitrary points (local frame clamped exactly as in forward)."""
        if len(pos_w) == 0:
            return np.zeros((0, 3), np.float32)
        lo, hi = model.lo[k].cpu().numpy(), model.hi[k].cpu().numpy()
        local = np.clip((pos_w - lo) / (hi - lo), 0, 1)
        dup = len(local) == 1
        args_ = [local, cam, light, nrm]
        if dup:
            args_ = [np.repeat(a, 2, 0) for a in args_]
        with torch.no_grad():
            pr = model.units[k](*(T(a) for a in args_)).squeeze(0).cpu().numpy()
        if dup:
            pr = pr[:1]
        return np.where(vis[:, None], pr[:, :3], pr[:, 3:])

    acc = {"seam": {}, "interior": {}, "cross_part": {}, "geometric_edge": {}, "seam_visdiff": {}}
    prof = {a: [[] for _ in range(len(BINS) - 1)] for a in arms}
    unit_err = {a: np.zeros(16) for a in arms}; unit_n = np.zeros(16)
    cross = {a: {"disagree": [], "own_err": [], "nbr_err": [], "inside_aabb": []} for a in arms if arms[a][1] == 16}
    blend = {a: {"J": [], "band_err_blend": [], "band_err_hard": [], "J_hard": []} for a in cross}
    pair_store = {"dref": [], "cls": []}
    per_arm_pairs = {a: {"J": [], "ep": [], "eq": []} for a in arms}
    footprints = []
    maps = {}
    for v in DEV_VIEWS:
        g = lambda k: np.nan_to_num(f[k][v].astype(np.float32))
        alpha = g("alpha")[:, 0]; m = alpha > 0; clean = alpha >= 0.999
        P, N, C, Lt = g("position"), g("normal"), g("camera_dir"), g("light_dir")
        vis = g("diffuse_direct")[:, 0] > 0
        ref = np.load(Path(args.ref_dir) / f"{args.cfg}_view{v}_ref.npz")["color"]
        o16 = -np.ones(res * res, np.int64); o16[m] = own16(P[m])[0]
        o4 = -np.ones(res * res, np.int64); o4[m] = own4(P[m])[0]
        idx = np.flatnonzero(m)
        preds = {}
        for a, (model, K) in arms.items():
            full = np.zeros((res * res, 3), np.float32)
            full[m] = run(model, K, P[m], C[m], Lt[m], N[m], vis[m], {16: o16[m], 4: o4[m]})
            preds[a] = full
        Lref = ldr(ref)
        E = {a: ldr(p) - Lref for a, p in preds.items()}
        # pairs
        img = np.arange(res * res).reshape(res, res)
        pp, qq = [], []
        for dy, dx in ((0, 1), (1, 0)):
            a_ = img[: res - dy, : res - dx].ravel(); b_ = img[dy:, dx:].ravel()
            ok = clean[a_] & clean[b_]
            pp.append(a_[ok]); qq.append(b_[ok])
        p_, q_ = np.concatenate(pp), np.concatenate(qq)
        same_part = o4[p_] == o4[q_]
        same_unit = o16[p_] == o16[q_]
        dist = np.linalg.norm(P[p_] - P[q_], axis=1)
        fpt = float(np.median(dist[same_unit]))
        footprints.append(fpt)
        geo = (dist <= 3 * fpt) & ((N[p_] * N[q_]).sum(1) >= COS20)
        visd = vis[p_] != vis[q_]
        cls = np.full(len(p_), "", object)
        cls[~same_part] = "cross_part"
        cls[same_part & ~geo] = "geometric_edge"
        cls[same_part & geo & same_unit & ~visd] = "interior"
        cls[same_part & geo & ~same_unit & ~visd] = "seam"
        cls[same_part & geo & ~same_unit & visd] = "seam_visdiff"
        dref = np.abs(Lref[p_] - Lref[q_])
        pair_store["dref"].append(dref); pair_store["cls"].append(cls)
        for a in arms:
            dp = np.abs(ldr(preds[a])[p_] - ldr(preds[a])[q_])
            per_arm_pairs[a]["J"].append(np.abs(dp - dref)); per_arm_pairs[a]["ep"].append(E[a][p_]); per_arm_pairs[a]["eq"].append(E[a][q_])
        # distance to nearest other same-part unit (in footprints)
        cm = np.flatnonzero(clean & (o16 >= 0))
        dseam = np.full(res * res, np.inf)
        dother = np.full(res * res, np.inf); nbr = -np.ones(res * res, np.int64); dself = np.full(res * res, np.inf)
        for u in range(16):
            sel = cm[o16[cm] == u]
            if not len(sel):
                continue
            dself[sel] = trees[u].query(P[sel])[0]
            for w in range(16):
                if w == u or unit_part[w] != unit_part[u]:
                    continue
                d_ = trees[w].query(P[sel])[0]
                better = d_ < dother[sel]
                dother[sel[better]] = d_[better]; nbr[sel[better]] = w
        dseam[cm] = dother[cm] / fpt
        for a in arms:
            ae = np.abs(E[a])
            for b in range(len(BINS) - 1):
                s = cm[(dseam[cm] >= BINS[b]) & (dseam[cm] < BINS[b + 1])]
                prof[a][b].append(ae[s])
            np.add.at(unit_err[a], o16[cm], ae[cm])
        np.add.at(unit_n, o16[cm], 1)
        # cross-evaluation and fixed-parameter blend in the band (half-width 2 footprints)
        h = 2 * fpt
        band = cm[(nbr[cm] >= 0) & (dother[cm] - dself[cm] < h)]
        for a in cross:
            model, K = arms[a]
            fj = np.zeros((res * res, 3), np.float32)
            for w in np.unique(nbr[band]):
                s = band[nbr[band] == w]
                fj[s] = run_unit(model, int(w), P[s], C[s], Lt[s], N[s], vis[s])
                lo, hi = model.lo[w].cpu().numpy(), model.hi[w].cpu().numpy()
                cross[a]["inside_aabb"].append(((P[s] >= lo) & (P[s] <= hi)).all(1))
            seam_px = np.unique(np.concatenate([p_[cls == "seam"], q_[cls == "seam"]]))
            seam_px = seam_px[np.isin(seam_px, band)]
            fi = preds[a]
            cross[a]["disagree"].append(np.abs(ldr(fi[seam_px]) - ldr(fj[seam_px])))
            cross[a]["own_err"].append(np.abs(ldr(fi[seam_px]) - Lref[seam_px]))
            cross[a]["nbr_err"].append(np.abs(ldr(fj[seam_px]) - Lref[seam_px]))
            s_ = dother - dself
            wj = np.zeros(res * res); wj[band] = np.clip(0.5 - s_[band] / (2 * h), 0, 0.5)
            bl = fi.copy(); bl[band] = (1 - wj[band, None]) * fi[band] + wj[band, None] * fj[band]
            sm = cls == "seam"
            dpb = np.abs(ldr(bl)[p_[sm]] - ldr(bl)[q_[sm]]); dph = np.abs(ldr(fi)[p_[sm]] - ldr(fi)[q_[sm]])
            blend[a]["J"].append(np.abs(dpb - dref[sm])); blend[a]["J_hard"].append(np.abs(dph - dref[sm]))
            blend[a]["band_err_blend"].append(np.abs(ldr(bl)[band] - Lref[band])); blend[a]["band_err_hard"].append(np.abs(ldr(fi)[band] - Lref[band]))
            maps[f"blend_{a}_{v}"] = bl.reshape(res, res, 3)
        maps[f"dseam_{v}"] = np.where(np.isfinite(dseam), dseam, -1).reshape(res, res).astype(np.float32)
        for a in arms:
            maps[f"pred_{a}_{v}"] = preds[a].reshape(res, res, 3)
        maps[f"ref_{v}"] = ref.reshape(res, res, 3); maps[f"owner16_{v}"] = o16.reshape(res, res)
        print("view", v, "footprint", round(fpt, 5), {c: int((cls == c).sum()) for c in acc}, flush=True)

    dref = np.concatenate(pair_store["dref"]); cls = np.concatenate(pair_store["cls"])
    seam = cls == "seam"
    edges = np.quantile(dref[seam], np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    b_seam = np.digitize(dref, edges[1:-1])
    frac_seam = np.bincount(b_seam[seam], minlength=10) / seam.sum()
    rec = {"cfg": args.cfg, "dev_views": DEV_VIEWS, "footprints": footprints, "pair_counts": {c: int((cls == c).sum()) for c in acc},
           "classes": {}, "profile_bins_footprints": [str(b) for b in BINS], "profile": {}, "unit_err": {}, "cross_eval": {}, "posthoc_blend": {}}
    for c in ("seam", "interior", "cross_part", "geometric_edge", "seam_visdiff"):
        sel = cls == c
        if not sel.any():
            continue
        w = np.ones(sel.sum())
        if c == "interior":  # reweight to the seam pairs' |d ref| distribution
            bi = b_seam[sel]; frac_i = np.bincount(bi, minlength=10) / sel.sum()
            w = frac_seam[bi] / np.maximum(frac_i[bi], 1e-12)
        row = {"pairs": int(sel.sum()), "dref_mean": float(np.average(dref[sel], weights=w))}
        for a in arms:
            J = np.concatenate(per_arm_pairs[a]["J"])[sel]; ep = np.concatenate(per_arm_pairs[a]["ep"])[sel]; eq = np.concatenate(per_arm_pairs[a]["eq"])[sel]
            row[a] = {"J": float(np.average(J, weights=w)), "rho": wpearson(ep, eq, w),
                      "abs_err": float(np.average((np.abs(ep) + np.abs(eq)) / 2, weights=w))}
        rec["classes"][c + ("_matched" if c == "interior" else "")] = row
    for a in arms:
        rec["profile"][a] = [{"n": int(sum(len(x) for x in prof[a][b])), "abs_err": float(np.concatenate(prof[a][b]).mean()) if sum(len(x) for x in prof[a][b]) else None}
                             for b in range(len(BINS) - 1)]
        rec["unit_err"][a] = (unit_err[a] / np.maximum(unit_n, 1)).tolist()
    rec["unit_pixels_dev"] = unit_n.astype(int).tolist()
    for a in cross:
        dis = np.concatenate(cross[a]["disagree"]); own = np.concatenate(cross[a]["own_err"]); nb = np.concatenate(cross[a]["nbr_err"])
        rec["cross_eval"][a] = {"seam_pixels": int(len(dis)), "median_disagreement": float(np.median(dis)), "median_own_err": float(np.median(own)),
                                "median_nbr_err": float(np.median(nb)), "mean_disagreement": float(dis.mean()), "mean_own_err": float(own.mean()),
                                "mean_nbr_err": float(nb.mean()), "frac_band_inside_nbr_aabb": float(np.concatenate(cross[a]["inside_aabb"]).mean())}
        rec["posthoc_blend"][a] = {"seam_J_hard": float(np.concatenate(blend[a]["J_hard"]).mean()), "seam_J_blend": float(np.concatenate(blend[a]["J"]).mean()),
                                   "band_err_hard": float(np.concatenate(blend[a]["band_err_hard"]).mean()),
                                   "band_err_blend": float(np.concatenate(blend[a]["band_err_blend"]).mean()), "band_halfwidth_footprints": 2}
    (out / "diag.json").write_text(json.dumps(rec, indent=1))
    np.savez_compressed(out / "maps.npz", **maps)
    print(json.dumps({k: rec[k] for k in ("classes", "cross_eval", "posthoc_blend")}, indent=1))
    for a in arms:
        print(a, "profile", [round(p["abs_err"], 4) if p["abs_err"] else None for p in rec["profile"][a]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
