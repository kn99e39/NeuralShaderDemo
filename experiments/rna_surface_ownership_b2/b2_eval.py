"""Architecture Batch 2 final evaluation of P16-PU vs unchanged controls (worklog 34; RNA venv, GPU).

    python b2_eval.py --cfg C2 --val <h5> --partitions <dir> --dev-ref-dir <dir> --heldout-ref-dir <dir> \
        --model-config <yml> --arms A=<dir> P16=<dir> P16w=<dir> REINIT=<dir> --h 0.0072 --out <dir>

View sets: heldout (views with an 8x reference in --heldout-ref-dir, never used for design), dev
(0/10/20/30, worklog-33 references), all40 vs the stored dataset GT (noisy, secondary).
Arms: A; P16, P16w, REINIT with their worklog-32 hard selection; the same three under P16-PU
composition (h fixed in protocol/b2_v1.json amendment 1).
Metrics (RNA tonemap ldr, channel mean): pooled PSNR, logL1, hf error (top-10% reference-gradient
pixels per view), per part, seam jump excess J and |e| on clean seam pairs vs |d ref|-matched clean
interior pairs (b2_diag definitions), band (w>0) vs off-band error, inference time.
REINIT regression: REINIT-PU vs P16-PU changed-pixel set vs the predicted influence set of unit 12.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "rna_surface_ownership"))
import b2_compose as C  # noqa: E402
import rso_model as M  # noqa: E402

COS20 = float(np.cos(np.deg2rad(20.0)))
DEV = (0, 10, 20, 30)
U_STAR = 12


def ldr3(x):
    x = np.clip(x, 0, None)
    return (x / (1 + x)) ** (1 / 2.2)


def main() -> int:
    ap = argparse.ArgumentParser()
    for a in ("--cfg", "--val", "--partitions", "--dev-ref-dir", "--model-config", "--out"):
        ap.add_argument(a, required=True)
    ap.add_argument("--heldout-ref-dir", default=None)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--h", type=float, required=True)
    args = ap.parse_args()
    import h5py
    import torch
    from rna import config, models
    from rso_partition import Owner

    dev = "cuda"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    conf = config.get_neumat_config(config_filename=str(Path(args.model_config).resolve()))
    f = h5py.File(args.val, "r")
    res = int(f.attrs["resolution"][0]); nv = f["color"].shape[0]
    aabb = (np.array(f.attrs["aabb_min"], np.float64), np.array(f.attrs["aabb_max"], np.float64))
    own16 = Owner(Path(args.partitions) / "partition_P16.npz"); own4 = Owner(Path(args.partitions) / "partition_P4.npz")
    arms = {}
    for spec in args.arms:
        name, d = spec.split("=", 1)
        sd = torch.load(Path(d) / "state.pt", map_location=dev)
        if (Path(d) / "specs.json").exists():
            specs = [{"params": s["params"], "lo": np.array(s["lo"]), "hi": np.array(s["hi"])} for s in json.loads((Path(d) / "specs.json").read_text())]
            m = M.OwnedTriplaneSet(specs, aabb).to(dev); m.load_state_dict(sd); m.eval(); m.set_iteration_index(10 ** 9)
            arms[name] = ("owned", C.PUComposer.from_partition(m, Path(args.partitions) / "partition_P16.npz", args.h))
        else:
            m = models.NeuralSurfaceTriPlane(**dict(conf.model_params)).to(dev); m.load_state_dict(sd); m.eval(); m.set_iteration_index(10 ** 9)
            arms[name] = ("plain", m)
    T = lambda a: torch.tensor(np.asarray(a, np.float32), device=dev).unsqueeze(0)

    sets = {"dev": [(v, Path(args.dev_ref_dir) / f"{args.cfg}_view{v}_ref.npz") for v in DEV], "all40_vs_dataset_gt": [(v, None) for v in range(nv)]}
    if args.heldout_ref_dir:
        sets["heldout"] = [(v, Path(args.heldout_ref_dir) / f"{args.cfg}_view{v}_ref.npz") for v in range(nv)
                           if v not in DEV and (Path(args.heldout_ref_dir) / f"{args.cfg}_view{v}_ref.npz").exists()]
    names = []
    for a, (kind, _) in arms.items():
        names.append(a)
        if kind == "owned":
            names.append(a + "-PU")
    record = {"cfg": args.cfg, "h": args.h, "sets": {}, "views": {k: [v for v, _ in s] for k, s in sets.items()}}
    reinit = {"changed": 0, "predicted": 0, "mismatch": 0, "own_px": 0, "nbr_band_px": 0, "outside_identical": True,
              "err_P16PU_infl": [], "err_REINITPU_infl": []}
    timing = {n: [] for n in names}
    cache = {}
    for sname, views in sets.items():
        acc = {n: {"sq": 0.0, "n": 0, "logl1": 0.0, "hf": [], "J_seam": [], "e_seam": [], "J_int": [], "e_int": [],
                   "band": [], "offband": []} for n in names}
        dref_seam_all, dref_int_all = [], []
        for v, refp in views:
            if v not in cache:
                g = lambda k: np.nan_to_num(f[k][v].astype(np.float32))
                alpha = g("alpha")[:, 0]; m = alpha > 0
                P, N, Cd, Lt, vis, gt = g("position"), g("normal"), g("camera_dir"), g("light_dir"), g("diffuse_direct")[:, 0] > 0, g("color")
                o16 = -np.ones(res * res, np.int64); o16[m] = own16(P[m])[0]
                o4 = -np.ones(res * res, np.int64); o4[m] = own4(P[m])[0]
                preds = {}
                for a, (kind, obj) in arms.items():
                    full = np.zeros((res * res, 3), np.float32)
                    if kind == "plain":
                        pos = (P[m] - aabb[0]) / (aabb[1] - aabb[0])
                        t0 = time.perf_counter()
                        with torch.no_grad():
                            pr = obj(T(pos), T(Cd[m]), T(Lt[m]), T(N[m])).squeeze(0)
                        rgb = torch.where(torch.tensor(vis[m], device=dev)[:, None], pr[:, :3], pr[:, 3:]).cpu().numpy()
                        timing[a].append(time.perf_counter() - t0)
                        full[m] = rgb; preds[a] = full
                    else:
                        t0 = time.perf_counter()
                        pu, parts = obj(P[m], Cd[m], Lt[m], N[m], vis[m], o16[m], return_parts=True)
                        timing[a + "-PU"].append(time.perf_counter() - t0)
                        hard = np.zeros((res * res, 3), np.float32); hard[m] = parts["hard"]; preds[a] = hard
                        full[m] = pu; preds[a + "-PU"] = full
                        if a == "P16":
                            w = np.zeros(res * res); w[m] = parts["w"]; nb = -np.ones(res * res, np.int64); nb[m] = parts["nbr"]
                cache[v] = dict(m=m, alpha=alpha, P=P, N=N, vis=vis, gt=gt, o16=o16, o4=o4, preds=preds, w=w, nbr=nb)
            c = cache[v]
            m, P, N = c["m"], c["P"], c["N"]
            tgt = np.load(refp)["color"] if refp is not None else c["gt"]
            Lt_ = ldr3(tgt).mean(-1)
            clean = c["alpha"] >= 0.999
            img = np.arange(res * res).reshape(res, res)
            pp, qq = [], []
            for dy, dx in ((0, 1), (1, 0)):
                a_ = img[: res - dy, : res - dx].ravel(); b_ = img[dy:, dx:].ravel()
                ok = clean[a_] & clean[b_]; pp.append(a_[ok]); qq.append(b_[ok])
            p_, q_ = np.concatenate(pp), np.concatenate(qq)
            same_unit = c["o16"][p_] == c["o16"][q_]
            dist = np.linalg.norm(P[p_] - P[q_], axis=1); fpt = np.median(dist[same_unit])
            ok = (c["o4"][p_] == c["o4"][q_]) & (dist <= 3 * fpt) & ((N[p_] * N[q_]).sum(1) >= COS20) & (c["vis"][p_] == c["vis"][q_])
            seam = ok & ~same_unit; inter = ok & same_unit
            dref = np.abs(Lt_[p_] - Lt_[q_])
            dref_seam_all.append(dref[seam]); dref_int_all.append(dref[inter])
            lum = ldr3(tgt).mean(-1).reshape(res, res); mm = m.reshape(res, res); grad = np.zeros_like(lum)
            for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                okg = mm & np.roll(mm, (dy, dx), (0, 1))
                grad = np.maximum(grad, np.where(okg, np.abs(lum - np.roll(lum, (dy, dx), (0, 1))), 0))
            gm = grad.reshape(-1)[m]; hf = np.zeros(res * res, bool); hf[np.flatnonzero(m)[gm >= np.quantile(gm, 0.9)]] = True
            band = m & (c["w"] > 0)
            for n in names:
                pr = c["preds"][n]
                Lp = ldr3(pr).mean(-1)
                e = np.zeros(res * res); e[m] = np.abs(ldr3(pr[m]) - ldr3(tgt[m])).mean(-1)
                acc[n]["sq"] += float(((ldr3(pr[m]) - ldr3(tgt[m])) ** 2).sum()); acc[n]["n"] += int(m.sum()) * 3
                acc[n]["logl1"] += float(np.abs(np.log1p(np.clip(pr[m], 0, None)) - np.log1p(np.clip(tgt[m], 0, None))).sum())
                acc[n]["hf"].append(e[hf])
                dp = np.abs(Lp[p_] - Lp[q_])
                acc[n]["J_seam"].append(np.abs(dp - dref)[seam]); acc[n]["J_int"].append(np.abs(dp - dref)[inter])
                acc[n]["e_seam"].append(((e[p_] + e[q_]) / 2)[seam]); acc[n]["e_int"].append(((e[p_] + e[q_]) / 2)[inter])
                acc[n]["band"].append(e[band]); acc[n]["offband"].append(e[m & ~band])
                acc[n].setdefault("part_sum", np.zeros(4)); acc[n].setdefault("part_n", np.zeros(4))
                np.add.at(acc[n]["part_sum"], c["o4"][m], e[m]); np.add.at(acc[n]["part_n"], c["o4"][m], 1)
            if sname == "heldout" and "REINIT" in arms:
                a_, b_ = c["preds"]["REINIT-PU"][m], c["preds"]["P16-PU"][m]
                changed = (a_ != b_).any(1)
                infl = (c["o16"][m] == U_STAR) | ((c["nbr"][m] == U_STAR) & (c["w"][m] > 0))
                reinit["changed"] += int(changed.sum()); reinit["predicted"] += int(infl.sum())
                reinit["mismatch"] += int((changed & ~infl).sum())
                reinit["outside_identical"] &= bool(np.array_equal(a_[~infl], b_[~infl]))
                reinit["own_px"] += int((c["o16"][m] == U_STAR).sum()); reinit["nbr_band_px"] += int((infl & (c["o16"][m] != U_STAR)).sum())
                tg = tgt[m][infl]
                reinit["err_P16PU_infl"].append(np.abs(ldr3(b_[infl]) - ldr3(tg)).mean(-1)); reinit["err_REINITPU_infl"].append(np.abs(ldr3(a_[infl]) - ldr3(tg)).mean(-1))
        ds = np.concatenate(dref_seam_all); di = np.concatenate(dref_int_all)
        edges = np.quantile(ds, np.linspace(0, 1, 11)); edges[0], edges[-1] = -np.inf, np.inf
        bs, bi = np.digitize(ds, edges[1:-1]), np.digitize(di, edges[1:-1])
        fs, fi = np.bincount(bs, minlength=10) / len(ds), np.bincount(bi, minlength=10) / len(di)
        wi = fs[bi] / np.maximum(fi[bi], 1e-12)
        rows = {}
        for n in names:
            A_ = acc[n]
            rows[n] = {"pooled_psnr": float(-10 * np.log10(A_["sq"] / A_["n"])), "logl1": A_["logl1"] / A_["n"],
                       "hf_abs_err": float(np.concatenate(A_["hf"]).mean()),
                       "seam_J": float(np.concatenate(A_["J_seam"]).mean()), "interior_J_matched": float(np.average(np.concatenate(A_["J_int"]), weights=wi)),
                       "seam_abs_err": float(np.concatenate(A_["e_seam"]).mean()), "interior_abs_err_matched": float(np.average(np.concatenate(A_["e_int"]), weights=wi)),
                       "band_abs_err": float(np.concatenate(A_["band"]).mean()), "offband_abs_err": float(np.concatenate(A_["offband"]).mean()),
                       "per_part_abs_err": (A_["part_sum"] / np.maximum(A_["part_n"], 1)).tolist()}
        record["sets"][sname] = {"views": len(views), "seam_pairs": int(len(ds)), "rows": rows}
        print(sname, json.dumps({n: {k: round(v, 4) for k, v in r.items() if not isinstance(v, list)} for n, r in rows.items()}, indent=0), flush=True)
    if reinit["predicted"]:
        e0 = np.concatenate(reinit.pop("err_P16PU_infl")); e1 = np.concatenate(reinit.pop("err_REINITPU_infl"))
        reinit.update({"changed_set_equals_predicted": reinit["mismatch"] == 0 and reinit["changed"] <= reinit["predicted"],
                       "err_P16PU_on_influence": float(e0.mean()), "err_REINITPU_on_influence": float(e1.mean()), "recovery_ratio": float(e1.mean() / e0.mean())})
        record["reinit_regression_heldout"] = reinit
        print("reinit", json.dumps(reinit), flush=True)
    record["inference_s_per_view"] = {n: (float(np.mean(t)) if t else None) for n, t in timing.items()}
    (out / "eval.json").write_text(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
