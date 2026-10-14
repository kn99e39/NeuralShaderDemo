"""Worklog-33 analysis of controls C (reference noise / convergence) and D1 (train/val gap
factorial) for the original RNA checkpoint A (RNA venv, GPU).

    python rbq_analyze.py --cfg C1|C2 --model-config <rna.yml> --weights <state.pt> \
        --train-h5 <h5|none> --val-h5 <h5> --ref-dir <dir> --fact-dir <dir> --out <json>

Metric conventions (identical to upstream RNA / worklog 32):
  ldr(x)   = (max(x,0) / (1 + max(x,0)))^(1/2.2)               upstream utils.imageops.tonemap
  psnr     = -10 log10(mean squared ldr difference) over masked pixels; "pooled" pools all views,
             "view-mean" averages per-view PSNRs (= Lightning val_psnr)
  loss     = upstream visibility-weighted log-L2: 10 * MSE(log1p(p)*(1-v), log1p(g)*(1-v)) + MSE(log1p(p)*v, log1p(g)*v)
  logL1    = mean |log1p(p) - log1p(g)|
  strata   = lit (diffuse_direct > 0) / shadowed (= 0) pixels
Control C (per validation view 0, 10, 20, 30): GT = stored H5 view; rep = same-spp independent
render; ref = 8x-spp render. Noise share of A's error = MSE(GT, ref) / MSE(A, GT).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(1, str(ROOT / "external" / "relightable-neural-assets"))

REF_VIEWS = (0, 10, 20, 30)
TRAIN_VIEWS = (0, 50, 100, 150)


def ldr(x):
    x = np.clip(x, 0, None)
    return (x / (1 + x)) ** (1 / 2.2)


def mse(a, b):
    return float(((ldr(a) - ldr(b)) ** 2).mean())


def psnr(m):
    return float(-10 * np.log10(m + 1e-12))


def loss(p, g, v):
    lp, lg = np.log1p(np.clip(p, -0.99, None)), np.log1p(np.clip(g, 0, None))
    v = v[:, None]
    return float(10 * (((lp - lg) * (1 - v)) ** 2).mean() + (((lp - lg) * v) ** 2).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--train-h5", required=True)
    ap.add_argument("--val-h5", required=True)
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--fact-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import h5py
    import torch
    from rna import config, models

    conf = config.get_neumat_config(config_filename=str(Path(args.model_config).resolve()))
    fv = h5py.File(args.val_h5, "r")
    aabb = (np.array(fv.attrs["aabb_min"], np.float64), np.array(fv.attrs["aabb_max"], np.float64))
    model = models.NeuralSurfaceTriPlane(**dict(conf.model_params)).cuda()
    model.load_state_dict(torch.load(args.weights, map_location="cuda"))
    model.eval(); model.set_iteration_index(10 ** 9)

    def h5view(f, i):
        g = lambda k: np.nan_to_num(f[k][i].astype(np.float32))
        return {k: g(k) for k in ("color", "alpha", "position", "camera_dir", "light_dir", "normal", "diffuse_direct")}

    def npzview(p):
        z = np.load(p)
        return {k: np.nan_to_num(z[k].astype(np.float32)).reshape(-1, z[k].shape[-1] if z[k].ndim > 1 else 1)
                for k in ("color", "alpha", "position", "camera_dir", "light_dir", "normal", "diffuse_direct")}

    def predict(v):
        m = v["alpha"][:, 0] > 0
        pos = (v["position"][m] - aabb[0]) / (aabb[1] - aabb[0])
        T = lambda a: torch.tensor(a, dtype=torch.float32, device="cuda").unsqueeze(0)
        with torch.no_grad():
            pr = model(T(pos), T(v["camera_dir"][m]), T(v["light_dir"][m]), T(v["normal"][m])).squeeze(0)
        vis = v["diffuse_direct"][m][:, 0] > 0
        rgb = torch.where(torch.tensor(vis, device="cuda")[:, None], pr[:, :3], pr[:, 3:]).cpu().numpy()
        return m, rgb, vis.astype(np.float32)

    def summarize(views):
        """views: list of (pred, gt, vis); pooled + view-mean + strata + loss + logL1."""
        ms = [mse(p, g) for p, g, _ in views]; n = [len(p) for p, _, _ in views]
        P = np.concatenate([p for p, _, _ in views]); G = np.concatenate([g for _, g, _ in views]); V = np.concatenate([v for _, _, v in views])
        lit, sh = V > 0, V == 0
        return {"pooled_psnr": psnr(float(np.dot(ms, n) / np.sum(n))), "view_mean_psnr": float(np.mean([psnr(x) for x in ms])),
                "per_view_psnr": [psnr(x) for x in ms], "loss": loss(P, G, V), "logl1": float(np.abs(np.log1p(np.clip(P, 0, None)) - np.log1p(np.clip(G, 0, None))).mean()),
                "lit_psnr": psnr(mse(P[lit], G[lit])) if lit.any() else None, "shadow_psnr": psnr(mse(P[sh], G[sh])) if sh.any() else None,
                "shadow_fraction": float(sh.mean()), "pixels": int(len(P))}

    rec = {"cfg": args.cfg, "weights": args.weights, "val_h5": args.val_h5, "definitions": __doc__}

    # ---- control C ------------------------------------------------------------------------
    rd = Path(args.ref_dir)
    C = {"views": {}}
    agg = {k: [] for k in ("A_gt", "A_ref", "gt_ref", "rep_ref", "gt_rep")}
    tail = {"A_gt_sq": [], "ref_lum": [], "vis": [], "gt_ref_sq": []}
    for v in REF_VIEWS:
        if not (rd / f"{args.cfg}_view{v}_ref.npz").exists():
            continue
        view = h5view(fv, v)
        m, pred, vis = predict(view)
        gt = view["color"][m]
        rep = np.load(rd / f"{args.cfg}_view{v}_rep.npz")["color"][m]
        ref = np.load(rd / f"{args.cfg}_view{v}_ref.npz")["color"][m]
        e = {"A_gt": mse(pred, gt), "A_ref": mse(pred, ref), "gt_ref": mse(gt, ref), "rep_ref": mse(rep, ref), "gt_rep": mse(gt, rep)}
        for k in agg:
            agg[k].append((e[k], len(gt)))
        row = {k + "_psnr": psnr(x) for k, x in e.items()}
        row["noise_share_of_A_error"] = e["gt_ref"] / e["A_gt"]
        row["A_logl1_gt"] = float(np.abs(np.log1p(np.clip(pred, 0, None)) - np.log1p(gt)).mean())
        row["A_logl1_ref"] = float(np.abs(np.log1p(np.clip(pred, 0, None)) - np.log1p(np.clip(ref, 0, None))).mean())
        row["gt_logl1_ref"] = float(np.abs(np.log1p(gt) - np.log1p(np.clip(ref, 0, None))).mean())
        row["gt_rel_std_lit"] = float(np.sqrt(((gt - rep) ** 2 / 2).mean(1)[vis > 0].mean()) / max(ref[vis > 0].mean(), 1e-9)) if (vis > 0).any() else None
        row["gt_max"] = float(gt.max()); row["ref_max"] = float(ref.max())
        if (rd / f"{args.cfg}_view{v}_orig.npz").exists():
            orig = np.load(rd / f"{args.cfg}_view{v}_orig.npz")["color"][m]
            o16 = orig.astype(np.float16).astype(np.float32)
            row["replay_max_abs_diff_vs_h5_after_fp16"] = float(np.abs(o16 - gt).max())
            row["replay_psnr_vs_h5"] = psnr(mse(orig, gt))
        C["views"][v] = row
        tail["A_gt_sq"].append(((ldr(pred) - ldr(gt)) ** 2).mean(1)); tail["gt_ref_sq"].append(((ldr(gt) - ldr(ref)) ** 2).mean(1))
        tail["ref_lum"].append(ref.mean(1)); tail["vis"].append(vis)
    if C["views"]:
        C["pooled"] = {k + "_psnr": psnr(sum(a * b for a, b in x) / sum(b for _, b in x)) for k, x in agg.items()}
        C["pooled"]["noise_share_of_A_error"] = (sum(a * b for a, b in agg["gt_ref"]) / sum(a * b for a, b in agg["A_gt"]))
        sq = np.concatenate(tail["A_gt_sq"]); nsq = np.concatenate(tail["gt_ref_sq"]); lum = np.concatenate(tail["ref_lum"]); vv = np.concatenate(tail["vis"])
        srt = np.sort(sq)[::-1]
        hi = lum >= np.quantile(lum, 0.99)
        C["tails"] = {"share_of_A_sq_error_top1pct_pixels": float(srt[: len(srt) // 100].sum() / srt.sum()),
                      "share_of_A_sq_error_top01pct_pixels": float(srt[: len(srt) // 1000].sum() / srt.sum()),
                      "share_of_A_sq_error_in_brightest1pct_ref": float(sq[hi].sum() / sq.sum()),
                      "share_of_A_sq_error_shadowed": float(sq[vv == 0].sum() / sq.sum()), "shadowed_pixel_fraction": float((vv == 0).mean()),
                      "share_of_gt_noise_sq_in_brightest1pct_ref": float(nsq[hi].sum() / nsq.sum()),
                      "A_psnr_vs_gt_excluding_brightest1pct_ref": psnr(float(sq[~hi].mean())),
                      "A_psnr_vs_gt_brightest1pct_ref_only": psnr(float(sq[hi].mean()))}
    rec["control_C"] = C

    # ---- control D1 -----------------------------------------------------------------------
    fd = Path(args.fact_dir)
    D = {}
    cells = {}
    if args.train_h5 != "none":
        ft = h5py.File(args.train_h5, "r")
        cells["TS_train_cam_stored"] = [h5view(ft, i) for i in TRAIN_VIEWS]
    for cell, key in (("TP_train_cam_new_perpixel_lights", "TP"), ("TI_train_cam_perimage_light", "TI"), ("VP_val_cam_perpixel_lights", "VP")):
        files = sorted(fd.glob(f"{args.cfg}_{key}_view*.npz"))
        if files:
            cells[cell] = [npzview(p) for p in files]
    cells["VI_val_cam_stored"] = [h5view(fv, i) for i in REF_VIEWS]
    for cell, vs in cells.items():
        out = []
        for view in vs:
            m, pred, vis = predict(view)
            out.append((pred, view["color"][m], vis))
        D[cell] = summarize(out)
    rec["control_D1"] = D
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, indent=1))
    print(json.dumps(C.get("pooled"), indent=1)); print(json.dumps(C.get("tails"), indent=1))
    for cell, s in D.items():
        print(cell, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if k != "per_view_psnr"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
