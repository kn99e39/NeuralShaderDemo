"""Worklog-32 diagnostics on the trained arms (runs in the RNA venv; no training).

    python rso_diag.py --model-config <C1.yml> --val <val.h5> --partitions <dir> --out <dir> \
        --arms A=<dir> P1=<dir> P4=<dir> P16=<dir> P16w=<dir> REINIT=<dir> [--reinit-ref P16]

Adds the follow-up spec's section-5/6/7 quantities to rso_eval (whose predeclared gate numbers are
NOT recomputed or replaced here). Definitions fixed before any of these were looked at:
  * HDR metrics on linear radiance, all masked pixels of all views:
      relMSE = mean((p - g)^2 / (g^2 + 0.01)), logL1 = mean |log(1 + p) - log(1 + g)| (RNA's training space)
  * high-frequency region: per view, the 10 % of masked pixels with the largest GT display-space
    luminance gradient (max abs 4-neighbour difference); display MAE and logL1 there
  * seams: a masked pixel is on an intra-part seam if a 4-neighbour has a different P16 owner of the
    SAME part (an ownership cut through continuous surface), on a cross-part boundary if the
    neighbour's owner belongs to another part (a geometric edge, present for every arm)
  * seam discontinuity: over 4-neighbour pairs straddling an intra-part seam, mean
    | |dp| - |dg| | in display space (dp, dg = pair differences of prediction / GT); the same pairs
    for every arm, so A (no seams) is the control
  * per P16 unit: pixels, surface area (partition), display MAE, ratio to A
  * output preservation (REINIT vs --reinit-ref): fraction of bit-identical predictions on pixels
    owned by frozen units, and whether u*'s pixels changed
  * model load time (state.pt -> eval-ready module) and inference time per view
Writes <out>/diag.json and <out>/maps.npz (owner / seam maps, GT and per-arm predictions for views
0, 10, 20, 30).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rso_model as M  # noqa: E402

KEEP_VIEWS = (0, 10, 20, 30)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--partitions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--reinit-ref", default="P16")
    args = ap.parse_args()
    import h5py
    import numpy as np
    import torch
    from rna import config, models
    from rso_partition import PARTS, Owner

    conf = config.get_neumat_config(config_filename=str(Path(args.model_config).resolve()))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    f = h5py.File(args.val, "r")
    res = int(f.attrs["resolution"][0])
    aabb = (np.array(f.attrs["aabb_min"], np.float64), np.array(f.attrs["aabb_max"], np.float64))
    n_views = f["color"].shape[0]
    pz = np.load(Path(args.partitions) / "partition_P16.npz")
    unit_part, unit_area = pz["unit_part"], pz["unit_area"]
    own16 = Owner(Path(args.partitions) / "partition_P16.npz")
    own4 = Owner(Path(args.partitions) / "partition_P4.npz")
    dev = "cuda"
    NB = ((0, 1), (1, 0), (0, -1), (-1, 0))

    def disp(x):
        return np.clip(np.clip(x, 0, None) ** (1 / 2.2), 0, 1)

    views = []
    for i in range(n_views):
        g = lambda k: np.nan_to_num(f[k][i].astype(np.float32))
        m = g("alpha")[:, 0] > 0
        v = {"mask": m, "pos": g("position"), "cam": g("camera_dir"), "light": g("light_dir"), "normal": g("normal"),
             "vis": g("diffuse_direct")[:, 0], "color": g("color")}
        lab16 = -np.ones(res * res, np.int64); lab16[m] = own16(v["pos"][m])[0]
        lab4 = -np.ones(res * res, np.int64); lab4[m] = own4(v["pos"][m])[0]
        img = lab16.reshape(res, res)
        part = np.where(img >= 0, unit_part[np.maximum(img, 0)], -1)
        intra = np.zeros((res, res), bool); cross = np.zeros((res, res), bool)
        pairs = []  # (flat index p, flat index q) straddling an intra-part seam, each pair once
        idx = np.arange(res * res).reshape(res, res)
        for dy, dx in NB:
            sh = np.roll(img, (dy, dx), (0, 1)); shp = np.roll(part, (dy, dx), (0, 1))
            both = (sh >= 0) & (img >= 0) & (sh != img)
            intra |= both & (shp == part); cross |= both & (shp != part)
            if (dy, dx) in ((0, 1), (1, 0)):
                sel = both & (shp == part)
                sel[:dy or None, :dx or None] = False  # drop np.roll wrap-around rows/cols
                pairs.append(np.stack([idx[sel], np.roll(idx, (dy, dx), (0, 1))[sel]], 1))
        lum = disp(v["color"]).mean(1).reshape(res, res)
        grad = np.zeros((res, res), np.float32)
        mm = m.reshape(res, res)
        for dy, dx in NB:
            ok = mm & np.roll(mm, (dy, dx), (0, 1))
            grad = np.maximum(grad, np.where(ok, np.abs(lum - np.roll(lum, (dy, dx), (0, 1))), 0))
        gm = grad.reshape(-1)[m]
        hf = np.zeros(res * res, bool); hf[np.flatnonzero(m)[gm >= np.quantile(gm, 0.9)]] = True
        v.update(lab16=lab16, lab4=lab4, intra=intra.reshape(-1), cross=cross.reshape(-1), pairs=np.concatenate(pairs), hf=hf)
        views.append(v)

    record = {"model_config": args.model_config, "validation_dataset": args.val, "views": n_views,
              "definitions": __doc__, "partition_unit_area": unit_area.tolist(), "partition_unit_part": unit_part.tolist(), "arms": {}}
    maps = {}
    for i in KEEP_VIEWS:
        maps[f"owner16_{i}"] = views[i]["lab16"].reshape(res, res); maps[f"owner4_{i}"] = views[i]["lab4"].reshape(res, res)
        maps[f"intra_{i}"] = views[i]["intra"].reshape(res, res); maps[f"cross_{i}"] = views[i]["cross"].reshape(res, res)
        maps[f"hf_{i}"] = views[i]["hf"].reshape(res, res)
        gtf = np.zeros((res * res, 3), np.float32); gtf[views[i]["mask"]] = views[i]["color"][views[i]["mask"]]
        maps[f"gt_{i}"] = gtf.reshape(res, res, 3)
    preds = {}
    for spec in args.arms:
        name, d = spec.split("=", 1)
        d = Path(d)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        sd = torch.load(d / "state.pt", map_location=dev)
        if (d / "specs.json").exists():
            specs = [{"params": s["params"], "lo": np.array(s["lo"]), "hi": np.array(s["hi"])} for s in json.loads((d / "specs.json").read_text())]
            model = M.OwnedTriplaneSet(specs, aabb).to(dev); K = len(specs)
        else:
            model = models.NeuralSurfaceTriPlane(**dict(conf.model_params)).to(dev); K = 0
        model.load_state_dict(sd); model.eval(); model.set_iteration_index(10 ** 9)
        torch.cuda.synchronize(); t_load = time.perf_counter() - t0
        acc = {k: [] for k in ("relmse", "logl1", "dmae", "hf_dmae", "hf_logl1", "intra_dmae", "cross_dmae", "interior_dmae", "seam_disc")}
        per_unit = np.zeros(16); per_unit_n = np.zeros(16)
        allpred = []
        t_inf = 0.0
        with torch.no_grad():
            for i, v in enumerate(views):
                m = v["mask"]
                pos = (v["pos"][m] - aabb[0]) / (aabb[1] - aabb[0])
                if K:
                    o = {16: v["lab16"][m], 4: v["lab4"][m], 1: np.zeros(m.sum(), np.int64)}[K]
                    pos = np.concatenate([pos, o[:, None]], 1)
                T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev).unsqueeze(0)
                torch.cuda.synchronize(); t1 = time.perf_counter()
                pr = model(T(pos), T(v["cam"][m]), T(v["light"][m]), T(v["normal"][m])).squeeze(0)
                torch.cuda.synchronize(); t_inf += time.perf_counter() - t1
                vis = torch.tensor(v["vis"][m], device=dev) > 0
                rgb = torch.where(vis[:, None], pr[:, :3], pr[:, 3:]).cpu().numpy()
                full = np.zeros((res * res, 3), np.float32); full[m] = rgb
                allpred.append(full)
                gt = v["color"]
                p, g = np.clip(full[m], 0, None), np.clip(gt[m], 0, None)
                acc["relmse"].append(((p - g) ** 2 / (g ** 2 + 0.01)).mean(1))
                acc["logl1"].append(np.abs(np.log1p(p) - np.log1p(g)).mean(1))
                e = np.zeros(res * res, np.float32); e[m] = np.abs(disp(full[m]) - disp(gt[m])).mean(1)
                ll = np.zeros(res * res, np.float32); ll[m] = acc["logl1"][-1]
                acc["dmae"].append(e[m]); acc["hf_dmae"].append(e[v["hf"]]); acc["hf_logl1"].append(ll[v["hf"]])
                acc["intra_dmae"].append(e[v["intra"]]); acc["cross_dmae"].append(e[v["cross"] & ~v["intra"]])
                acc["interior_dmae"].append(e[m & ~v["intra"] & ~v["cross"]])
                dp, dg = disp(full), disp(gt)
                pq = v["pairs"]
                acc["seam_disc"].append(np.abs(np.abs(dp[pq[:, 0]] - dp[pq[:, 1]]).mean(1) - np.abs(dg[pq[:, 0]] - dg[pq[:, 1]]).mean(1)))
                lab = v["lab16"]
                np.add.at(per_unit, lab[m], e[m]); np.add.at(per_unit_n, lab[m], 1)
        preds[name] = allpred
        row = {k: float(np.concatenate(a).mean()) for k, a in acc.items()}
        row.update(seam_pairs=int(sum(len(v["pairs"]) for v in views)), intra_pixels=int(sum(v["intra"].sum() for v in views)),
                   units=max(K, 1), parameters=int(sum(p.numel() for p in model.parameters())),
                   per_unit_params=[int(sum(p.numel() for p in u.parameters())) for u in model.units] if K else None,
                   load_s=t_load, inference_s_per_view=t_inf / n_views,
                   per_p16_unit_dmae=(per_unit / np.maximum(per_unit_n, 1)).tolist(), per_p16_unit_pixels=per_unit_n.astype(int).tolist())
        record["arms"][name] = row
        for i in KEEP_VIEWS:
            maps[f"pred_{name}_{i}"] = allpred[i].reshape(res, res, 3)
        print(name, json.dumps({k: row[k] for k in ("relmse", "logl1", "dmae", "hf_dmae", "intra_dmae", "seam_disc")}), flush=True)
        del model; torch.cuda.empty_cache()

    if "A" in record["arms"]:
        a = np.array(record["arms"]["A"]["per_p16_unit_dmae"])
        for name, row in record["arms"].items():
            r = np.array(row["per_p16_unit_dmae"]) / a
            row["per_p16_unit_ratio_to_A"] = r.tolist()
            rank = lambda x: np.argsort(np.argsort(x))
            row["spearman_unit_area_vs_ratio"] = float(np.corrcoef(rank(unit_area), rank(r))[0, 1])
    if "REINIT" in preds and args.reinit_ref in preds:
        u = json.loads((Path(dict(s.split("=", 1) for s in args.arms)["REINIT"]) / "train.json").read_text())["init"]["reinitialised_units"]
        same_frozen = n_frozen = 0; changed_u = 0; n_u = 0
        for v, pr, rf in zip(views, preds["REINIT"], preds[args.reinit_ref]):
            m = v["mask"]; ust = np.isin(v["lab16"], u) & m; fr = m & ~ust
            same_frozen += int((pr[fr] == rf[fr]).all(1).sum()); n_frozen += int(fr.sum())
            changed_u += int((pr[ust] != rf[ust]).any(1).sum()); n_u += int(ust.sum())
        record["output_preservation"] = {"reference": args.reinit_ref, "reinitialised_units": u,
                                         "frozen_pixels": n_frozen, "frozen_pixels_bit_identical": same_frozen,
                                         "fraction_bit_identical": same_frozen / max(n_frozen, 1),
                                         "u_star_pixels": n_u, "u_star_pixels_changed": changed_u}
        print("output_preservation", json.dumps(record["output_preservation"]), flush=True)
    (out / "diag.json").write_text(json.dumps(record, indent=1))
    np.savez_compressed(out / "maps.npz", **maps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
