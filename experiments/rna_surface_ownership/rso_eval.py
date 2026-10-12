"""Evaluate RNA ownership arms on the validation H5 (runs in the RNA venv).

    python rso_eval.py --config <arm.yml> --arms <name=dir> ... --partitions <dir> --out <dir>

For every arm (A: upstream model, dir/state.pt from rso_train --partition none; owned arms:
dir/state.pt + dir/specs.json) predicts every validation view and reports:
  * RNA's own val PSNR (rna tonemap x/(1+x), gamma 2.2; all masked pixels of all views)
  * display MAE (project transform clamp(x^(1/2.2))) overall, per part, per P16 unit
  * boundary vs interior: pixels whose 4-neighbourhood contains a different P16 owner (the finest
    partition) vs the rest -- evaluated identically for every arm, so seams are comparable
  * inference seconds per view
Writes <out>/eval.json and per-arm prediction images for views 0, 10, 20, 30 (npz).
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--partitions", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import h5py
    import numpy as np
    import torch
    from rna import config, models
    from rso_partition import PARTS, Owner

    conf = config.get_neumat_config(config_filename=str(Path(args.config).resolve()))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    f = h5py.File(conf.validation_dataset, "r")
    res = int(f.attrs["resolution"][0])
    aabb = (np.array(f.attrs["aabb_min"], np.float64), np.array(f.attrs["aabb_max"], np.float64))
    n_views = f["color"].shape[0]
    own16 = Owner(Path(args.partitions) / "partition_P16.npz")
    own4 = Owner(Path(args.partitions) / "partition_P4.npz")
    dev = "cuda"

    def view(i):
        g = lambda k: np.nan_to_num(f[k][i].astype(np.float32))
        alpha = g("alpha")[:, 0] > 0
        pos = g("position")
        return {"mask": alpha, "pos": pos, "cam": g("camera_dir"), "light": g("light_dir"), "normal": g("normal"),
                "vis": g("diffuse_direct")[:, 0], "color": g("color")}

    views = [view(i) for i in range(n_views)]
    owners16, owners4, boundary = [], [], []
    for v in views:
        lab = -np.ones(res * res, np.int64)
        lab[v["mask"]] = own16(v["pos"][v["mask"]])[0]
        owners16.append(lab)
        lab4 = -np.ones(res * res, np.int64)
        lab4[v["mask"]] = own4(v["pos"][v["mask"]])[0]
        owners4.append(lab4)
        img = lab.reshape(res, res)
        b = np.zeros((res, res), bool)
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            sh = np.roll(img, (dy, dx), (0, 1))
            b |= (sh != img) & (sh >= 0) & (img >= 0)
        boundary.append(b.reshape(-1))

    def tonemap_rna(x):
        x = np.clip(x, 0, None)
        return (x / (1 + x)) ** (1 / 2.2)

    def disp(x):
        return np.clip(np.clip(x, 0, None) ** (1 / 2.2), 0, 1)

    record = {"config": args.config, "validation_dataset": conf.validation_dataset, "views": n_views, "arms": {}}
    for spec in args.arms:
        name, d = spec.split("=", 1)
        d = Path(d)
        sd = torch.load(d / "state.pt", map_location=dev)
        if (d / "specs.json").exists():
            specs = [{"params": s["params"], "lo": np.array(s["lo"]), "hi": np.array(s["hi"])} for s in json.loads((d / "specs.json").read_text())]
            model = M.OwnedTriplaneSet(specs, aabb).to(dev)
            model.load_state_dict(sd)
            owned = True
        else:
            model = models.NeuralSurfaceTriPlane(**dict(conf.model_params)).to(dev)
            model.load_state_dict(sd)
            owned = False
        model.eval()
        model.set_iteration_index(10 ** 9)
        sq = {"rna": 0.0}
        n_all = 0
        abs_err = []
        t_inf = 0.0
        keep = {}
        with torch.no_grad():
            for i, v in enumerate(views):
                m = v["mask"]
                pos = (v["pos"][m] - aabb[0]) / (aabb[1] - aabb[0])
                if owned:
                    o = own16(v["pos"][m])[0] if len(specs) == 16 else (own4(v["pos"][m])[0] if len(specs) == 4 else np.zeros(m.sum(), np.int64))
                    pos = np.concatenate([pos, o[:, None]], 1)
                T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev).unsqueeze(0)
                torch.cuda.synchronize(); t0 = time.perf_counter()
                pred = model(T(pos), T(v["cam"][m]), T(v["light"][m]), T(v["normal"][m])).squeeze(0)
                torch.cuda.synchronize(); t_inf += time.perf_counter() - t0
                vis = torch.tensor(v["vis"][m], device=dev) > 0
                rgb = torch.where(vis[:, None], pred[:, :3], pred[:, 3:]).cpu().numpy()
                gt = v["color"][m]
                sq["rna"] += float(((tonemap_rna(rgb) - tonemap_rna(gt)) ** 2).sum())
                n_all += rgb.size
                e = np.zeros(res * res, np.float32)
                e[m] = np.abs(disp(rgb) - disp(gt)).mean(1)
                abs_err.append(e)
                if i in (0, 10, 20, 30):
                    full = np.zeros((res * res, 3), np.float32); full[m] = rgb
                    gtf = np.zeros((res * res, 3), np.float32); gtf[m] = gt
                    keep[f"pred_{i}"] = full.reshape(res, res, 3); keep[f"gt_{i}"] = gtf.reshape(res, res, 3)
        E = np.concatenate(abs_err); O16 = np.concatenate(owners16); O4 = np.concatenate(owners4); B = np.concatenate(boundary)
        valid = O16 >= 0
        row = {"owned": owned, "units": (len(specs) if owned else 1), "val_psnr_rna": float(-10 * np.log10(sq["rna"] / n_all)),
               "display_mae": float(E[valid].mean()),
               "per_part_display_mae": {p: float(E[valid & (O4 == i)].mean()) for i, p in enumerate(PARTS)},
               "per_p16_unit_display_mae": [float(E[O16 == u].mean()) if (O16 == u).any() else None for u in range(16)],
               "p16_unit_pixels": [int((O16 == u).sum()) for u in range(16)],
               "boundary_display_mae": float(E[valid & B].mean()), "interior_display_mae": float(E[valid & ~B].mean()),
               "boundary_pixels": int((valid & B).sum()), "inference_s_per_view": t_inf / len(views),
               "parameters": int(sum(p.numel() for p in model.parameters()))}
        record["arms"][name] = row
        np.savez_compressed(out / f"pred_{name}.npz", **keep)
        print(name, json.dumps({k: row[k] for k in ("val_psnr_rna", "display_mae", "boundary_display_mae", "interior_display_mae", "parameters")}), flush=True)
    (out / "eval.json").write_text(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
