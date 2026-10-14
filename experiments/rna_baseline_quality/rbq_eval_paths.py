"""Worklog-33 control A: evaluation-path equivalence for one RNA checkpoint (RNA venv, GPU).

    python rbq_eval_paths.py --config <rna.yml> --weights <state.pt | lightning .ckpt> --out <json>

Same weights, same validation H5, three paths:
  U  upstream: rna.interfaces.NeuralSurfaceDataModule + NeuralSurfaceTriplaneModule, the
     validation_step arithmetic re-executed per batch (one view per batch), plus
     pl.Trainer.validate() for the value Lightning itself logs;
  O  offline: worklog-32 rso_eval's own loading (H5 read, alpha mask, AABB normalisation,
     lit/shadow choice from diffuse_direct channel 0) and forward.
Reports per-pixel agreement of U and O predictions, the per-view PSNRs, and the aggregations
  * mean over views of per-view PSNR (= upstream/Lightning val_psnr),
  * pooled PSNR over all masked pixels of all views (= rso_eval val_psnr_rna, the worklog-32 gate),
  * pixel-weighted mean of per-view PSNR,
so the gap between Lightning's and the offline number is decomposed exactly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RNA = ROOT / "external" / "relightable-neural-assets"
sys.path.insert(1, str(RNA))


def load_weights(model, path):
    import torch

    sd = torch.load(path, map_location="cuda", weights_only=False)
    if "state_dict" in sd:  # Lightning checkpoint: module state, model.* prefix
        sd = {k[len("model."):]: v for k, v in sd["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(sd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import h5py
    import numpy as np
    import pytorch_lightning as pl
    import torch
    from rna import config, interfaces, models
    from utils import imageops, ops

    conf = config.get_neumat_config(config_filename=str(Path(args.config).resolve()))
    dev = "cuda"

    # ---- U: upstream data module + module -------------------------------------------------
    dm = interfaces.NeuralSurfaceDataModule(conf, **conf.data_interface_params)
    dm.setup("validate")
    mod = interfaces.NeuralSurfaceTriplaneModule(conf, **conf.interface_params).to(dev)
    load_weights(mod.model, args.weights)
    # validation_step calls model.set_iteration_index(global_step); a fresh Trainer starts at
    # global_step 0, which would re-widen the triplane blur to its initial sigma. Pin the index to
    # the end of the sigma schedule, as at the end of training (250 epochs x 200 steps > 20 k).
    _set = mod.model.set_iteration_index
    mod.model.set_iteration_index = lambda idx: _set(10 ** 9)
    mod.model.set_iteration_index(0)
    mod.eval()
    up_pred, up_psnr, up_n, up_gt = [], [], [], []
    with torch.no_grad():
        for batch in dm.val_dataloader():
            batch = [b.to(dev) if torch.is_tensor(b) else b for b in batch]
            (uvp, cam, light, nrm, vis, gt, res, mask) = batch
            p = mod.forward(uvp, cam, light, nrm)
            rgb = torch.where(vis > 0.0, p[..., :3], p[..., 3:])
            up_psnr.append(float(ops.psnr(imageops.tonemap(rgb), imageops.tonemap(gt))))
            up_pred.append(rgb.reshape(-1, 3).cpu().numpy()); up_gt.append(gt.reshape(-1, 3).cpu().numpy())
            up_n.append(int(rgb.reshape(-1, 3).shape[0]))
    trainer = pl.Trainer(accelerator="gpu", devices=[0], logger=False, enable_progress_bar=False, enable_checkpointing=False)
    lightning_logged = trainer.validate(mod, datamodule=dm, verbose=False)[0]

    # ---- O: worklog-32 rso_eval loading ---------------------------------------------------
    f = h5py.File(conf.validation_dataset, "r")
    aabb = (np.array(f.attrs["aabb_min"], np.float64), np.array(f.attrs["aabb_max"], np.float64))
    model = models.NeuralSurfaceTriPlane(**dict(conf.model_params)).to(dev)
    load_weights(model, args.weights)
    model.eval(); model.set_iteration_index(10 ** 9)
    off_pred, off_gt = [], []
    with torch.no_grad():
        for i in range(f["color"].shape[0]):
            g = lambda k: np.nan_to_num(f[k][i].astype(np.float32))
            m = g("alpha")[:, 0] > 0
            pos = (g("position")[m] - aabb[0]) / (aabb[1] - aabb[0])
            T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev).unsqueeze(0)
            pr = model(T(pos), T(g("camera_dir")[m]), T(g("light_dir")[m]), T(g("normal")[m])).squeeze(0)
            vis = torch.tensor(g("diffuse_direct")[m][:, 0], device=dev) > 0
            off_pred.append(torch.where(vis[:, None], pr[:, :3], pr[:, 3:]).cpu().numpy())
            off_gt.append(g("color")[m])

    def tm(x):
        x = np.clip(x, 0, None)
        return (x / (1 + x)) ** (1 / 2.2)

    def agg(preds, gts):
        mse = np.array([((tm(p) - tm(g)) ** 2).mean() for p, g in zip(preds, gts)])
        n = np.array([p.size for p in preds])
        per_view = -10 * np.log10(mse + 1e-8)
        return {"mean_per_view_psnr": float(per_view.mean()),
                "pooled_psnr": float(-10 * np.log10((mse * n).sum() / n.sum())),
                "pixel_weighted_mean_per_view_psnr": float((per_view * n).sum() / n.sum()),
                "per_view_psnr": per_view.tolist(), "per_view_pixels": (n // 3).tolist()}

    same_count = [int(a.shape[0]) == int(b.shape[0]) for a, b in zip(up_pred, off_pred)]
    diffs = [float(np.abs(a - b).max()) if a.shape == b.shape else None for a, b in zip(up_pred, off_pred)]
    gt_diffs = [float(np.abs(a - b).max()) if a.shape == b.shape else None for a, b in zip(up_gt, off_gt)]
    rec = {"config": args.config, "weights": args.weights, "views": len(up_pred),
           "lightning_validate_logged": {k: float(v) for k, v in lightning_logged.items()},
           "upstream_steps_mean_val_psnr": float(np.mean(up_psnr)),
           "U": agg(up_pred, up_gt), "O": agg(off_pred, off_gt),
           "masked_pixel_counts_equal": all(same_count),
           "max_abs_pred_diff_U_vs_O": max(d for d in diffs if d is not None) if any(d is not None for d in diffs) else None,
           "max_abs_gt_diff_U_vs_O": max(d for d in gt_diffs if d is not None) if any(d is not None for d in gt_diffs) else None}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k not in ("U", "O")}, indent=1))
    print("U", {k: v for k, v in rec["U"].items() if not k.startswith("per_view")})
    print("O", {k: v for k, v in rec["O"].items() if not k.startswith("per_view")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
