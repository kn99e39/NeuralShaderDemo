"""Worklog-35: evaluate released / reproduced Lego checkpoints on a natively generated validation H5
through the UNCHANGED upstream Lightning validation path (RNA venv, GPU).

    cd results/rna_reproduction/work && PYTHONPATH=<upstream> python rr_eval_val.py --config <train yml> \
        --val <val.h5> --ckpt <lightning ckpt> --out <json>

* upstream NeuralSurfaceDataModule + NeuralSurfaceTriplaneModule.load_from_checkpoint (neumat alias,
  rr_run.py) + pl.Trainer.validate -> the logged val_psnr = mean over views of per-view PSNR on
  tonemap(x/(1+x))^(1/2.2) of alpha>0 pixels (the code's definition, = paper Table 1 metric if the
  paper used this code);
* validation_step resets the triplane blur from global_step (0 for a fresh trainer, i.e. sigma 8): the
  index is pinned to the end of the schedule exactly as at the end of training (worklog 33);
* additionally per view: PSNR, pooled PSNR, logL1 (HDR) and hf (top-10% GT-gradient) error, from a
  manual pass over the same DataModule batches.
The DataModule also loads `training_dataset`; for validation-only use the config's training path is
pointed at the validation file (recorded in the output).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import pytorch_lightning as pl
    import rna
    import rna.config
    import torch
    import yaml

    sys.modules.setdefault("neumat", rna); sys.modules.setdefault("neumat.config", rna.config)  # rr_run.py adapter
    from rna import config, interfaces
    from utils import imageops, ops

    y = yaml.safe_load(open(args.config))
    y["training_dataset"] = args.val; y["validation_dataset"] = args.val
    tmp = Path(tempfile.mkdtemp()) / "val_only.yml"; tmp.write_text(yaml.safe_dump(y))
    conf = config.get_neumat_config(config_filename=str(tmp))
    dm = interfaces.NeuralSurfaceDataModule(conf, **conf.data_interface_params); dm.setup("validate")
    mod = interfaces.NeuralSurfaceTriplaneModule.load_from_checkpoint(args.ckpt, map_location="cuda", strict=True)
    _set = mod.model.set_iteration_index
    mod.model.set_iteration_index = lambda idx: _set(10 ** 9)
    mod.model.set_iteration_index(0)
    mod.eval()
    per, sq, n, l1, hfs = [], 0.0, 0, 0.0, []
    with torch.no_grad():
        for batch in dm.val_dataloader():
            batch = [b.cuda() if torch.is_tensor(b) else b for b in batch]
            (uvp, cam, light, nrm, vis, gt, res, mask) = batch
            p = mod.forward(uvp, cam, light, nrm)
            rgb = torch.where(vis > 0.0, p[..., :3], p[..., 3:])
            a, b = imageops.tonemap(rgb), imageops.tonemap(gt)
            per.append(float(ops.psnr(a, b)))
            d = ((a - b) ** 2).cpu().numpy().reshape(-1, 3)
            sq += float(d.sum()); n += d.size
            l1 += float((torch.log1p(rgb.clamp(min=0)) - torch.log1p(gt.clamp(min=0))).abs().sum());
            e = (a - b).abs().mean(-1).reshape(-1).cpu().numpy(); g = b.mean(-1).reshape(-1).cpu().numpy()
            # high-frequency proxy on the masked pixel list: largest |gt - median-neighbour| is not available
            # without the image grid, so use the top-10% GT luminance-deviation pixels from the view mean
            dev = np.abs(g - np.median(g)); hfs.append(e[dev >= np.quantile(dev, 0.9)])
    logged = pl.Trainer(accelerator="gpu", devices=[0], logger=False, enable_progress_bar=False,
                        enable_checkpointing=False).validate(mod, datamodule=dm, verbose=False)[0]
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    best = None
    for k, v in ck.get("callbacks", {}).items():
        if "ModelCheckpoint" in k:
            best = {"best_model_path": str(v.get("best_model_path")), "best_model_score": float(v["best_model_score"]) if v.get("best_model_score") is not None else None}
    rec = {"ckpt": args.ckpt, "val": args.val, "views": len(per), "lightning_logged_val_psnr": float(logged["val_psnr"]),
           "view_mean_psnr": float(np.mean(per)), "pooled_psnr": float(-10 * np.log10(sq / n)), "per_view_psnr": per,
           "logl1": l1 / n, "hf_proxy_abs_ldr": float(np.concatenate(hfs).mean()),
           "params": int(sum(p.numel() for p in mod.model.parameters())), "ckpt_epoch": ck.get("epoch"), "ckpt_global_step": ck.get("global_step"),
           "ckpt_logged_best": best, "note": "training_dataset pointed at the validation file for validation-only use"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "per_view_psnr"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
