"""Train an RNA ownership arm (runs in the RNA venv; mirrors upstream scripts/train.py).

    python rso_train.py --config <arm.yml> --partition <partition_P*.npz|none> --out <dir> [--seed 0]
        [--matched 1] [--epochs N] [--init <state.pt>] [--train-units 3 ...]

--partition none : the upstream RNA model unchanged (NeuralSurfaceTriplaneModule), arm A.
otherwise        : OwnedSurfaceTriplaneModule over that partition (arms P1/P4/P16).
--init + --train-units : ownership test: load a trained owned model, re-initialise the listed
                   units, freeze every other unit (requires_grad False), train only the listed ones.
Writes <out>/state.pt (final weights), <out>/specs.json, <out>/train.json (timing, counts, digests).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rso_model as M  # noqa: E402  (puts the RNA repo on sys.path)


def digest(module) -> dict:
    out = {}
    for n, p in module.state_dict().items():
        out[n] = hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--partition", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--matched", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--init", default=None)
    ap.add_argument("--train-units", type=int, nargs="*", default=None)
    args = ap.parse_args()
    import numpy as np
    import pytorch_lightning as pl
    import torch
    from rna import config, interfaces
    from rso_partition import Owner

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)  # upstream validation writes debug EXRs into the working directory
    conf = config.get_neumat_config(config_filename=str(Path(args.config).resolve()))
    if args.epochs:
        conf.num_epochs = args.epochs
    pl.seed_everything(args.seed)
    rec = {"args": vars(args), "torch": torch.__version__, "pl": pl.__version__, "gpu": torch.cuda.get_device_name(0)}
    t_start = time.time()
    if args.partition == "none":
        model = interfaces.NeuralSurfaceTriplaneModule(conf, **conf.interface_params)
        data = interfaces.NeuralSurfaceDataModule(conf, **conf.data_interface_params)
        specs = None
    else:
        import h5py

        with h5py.File(conf.training_dataset, "r") as f:
            aabb = (np.array(f.attrs["aabb_min"], np.float64), np.array(f.attrs["aabb_max"], np.float64))
        from omegaconf import OmegaConf

        # plain containers: with K=1 or --matched 0 the params pass through unchanged (ListConfig is not JSON-serialisable)
        base_params = OmegaConf.to_container(conf.model_params, resolve=True)
        specs = M.unit_specs(args.partition, base_params, aabb, matched=bool(args.matched))
        owner = Owner(args.partition)
        model = M.OwnedSurfaceTriplaneModule(conf, specs=specs, dataset_aabb=aabb, **conf.interface_params)
        data = M.OwnedSurfaceDataModule(conf, owner=owner, **conf.data_interface_params)
        (out / "specs.json").write_text(json.dumps([{"params": s["params"], "lo": s["lo"].tolist(), "hi": s["hi"].tolist()} for s in specs], indent=1))
        if args.init:
            sd = torch.load(args.init, map_location="cpu")
            model.model.load_state_dict(sd)
            init_digest = digest(model.model)
            for k in args.train_units:
                fresh = M.rmodels.NeuralSurfaceTriPlane(**specs[k]["params"])
                model.model.units[k].load_state_dict(fresh.state_dict())
            for k, u in enumerate(model.model.units):
                u.requires_grad_(k in args.train_units)
            rec["init"] = {"from": args.init, "reinitialised_units": args.train_units}
            rec["init_digest"] = init_digest
    n_params = sum(p.numel() for p in model.model.parameters())
    rec["parameters_total"] = n_params
    if specs is not None:
        rec["parameters_per_unit"] = [sum(p.numel() for p in u.parameters()) for u in model.model.units]
        rec["unit_resolution"] = [s["params"]["resolution"][0] for s in specs]
        rec["unit_mlp_width"] = [s["params"]["mlp_hidden_layer_neurons"] for s in specs]

    class Timing(pl.Callback):
        def __init__(self):
            self.epochs = []
            self.t0 = None

        def on_train_epoch_start(self, trainer, pl_module):
            self.t0 = time.time()

        def on_train_epoch_end(self, trainer, pl_module):
            m = {k: float(v) for k, v in trainer.callback_metrics.items() if hasattr(v, "item")}
            self.epochs.append({"epoch": trainer.current_epoch, "seconds": time.time() - self.t0, **m})

    tcb = Timing()
    ckpt = pl.callbacks.ModelCheckpoint(dirpath=str(out / "checkpoints"), save_last=True, save_top_k=1,
                                        monitor="val_psnr", filename="{epoch}-{val_psnr:.2f}dB", mode="max")
    logger = pl.loggers.TensorBoardLogger(str(out / "tb"), name=conf.name)
    trainer = pl.Trainer(logger=logger, default_root_dir=str(out), accelerator="gpu", max_epochs=conf.num_epochs,
                         devices=[0], callbacks=[ckpt, tcb], enable_progress_bar=False)
    t_fit = time.time()
    trainer.fit(model, data)  # Lightning calls data.setup() (H5 ingest, owner assignment) inside fit
    rec["seconds"] = {"total": time.time() - t_start, "fit_including_data_setup": time.time() - t_fit}
    rec["epochs"] = tcb.epochs
    rec["best_ckpt"] = ckpt.best_model_path
    rec["best_val_psnr"] = float(ckpt.best_model_score) if ckpt.best_model_score is not None else None
    torch.save(model.model.state_dict(), out / "state.pt")
    rec["final_digest"] = digest(model.model)
    if specs is not None:
        ds = data.trainset
        rec["owner_pixel_counts_train"] = ds.owner_counts.tolist()
        rec["owner_max_distance_train"] = ds.owner_max_dist
    if args.init:
        changed = sorted({n.split(".")[1] for n in rec["final_digest"] if rec["final_digest"][n] != rec["init_digest"][n] and n.startswith("units.")}, key=int)
        rec["units_changed"] = [int(c) for c in changed]
        rec["frozen_units_bit_identical"] = all(int(n.split(".")[1]) in args.train_units or rec["final_digest"][n] == rec["init_digest"][n]
                                                for n in rec["final_digest"] if n.startswith("units."))
    rec["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    (out / "train.json").write_text(json.dumps(rec, indent=1, default=str))
    print("done", json.dumps({k: rec[k] for k in ("parameters_total", "best_val_psnr", "seconds")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
