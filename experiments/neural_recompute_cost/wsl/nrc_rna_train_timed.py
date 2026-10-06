"""Timed RNA representation rebuild (instrumentation only; runs in RNA's WSL venv).

    cd external/relightable-neural-assets
    .venv/bin/python ../../experiments/neural_recompute_cost/wsl/nrc_rna_train_timed.py \
        --events <events.jsonl> --snapshots <dir> --snapshot-every 5 --project-commit <sha> -- \
        --config <config.yml> --checkpoint_dir <dir> --seed 0

Everything after ``--`` goes unchanged to the official ``scripts/train.py``
(run with runpy from the RNA root, as worklogs 22/23 did).  This wrapper
only adds timing: it wraps the dataset constructor and Trainer.save_checkpoint
with begin/end events and appends one Lightning callback.  It draws no random
numbers and does not touch the model, optimiser, data or schedule; side
effects are ``torch.cuda.synchronize()`` at phase boundaries and a weights-only
checkpoint every ``--snapshot-every`` epochs (and after epoch 0) written to a
separate directory for offline evaluation (charged to ``snapshot_write``).

The project tree's clean state is checked by the Windows-side chain driver
(git on a Windows checkout from WSL reports spurious file-mode changes); the
commit it saw is passed in and recorded.
"""

from __future__ import annotations

import argparse
import functools
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import nrc_records as R  # noqa: E402


def main() -> int:
    created = R.process_create_time()
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("usage: nrc_rna_train_timed.py <wrapper args> -- <scripts/train.py args>")
    cut = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--snapshots", required=True)
    ap.add_argument("--snapshot-every", type=int, default=5)
    ap.add_argument("--project-commit", required=True)
    ap.add_argument("--stage", default="rna_train")
    args = ap.parse_args(argv[:cut])
    train_argv = argv[cut + 1:]

    log = R.EventLog(args.events, args.stage)
    log.emit("process_created", t_created=created)
    poller = R.GpuMemoryPoller()
    poller.start()
    sys.path.insert(1, os.path.abspath("."))  # as scripts/train.py does (run from the RNA root)
    import pytorch_lightning as pl
    import torch

    from rna import datasets

    log.emit("imports_done")
    log.emit("environment", git={"commit": args.project_commit, "checked_by": "chain driver (Windows)"},
             rna_commit=os.popen("git rev-parse HEAD 2>/dev/null").read().strip() or None,
             python=sys.version.split()[0], torch=torch.__version__, pytorch_lightning=pl.__version__,
             cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), train_argv=train_argv,
             wrapper_argv=argv[:cut], cwd=os.getcwd())

    sync = torch.cuda.synchronize
    DS = datasets.NeuralSurfaceDataset
    orig_ds_init = DS.__init__

    @functools.wraps(orig_ds_init)
    def ds_init(self, *a, **k):
        key = Path(str(k.get("datafile", a[0] if a else ""))).name
        log.begin("data_load", key=key)
        try:
            return orig_ds_init(self, *a, **k)
        finally:
            sync()
            log.end("data_load", key=key)

    DS.__init__ = ds_init

    Trainer = pl.Trainer
    orig_save = Trainer.save_checkpoint

    @functools.wraps(orig_save)
    def save_checkpoint(self, filepath, *a, **k):
        key = {"file": Path(str(filepath)).name, "epoch": int(self.current_epoch)}
        log.begin("checkpoint_write", key=key)
        try:
            return orig_save(self, filepath, *a, **k)
        finally:
            log.end("checkpoint_write", key=key)

    Trainer.save_checkpoint = save_checkpoint
    snap_dir = Path(args.snapshots)
    snap_dir.mkdir(parents=True, exist_ok=True)

    class NrcTiming(pl.Callback):
        def on_fit_start(self, trainer, pl_module):
            log.begin("fit")

        # An epoch runs from its on_train_epoch_start to the next one's (or to fit end):
        # Lightning runs ModelCheckpoint's hooks after every other callback, and in
        # Lightning 2.1 it writes the epoch's checkpoints in on_train_epoch_end, i.e.
        # after this callback's on_train_epoch_end.  This keeps those writes inside
        # the epoch they belong to.
        open_epoch = None

        def _close_epoch(self):
            if self.open_epoch is not None:
                log.end("epoch", key=self.open_epoch)
                self.open_epoch = None

        def on_fit_end(self, trainer, pl_module):
            sync()
            self._close_epoch()
            log.end("fit")

        def on_train_epoch_start(self, trainer, pl_module):
            self._close_epoch()
            self.open_epoch = int(trainer.current_epoch)
            log.begin("epoch", key=self.open_epoch)

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            if batch_idx == 0:
                sync()
                log.begin("optimization", key=int(trainer.current_epoch))

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if batch_idx == trainer.num_training_batches - 1:
                sync()
                log.end("optimization", key=int(trainer.current_epoch), steps=batch_idx + 1,
                        global_step=int(trainer.global_step))

        def _val_name(self, trainer):
            return "sanity_validation" if trainer.sanity_checking else "validation"

        def on_validation_start(self, trainer, pl_module):
            sync()
            log.begin(self._val_name(trainer), key=int(trainer.current_epoch))

        def on_validation_end(self, trainer, pl_module):
            sync()
            name = self._val_name(trainer)
            log.end(name, key=int(trainer.current_epoch))
            if name == "validation":
                m = {k: float(v) for k, v in trainer.callback_metrics.items() if hasattr(v, "item") or isinstance(v, float)}
                log.emit("val_metrics", epoch=int(trainer.current_epoch), metrics=m)

        def on_train_epoch_end(self, trainer, pl_module):
            ep = int(trainer.current_epoch)
            if ep == 0 or (ep + 1) % args.snapshot_every == 0:
                path = snap_dir / f"epoch_{ep:03d}.ckpt"
                log.begin("snapshot_write", key=ep)
                # the original (unwrapped) save, so it is not counted as the method's checkpoint_write
                orig_save(trainer, str(path), weights_only=True)
                log.end("snapshot_write", key=ep, file=path.name, global_step=int(trainer.global_step))
            log.emit("torch_memory", epoch=ep, max_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
                     max_reserved_mib=torch.cuda.max_memory_reserved() / 2 ** 20)

    orig_trainer_init = Trainer.__init__

    @functools.wraps(orig_trainer_init)
    def trainer_init(self, *a, callbacks=None, **k):
        callbacks = list(callbacks or []) + [NrcTiming()]
        return orig_trainer_init(self, *a, callbacks=callbacks, **k)

    Trainer.__init__ = trainer_init

    sys.argv = ["scripts/train.py", *train_argv]
    log.begin("launcher")
    runpy.run_path("scripts/train.py", run_name="__main__")
    log.end("launcher")
    mem = poller.halt()
    mem.update({"torch_max_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
                "torch_max_reserved_mib": torch.cuda.max_memory_reserved() / 2 ** 20,
                "host_peak_rss_mb": R.host_peak_memory_mb()})
    log.emit("memory", **mem)
    log.emit("process_done")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
