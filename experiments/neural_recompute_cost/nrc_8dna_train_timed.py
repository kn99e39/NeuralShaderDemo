"""Timed 8DNA representation rebuild on one locked teaset state (instrumentation only).

    windows/run.ps1 ../neural_recompute_cost/nrc_8dna_train_timed.py --events <events.jsonl> \
        --snapshots <dir> --export <released_layout.ckpt> -- \
        --state T3 --protocol protocol/teaset_frozen_locked.json --device 0 --max_epochs 30 --seed 9 \
        --log_path <dir> --experiment_name <name>

Everything after ``--`` is handed unchanged to the historical launcher
``train_8dna_state.py`` (worklog-22 refit), which registers the state's scene
and runs the unmodified upstream ``train.py``.  This wrapper only adds
timing: it wraps a few upstream methods with begin/end events and appends one
Lightning callback.  None of the wrappers draws random numbers or touches the
model, optimiser, data or schedule; the only side effects are
``torch.cuda.synchronize()`` at phase boundaries (so GPU work is charged to
the phase that issued it) and one per-epoch weight snapshot written to a
separate directory for offline evaluation (charged to ``snapshot_write``).

Phases (nrc_records): process_setup, model_init, dataset_init (the
PathSamplingDataset constructor: loads the state's scene into Mitsuba -- this
track's scene preparation -- and allocates the sample buffers), sanity_validation,
per epoch {path_generation, resample, optimization, validation,
checkpoint_write, snapshot_write}, final checkpoint_write, export (the
``load_asset``-readable copy of last.ckpt, render_8dna_refits.as_released_layout).
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "8dna_deformation_replication"))

import nrc_records as R  # noqa: E402


def main() -> int:
    created = R.process_create_time()
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("usage: nrc_8dna_train_timed.py <wrapper args> -- <train_8dna_state.py args>")
    cut = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--snapshots", required=True)
    ap.add_argument("--export", required=True)
    ap.add_argument("--stage", default="8dna_train")
    ap.add_argument("--allow-dirty", action="store_true", help="smoke runs only; never evidence")
    args = ap.parse_args(argv[:cut])
    launcher_argv = argv[cut + 1:]

    log = R.EventLog(args.events, args.stage)
    # process_setup = imports_done.t - t_created (interpreter start-up, imports, runtime init)
    log.emit("process_created", t_created=created)
    import ednalib as L

    git = R.git_state(L.ROOT)
    R.require_clean(git, args.allow_dirty)
    poller = R.GpuMemoryPoller()
    poller.start()
    mi, dr = L.init_upstream()
    import torch
    import lightning
    from lightning.pytorch.callbacks import Callback

    import models.eight_dna as eight_dna
    import utils.dataset as udata
    import render_8dna_refits
    import train_8dna_state

    log.emit("imports_done")
    log.emit("environment", environment=L.environment_record(), git=git, launcher_argv=launcher_argv,
             wrapper_argv=argv[:cut])

    sync = torch.cuda.synchronize

    def timed(event, key_fn=None, pre_sync=False, post_sync=True):
        def deco(fn):
            @functools.wraps(fn)
            def inner(*a, **k):
                key = key_fn(a, k) if key_fn else None
                if pre_sync:
                    sync()
                log.begin(event, key=key)
                try:
                    return fn(*a, **k)
                finally:
                    if post_sync:
                        sync()
                    log.end(event, key=key)
            return inner
        return deco

    PSD = udata.PathSamplingDataset
    counters = {"reload": 0, "resample": 0}

    def _count(name):
        def f(a, k):
            counters[name] += 1
            return counters[name] - 1
        return f

    # The constructor loads the state's scene (geometry, materials, acceleration
    # structure) and the path-sampling integrator and allocates the sample buffers;
    # it is reported as dataset_init and is this track's scene preparation.
    # (Mitsuba caches attribute lookups per variant, so mi.load_dict itself cannot
    # be wrapped without reaching into its private cache.)
    PSD.__init__ = timed("dataset_init", post_sync=False)(PSD.__init__)
    PSD.reload = timed("path_generation", _count("reload"), pre_sync=True)(PSD.reload)
    PSD.resample = timed("resample", _count("resample"), pre_sync=True)(PSD.resample)
    eight_dna.EightDNA.__init__ = timed("model_init", post_sync=False)(eight_dna.EightDNA.__init__)

    Trainer = lightning.Trainer
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

    class NrcTiming(Callback):
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
            from omegaconf import OmegaConf

            ep = int(trainer.current_epoch)
            path = snap_dir / f"epoch_{ep:03d}.ckpt"
            log.begin("snapshot_write", key=ep)
            # load_asset-readable layout (render_8dna_refits.as_released_layout): the same
            # two fields, so the offline evaluation loads it through unchanged upstream code
            hp = OmegaConf.to_container(pl_module.hparams.model, resolve=True)
            sd = {k: v.detach().to("cpu", copy=True) for k, v in pl_module.state_dict().items()}
            torch.save({"state_dict": sd, "hyper_parameters": {"model": hp}}, path)
            log.end("snapshot_write", key=ep, file=path.name, global_step=int(trainer.global_step))
            log.emit("torch_memory", epoch=ep, max_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
                     max_reserved_mib=torch.cuda.max_memory_reserved() / 2 ** 20)

    orig_trainer_init = Trainer.__init__

    @functools.wraps(orig_trainer_init)
    def trainer_init(self, *a, callbacks=None, **k):
        callbacks = list(callbacks or []) + [NrcTiming()]
        return orig_trainer_init(self, *a, callbacks=callbacks, **k)

    Trainer.__init__ = trainer_init

    # --- the historical launcher, unchanged ---------------------------------
    sys.argv = [str(L.EXPERIMENT / "train_8dna_state.py"), *launcher_argv]
    log.begin("launcher")
    rc = train_8dna_state.main()
    log.end("launcher", rc=rc)

    # --- usable artifact: load_asset-readable copy of the schedule's last.ckpt
    la = argparse.ArgumentParser()
    la.add_argument("--log_path", required=True)
    la.add_argument("--experiment_name", required=True)
    known, _ = la.parse_known_args(launcher_argv)
    last = Path(known.log_path) / known.experiment_name / "last.ckpt"
    log.begin("export")
    render_8dna_refits.as_released_layout(last, Path(args.export))
    log.end("export", source=str(last), sha256=L.sha256(last), export_sha256=L.sha256(args.export))
    mem = poller.halt()
    mem.update({"torch_max_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
                "torch_max_reserved_mib": torch.cuda.max_memory_reserved() / 2 ** 20,
                "host_peak_working_set_mb": R.host_peak_memory_mb()})
    log.emit("memory", **mem)
    log.emit("process_done")
    log.close()
    return rc


if __name__ == "__main__":
    rc = main()
    # Outputs are complete here; skip the interpreter teardown, which crashes with
    # an access violation on this Windows setup once DrJit and torch are loaded.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc or 0)
