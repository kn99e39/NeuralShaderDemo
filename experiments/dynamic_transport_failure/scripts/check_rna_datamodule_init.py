"""Read-only RNA data-module initialization check with a completion marker.

Builds the official NeuralSurfaceDataModule from a train config, runs setup,
draws one training batch, and prints CPU_STAGED_READY (device and ops_device
both cpu) or GPU_RESIDENT_READY.  Nothing is trained or written.

Run from the official RNA repository with its environment:
  .venv/bin/python <this file> --config <train.yml> --device cpu --ops-device cpu
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import torch

OFFICIAL_ROOT = pathlib.Path(__file__).resolve().parents[3] / "external" / "relightable-neural-assets"
sys.path.insert(0, str(OFFICIAL_ROOT))

from rna import config, interfaces  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ops-device", default="cpu")
    args = parser.parse_args()

    staged = args.device == "cpu" and args.ops_device == "cpu"
    marker = "CPU_STAGED_READY" if staged else "GPU_RESIDENT_READY"
    start = time.time()
    conf = config.get_neumat_config(str(args.config.resolve()))
    params = dict(conf.data_interface_params)
    params.update(device=args.device, ops_device=args.ops_device)
    print(f"start device={args.device} ops_device={args.ops_device} cuda={torch.cuda.is_available()}", flush=True)

    data = interfaces.NeuralSurfaceDataModule(conf, **params)
    data.setup("fit")
    print(f"train_len={len(data.trainset)} val_len={len(data.valset) if data.valset else 0}", flush=True)
    next(iter(data.train_dataloader()))
    if torch.cuda.is_available():
        print(f"cuda_max_allocated_MiB={torch.cuda.max_memory_allocated() / 2**20:.0f}", flush=True)
    print(f"{marker} elapsed_s={time.time() - start:.1f}", flush=True)


if __name__ == "__main__":
    main()
