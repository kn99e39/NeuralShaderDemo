"""Seed the official RNA generator without modifying its source tree."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import runpy
import sys

import numpy as np


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=pathlib.Path, required=True)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--record", type=pathlib.Path, required=True)
    args = parser.parse_args()
    official_root = args.official_root.resolve()
    config = args.config.resolve()
    generator = official_root / "scripts" / "generate_dataset.py"
    if not generator.is_file() or not config.is_file():
        raise FileNotFoundError("official generator or config is absent")
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps({
        "label": "RNA DATASET GENERATION LAUNCH",
        "official_generator": str(generator),
        "official_generator_sha256": sha256(generator),
        "config": str(config),
        "config_sha256": sha256(config),
        "mode": "rna",
        "numpy_seed": args.seed,
        "camera_sampling": "official random hemisphere camera; matrices exported by generator beside HDF5",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }, indent=2), encoding="utf-8")
    np.random.seed(args.seed)
    sys.argv = [str(generator), "--mode", "rna", "--config", str(config)]
    runpy.run_path(str(generator), run_name="__main__")


if __name__ == "__main__":
    main()
