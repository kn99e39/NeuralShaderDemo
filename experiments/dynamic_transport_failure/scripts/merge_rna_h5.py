"""Stream compatible official RNA HDF5 shards into one training dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import h5py


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=pathlib.Path, nargs="+", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    args = parser.parse_args()
    inputs = [path.resolve() for path in args.inputs]
    if len(set(inputs)) != len(inputs):
        raise ValueError("input shard paths must be distinct")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(inputs[0], "r") as first:
        keys = tuple(sorted(first.keys()))
        attrs = {key: first.attrs[key] for key in first.attrs.keys()}
        tails = {key: first[key].shape[1:] for key in keys}
        dtypes = {key: first[key].dtype for key in keys}
    counts = []
    for path in inputs:
        with h5py.File(path, "r") as source:
            if tuple(sorted(source.keys())) != keys:
                raise ValueError(f"incompatible dataset keys: {path}")
            if any(source[key].shape[1:] != tails[key] or source[key].dtype != dtypes[key] for key in keys):
                raise ValueError(f"incompatible dataset layout: {path}")
            if list(source.attrs.get("resolution", ())) != list(attrs.get("resolution", ())):
                raise ValueError(f"incompatible resolution: {path}")
            counts.append(source[keys[0]].shape[0])
    with h5py.File(args.output, "w") as target:
        for key, value in attrs.items():
            target.attrs[key] = value
        targets = {
            key: target.create_dataset(key, shape=(sum(counts), *tails[key]), dtype=dtypes[key])
            for key in keys
        }
        offset = 0
        for path, count in zip(inputs, counts):
            with h5py.File(path, "r") as source:
                for index in range(count):
                    for key in keys:
                        targets[key][offset + index] = source[key][index]
            offset += count
    args.manifest.write_text(json.dumps({
        "label": "STATIC DATASET MERGE ONLY",
        "output": str(args.output.resolve()),
        "image_count": sum(counts),
        "inputs": [{"path": str(path), "images": count, "sha256": sha256(path)} for path, count in zip(inputs, counts)],
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
