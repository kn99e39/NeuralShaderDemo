"""Validate an official RNA HDF5 dataset without changing it."""

from __future__ import annotations

import argparse
import json
import pathlib

import h5py
import numpy as np


REQUIRED_DATASETS = (
    "alpha", "camera_dir", "color", "diffuse_direct", "light_dir",
    "normal", "position", "tangent", "uv",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--expected-images", type=int)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    dataset_path = args.dataset.resolve()
    with h5py.File(dataset_path, "r") as handle:
        missing = [name for name in REQUIRED_DATASETS if name not in handle]
        if missing:
            raise ValueError(f"missing required datasets: {missing}")
        image_count = int(handle["color"].shape[0])
        if args.expected_images is not None and image_count != args.expected_images:
            raise ValueError(f"expected {args.expected_images} images, found {image_count}")
        resolution = [int(value) for value in handle.attrs["resolution"]]
        records: dict[str, dict] = {}
        for name in REQUIRED_DATASETS:
            values = handle[name]
            if values.shape[0] != image_count:
                raise ValueError(f"{name} has a mismatched image axis")
            finite = True
            absolute_sum = 0.0
            for index in range(image_count):
                slice_values = values[index]
                finite = finite and bool(np.isfinite(slice_values).all())
                absolute_sum += float(np.abs(slice_values).sum(dtype=np.float64))
            if not finite:
                raise ValueError(f"{name} contains non-finite values")
            records[name] = {"shape": list(values.shape), "absolute_sum": absolute_sum}
        alpha_sum = records["alpha"]["absolute_sum"]
        if alpha_sum <= 0:
            raise ValueError("alpha is empty; no renderable canonical surface was recorded")
        result = {
            "label": "STATIC DATASET INTEGRITY ONLY",
            "dataset": str(dataset_path),
            "image_count": image_count,
            "resolution": resolution,
            "aabb_min": [float(value) for value in handle.attrs["aabb_min"]],
            "aabb_max": [float(value) for value in handle.attrs["aabb_max"]],
            "datasets": records,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
