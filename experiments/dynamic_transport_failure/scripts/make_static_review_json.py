"""Create deterministic held-out camera/light frames for RNA static review."""

from __future__ import annotations

import argparse
import json
import pathlib

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--camera-json", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--camera-angle-x-degrees", type=float, default=45.0)
    parser.add_argument("--point-light-intensity", type=float, default=5.0)
    args = parser.parse_args()
    source_cameras = json.loads(args.camera_json.resolve().read_text(encoding="utf-8"))["cameras"]
    if len(source_cameras) < args.frames:
        raise ValueError("not enough held-out cameras")
    with h5py.File(args.dataset.resolve(), "r") as dataset:
        aabb_min, aabb_max = dataset.attrs["aabb_min"], dataset.attrs["aabb_max"]
    center = (aabb_min + aabb_max) * 0.5
    radius = float(np.linalg.norm(aabb_max - aabb_min) * 2.25)
    rng = np.random.default_rng(args.seed)
    frames = []
    for index, camera in enumerate(source_cameras[:args.frames]):
        direction = rng.normal(size=3)
        direction[2] = abs(direction[2]) + 0.2
        direction = direction / np.linalg.norm(direction)
        frames.append({
            "transform_matrix": camera["mtx"],
            "pl_pos": (center + radius * direction).tolist(),
            "pl_intensity": [args.point_light_intensity],
            "frame_index": index,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "camera_angle_x": float(np.deg2rad(args.camera_angle_x_degrees)),
        "frames": frames,
        "provenance": {
            "held_out_camera_source": str(args.camera_json.resolve()),
            "dataset": str(args.dataset.resolve()),
            "light_sampling_seed": args.seed,
            "light_semantics": "deterministic point lights on a positive-z sphere around canonical AABB center",
        },
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
