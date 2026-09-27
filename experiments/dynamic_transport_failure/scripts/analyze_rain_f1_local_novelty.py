"""Compare F1 cyan-artifact directional inputs with same-surface training data."""

from __future__ import annotations

import argparse
import json
import pathlib

import bpy
import h5py
import numpy as np
import pyexr
from scipy.spatial import cKDTree


def load(path: pathlib.Path) -> np.ndarray:
    return pyexr.open(str(path)).get().astype(np.float32)


def mesh_points(blend: pathlib.Path, name: str) -> np.ndarray:
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    obj = bpy.data.objects[name]
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        return np.asarray([evaluated.matrix_world @ vertex.co for vertex in mesh.vertices], dtype=np.float32)
    finally:
        evaluated.to_mesh_clear()


def normalize(value: np.ndarray) -> np.ndarray:
    return value / np.clip(np.linalg.norm(value, axis=-1, keepdims=True), 1e-8, None)


def angle_degrees(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    dot = np.sum(normalize(left) * normalize(right), axis=-1)
    return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))


def describe(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    return {
        "supported_pixels": int(len(finite)),
        "unsupported_pixels": int(len(values) - len(finite)),
        "median_degrees": None if not len(finite) else float(np.median(finite)),
        "p95_degrees": None if not len(finite) else float(np.quantile(finite, 0.95)),
        "mean_degrees": None if not len(finite) else float(np.mean(finite)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=pathlib.Path, required=True)
    parser.add_argument("--rna", type=pathlib.Path, required=True)
    parser.add_argument("--canonical-position", type=pathlib.Path, required=True)
    parser.add_argument("--current-position", type=pathlib.Path, required=True)
    parser.add_argument("--current-normal", type=pathlib.Path, required=True)
    parser.add_argument("--current-camera-dir", type=pathlib.Path, required=True)
    parser.add_argument("--camera-json", type=pathlib.Path, required=True)
    parser.add_argument("--training-h5", type=pathlib.Path, required=True)
    parser.add_argument("--canonical-blend", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--support-radius", type=float, default=0.005)
    args = parser.parse_args()
    reference, rna, canonical = load(args.reference), load(args.rna), load(args.canonical_position)
    position, normal, camera = load(args.current_position), load(args.current_normal), load(args.current_camera_dir)
    valid = (reference[..., 3] > 0) | (rna[..., 3] > 0)
    error = np.abs(reference[..., :3] - rna[..., :3]).mean(axis=-1)
    cyan = valid & (rna[..., 1] > 0.25) & (rna[..., 2] > 0.25) & (rna[..., 0] < 0.6 * np.minimum(rna[..., 1], rna[..., 2])) & (error >= np.quantile(error[valid], 0.95))
    scarf_tree = cKDTree(mesh_points(args.canonical_blend.resolve(), "GEO-rain_scarf"))
    top_tree = cKDTree(mesh_points(args.canonical_blend.resolve(), "GEO-rain_top"))
    all_points = canonical[..., :3].reshape(-1, 3)
    all_scarf = scarf_tree.query(all_points)[0] < top_tree.query(all_points)[0]
    cyan_flat = cyan.reshape(-1)
    majority_scarf = bool(all_scarf[cyan_flat].mean() >= 0.5)
    control_candidates = np.flatnonzero(valid.reshape(-1) & ~cyan_flat & (all_scarf == majority_scarf) & (error.reshape(-1) <= np.quantile(error[valid], 0.5)))
    artifact_indices = np.flatnonzero(cyan_flat)
    if len(control_candidates) < len(artifact_indices):
        raise ValueError("not enough deterministic same-surface low-error controls")
    control_indices = control_candidates[np.linspace(0, len(control_candidates) - 1, len(artifact_indices), dtype=np.int64)]
    selected_indices = np.concatenate([artifact_indices, control_indices])
    selected_points = all_points[selected_indices]
    selected_scarf = all_scarf[selected_indices]
    current_normal = normal[..., :3].reshape(-1, 3)[selected_indices]
    current_camera = camera[..., :3].reshape(-1, 3)[selected_indices]
    light_position = np.asarray(json.loads(args.camera_json.read_text(encoding="utf-8"))["frames"][0]["pl_pos"], dtype=np.float32)
    current_light = normalize(light_position[None, :] - position[..., :3].reshape(-1, 3)[selected_indices])
    best_normal = np.full(len(selected_indices), np.inf, dtype=np.float32)
    best_camera = np.full(len(selected_indices), np.inf, dtype=np.float32)
    best_light = np.full(len(selected_indices), np.inf, dtype=np.float32)
    with h5py.File(args.training_h5.resolve(), "r") as dataset:
        for frame in range(dataset["position"].shape[0]):
            alpha = dataset["alpha"][frame, :, 0] > 0
            positions = dataset["position"][frame, alpha, :].astype(np.float32)
            distances, indices = cKDTree(positions).query(selected_points)
            nearest_positions = positions[indices]
            same = (scarf_tree.query(nearest_positions)[0] < top_tree.query(nearest_positions)[0]) == selected_scarf
            keep = same & (distances <= args.support_radius)
            if not keep.any():
                continue
            normals = dataset["normal"][frame, alpha, :].astype(np.float32)[indices]
            cameras = dataset["camera_dir"][frame, alpha, :].astype(np.float32)[indices]
            lights = dataset["light_dir"][frame, alpha, :].astype(np.float32)[indices]
            best_normal[keep] = np.minimum(best_normal[keep], angle_degrees(current_normal[keep], normals[keep]))
            best_camera[keep] = np.minimum(best_camera[keep], angle_degrees(current_camera[keep], cameras[keep]))
            best_light[keep] = np.minimum(best_light[keep], angle_degrees(current_light[keep], lights[keep]))
    count = len(artifact_indices)
    payload = {
        "label": "RAIN F1 SAME-SURFACE LOCAL INPUT NOVELTY",
        "support_radius": args.support_radius,
        "control_definition": "same canonical provenance as artifact majority; non-cyan; valid-pixel error at or below full-frame median; deterministic evenly spaced sample",
        "artifact": {"pixels": count, "normal": describe(best_normal[:count]), "camera_direction": describe(best_camera[:count]), "light_direction": describe(best_light[:count])},
        "control": {"pixels": count, "normal": describe(best_normal[count:]), "camera_direction": describe(best_camera[count:]), "light_direction": describe(best_light[count:])},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
