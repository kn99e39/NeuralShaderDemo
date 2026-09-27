"""Quantify canonical training support for the historical Rain F1 cyan artifact."""

from __future__ import annotations

import argparse
import json
import pathlib

import bpy
import h5py
import numpy as np
import pyexr
from PIL import Image
from scipy.spatial import cKDTree


def exr(path: pathlib.Path) -> np.ndarray:
    return pyexr.open(str(path)).get().astype(np.float32)


def evaluated_points(blend: pathlib.Path, object_name: str) -> np.ndarray:
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    obj = bpy.data.objects.get(object_name)
    if obj is None or obj.type != "MESH":
        raise ValueError(f"missing mesh: {object_name}")
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        return np.asarray([evaluated.matrix_world @ vertex.co for vertex in mesh.vertices], dtype=np.float32)
    finally:
        evaluated.to_mesh_clear()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=pathlib.Path, required=True)
    parser.add_argument("--rna", type=pathlib.Path, required=True)
    parser.add_argument("--canonical-position", type=pathlib.Path, required=True)
    parser.add_argument("--training-h5", type=pathlib.Path, required=True)
    parser.add_argument("--canonical-blend", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--support-radius", type=float, default=0.005)
    args = parser.parse_args()
    if args.support_radius <= 0.0:
        raise ValueError("support radius must be positive")
    reference, rna, canonical = exr(args.reference), exr(args.rna), exr(args.canonical_position)
    if reference.shape[:2] != rna.shape[:2] or canonical.shape[:2] != reference.shape[:2]:
        raise ValueError("reference, RNA, and canonical AOV dimensions must match")
    valid = (reference[..., 3] > 0.0) | (rna[..., 3] > 0.0)
    error = np.abs(reference[..., :3] - rna[..., :3]).mean(axis=-1)
    threshold = float(np.quantile(error[valid], 0.95))
    cyan = (
        valid
        & (rna[..., 1] > 0.25)
        & (rna[..., 2] > 0.25)
        & (rna[..., 0] < 0.6 * np.minimum(rna[..., 1], rna[..., 2]))
        & (error >= threshold)
    )
    artifact_points = canonical[..., :3][cyan]
    if len(artifact_points) == 0:
        raise ValueError("cyan artifact rule selected no pixels")
    scarf_tree = cKDTree(evaluated_points(args.canonical_blend.resolve(), "GEO-rain_scarf"))
    top_tree = cKDTree(evaluated_points(args.canonical_blend.resolve(), "GEO-rain_top"))
    scarf_distance, _ = scarf_tree.query(artifact_points)
    top_distance, _ = top_tree.query(artifact_points)
    is_scarf = scarf_distance < top_distance
    support = np.zeros(len(artifact_points), dtype=np.int32)
    nearest = np.full(len(artifact_points), np.inf, dtype=np.float32)
    with h5py.File(args.training_h5.resolve(), "r") as dataset:
        for index in range(dataset["position"].shape[0]):
            alpha = dataset["alpha"][index, :, 0] > 0.0
            positions = dataset["position"][index, alpha, :].astype(np.float32)
            if len(positions) == 0:
                continue
            distances, _ = cKDTree(positions).query(artifact_points)
            support += distances <= args.support_radius
            nearest = np.minimum(nearest, distances)
    artifact_map = np.zeros((*cyan.shape, 3), dtype=np.uint8)
    flat = artifact_map.reshape(-1, 3)
    flat_indices = np.flatnonzero(cyan.reshape(-1))
    # Green: canonical top; red: scarf. Brightness encodes observed view count.
    brightness = np.clip(support / 10.0, 0.15, 1.0)
    flat[flat_indices[is_scarf], 0] = (255 * brightness[is_scarf]).astype(np.uint8)
    flat[flat_indices[~is_scarf], 1] = (255 * brightness[~is_scarf]).astype(np.uint8)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(artifact_map, mode="RGB").save(output / "F1_cyan_artifact_canonical_coverage.png")
    payload = {
        "label": "HISTORICAL RAIN F1 CYAN ARTIFACT CANONICAL COVERAGE",
        "selection": {
            "cyan_rule": "RNA G,B > 0.25; RNA R < 0.6*min(G,B); mean absolute RGB error >= full-frame valid-pixel p95",
            "error_p95": threshold,
            "pixels": int(cyan.sum()),
            "support_radius": args.support_radius,
        },
        "surface_identity": {
            "scarf_pixels": int(is_scarf.sum()),
            "top_pixels": int((~is_scarf).sum()),
            "scarf_fraction": float(is_scarf.mean()),
            "top_fraction": float((~is_scarf).mean()),
        },
        "canonical_training_support_views": {
            "mean": float(support.mean()),
            "median": float(np.median(support)),
            "zero_view_pixels": int((support == 0).sum()),
            "zero_view_fraction": float((support == 0).mean()),
            "at_least_one_view_fraction": float((support > 0).mean()),
            "nearest_sample_distance_median": float(np.median(nearest)),
            "nearest_sample_distance_p95": float(np.quantile(nearest, 0.95)),
        },
    }
    (output / "F1_cyan_artifact_canonical_coverage.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
