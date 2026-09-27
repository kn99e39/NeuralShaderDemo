"""Measure topology and cross-segment geometry for a locked Rain state."""

from __future__ import annotations

import argparse
import json
import pathlib

import bpy
import numpy as np


def group_indices(obj: bpy.types.Object, group_name: str, threshold: float) -> np.ndarray:
    group = obj.vertex_groups.get(group_name)
    if group is None:
        raise ValueError(f"missing vertex group: {group_name}")
    selected = []
    for vertex in obj.data.vertices:
        weight = next((item.weight for item in vertex.groups if item.group == group.index), 0.0)
        if weight >= threshold:
            selected.append(vertex.index)
    if not selected:
        raise ValueError(f"no vertices pass {group_name} threshold {threshold}")
    return np.asarray(selected, dtype=np.int64)


def min_pair_distance(left: np.ndarray, right: np.ndarray) -> float:
    best = np.inf
    for start in range(0, len(left), 256):
        block = left[start : start + 256]
        distances = np.linalg.norm(block[:, None, :] - right[None, :, :], axis=-1)
        best = min(best, float(distances.min()))
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--mesh", default="GEO-rain_scarf")
    parser.add_argument("--group-a", default="DEF-Scarf2")
    parser.add_argument("--group-b", default="DEF-Scarf3")
    parser.add_argument("--weight-threshold", type=float, default=0.8)
    parser.add_argument("--scarf1-x-degrees", type=float)
    parser.add_argument("--scarf2-x-degrees", type=float)
    parser.add_argument("--scarf3-x-degrees", type=float)
    args = parser.parse_args()
    if not 0.0 < args.weight_threshold <= 1.0:
        raise ValueError("weight threshold must be in (0, 1]")

    bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
    if any(value is not None for value in (args.scarf1_x_degrees, args.scarf2_x_degrees, args.scarf3_x_degrees)):
        armature = bpy.data.objects.get("RIG-rain")
        if armature is None or armature.type != "ARMATURE":
            raise ValueError("expected RIG-rain armature for pose probe")
        for bone_name, degrees in (("FK-Scarf1", args.scarf1_x_degrees), ("FK-Scarf2", args.scarf2_x_degrees), ("FK-Scarf3", args.scarf3_x_degrees)):
            if degrees is None:
                continue
            bone = armature.pose.bones.get(bone_name)
            if bone is None:
                raise ValueError(f"expected pose bone: {bone_name}")
            bone.rotation_mode = "XYZ"
            bone.rotation_euler.x += np.deg2rad(degrees)
        bpy.context.view_layer.update()
    obj = bpy.data.objects.get(args.mesh)
    if obj is None or obj.type != "MESH":
        raise ValueError(f"expected mesh: {args.mesh}")
    indices_a = group_indices(obj, args.group_a, args.weight_threshold)
    indices_b = group_indices(obj, args.group_b, args.weight_threshold)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    evaluated_mesh = evaluated.to_mesh()
    try:
        if len(evaluated_mesh.vertices) != len(obj.data.vertices):
            raise ValueError("evaluated scarf topology no longer preserves source vertex identity")
        coordinates = np.asarray([evaluated.matrix_world @ vertex.co for vertex in evaluated_mesh.vertices], dtype=np.float64)
    finally:
        evaluated.to_mesh_clear()

    points_a, points_b = coordinates[indices_a], coordinates[indices_b]
    payload = {
        "label": "RAIN INTERACTION GEOMETRY AUDIT",
        "blend": str(args.blend.resolve()),
        "mesh": obj.name,
        "source_vertices": len(obj.data.vertices),
        "source_polygons": len(obj.data.polygons),
        "evaluated_vertices": len(coordinates),
        "group_a": {"name": args.group_a, "vertices": int(len(indices_a)), "centroid": points_a.mean(axis=0).tolist()},
        "group_b": {"name": args.group_b, "vertices": int(len(indices_b)), "centroid": points_b.mean(axis=0).tolist()},
        "cross_segment_min_distance": min_pair_distance(points_a, points_b),
        "cross_segment_centroid_distance": float(np.linalg.norm(points_a.mean(axis=0) - points_b.mean(axis=0))),
        "weight_threshold": args.weight_threshold,
        "scarf1_x_degrees_added": args.scarf1_x_degrees,
        "scarf2_x_degrees_added": args.scarf2_x_degrees,
        "scarf3_x_degrees_added": args.scarf3_x_degrees,
        "topology_identity_preserved": True,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()