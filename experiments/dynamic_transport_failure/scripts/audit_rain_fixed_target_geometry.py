"""GT-independent geometry audit for a fixed Rain target and moving neighbor.

The target is selected entirely from production rig weights and source
topology.  It reports the exact target/mover vertex sets, target world-space
invariance, and closest current-surface distance for a declared pose.  No RNA
render or neural error is read by this program.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import bpy
import numpy as np


def script_argv() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]


def group_indices(obj: bpy.types.Object, group_name: str, threshold: float) -> np.ndarray:
    group = obj.vertex_groups.get(group_name)
    if group is None:
        raise ValueError(f"missing vertex group: {group_name}")
    indices = []
    for vertex in obj.data.vertices:
        weight = next((entry.weight for entry in vertex.groups if entry.group == group.index), 0.0)
        if weight >= threshold:
            indices.append(vertex.index)
    if not indices:
        raise ValueError(f"no {group_name} vertices satisfy threshold {threshold}")
    return np.asarray(indices, dtype=np.int64)


def evaluated_world_vertices(obj: bpy.types.Object) -> np.ndarray:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        if len(mesh.vertices) != len(obj.data.vertices):
            raise ValueError("evaluated topology does not preserve source vertex indices")
        return np.asarray([evaluated.matrix_world @ vertex.co for vertex in mesh.vertices], dtype=np.float64)
    finally:
        evaluated.to_mesh_clear()


def target_triangle_indices(obj: bpy.types.Object, target_vertices: np.ndarray) -> np.ndarray:
    target_set = set(target_vertices.tolist())
    triangles = [poly.index for poly in obj.data.polygons if set(poly.vertices).issubset(target_set)]
    if not triangles:
        raise ValueError("weight-selected target has no wholly contained source triangles")
    return np.asarray(triangles, dtype=np.int64)


def min_distance(left: np.ndarray, right: np.ndarray) -> float:
    best = np.inf
    for start in range(0, len(left), 256):
        distances = np.linalg.norm(left[start : start + 256, None, :] - right[None, :, :], axis=-1)
        best = min(best, float(distances.min()))
    return best


def apply_pose(armature: bpy.types.Object, scarf3_x_degrees: float) -> None:
    bone = armature.pose.bones.get("FK-Scarf3")
    if bone is None:
        raise ValueError("missing FK-Scarf3")
    bone.rotation_mode = "XYZ"
    bone.rotation_euler.x += np.deg2rad(scarf3_x_degrees)
    bpy.context.view_layer.update()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--target-group", default="DEF-Scarf1")
    parser.add_argument("--moving-group", default="DEF-Scarf3")
    parser.add_argument("--weight-threshold", type=float, default=0.99)
    parser.add_argument("--scarf3-x-degrees", type=float, required=True)
    args = parser.parse_args(script_argv())
    if not 0.0 < args.weight_threshold <= 1.0:
        raise ValueError("weight threshold must be in (0, 1]")
    bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
    scarf = bpy.data.objects.get("GEO-rain_scarf")
    armature = bpy.data.objects.get("RIG-rain")
    if scarf is None or scarf.type != "MESH" or armature is None or armature.type != "ARMATURE":
        raise ValueError("expected GEO-rain_scarf and RIG-rain")
    target_vertices = group_indices(scarf, args.target_group, args.weight_threshold)
    moving_vertices = group_indices(scarf, args.moving_group, args.weight_threshold)
    baseline = evaluated_world_vertices(scarf)
    apply_pose(armature, args.scarf3_x_degrees)
    current = evaluated_world_vertices(scarf)
    target_delta = np.linalg.norm(current[target_vertices] - baseline[target_vertices], axis=1)
    moving_delta = np.linalg.norm(current[moving_vertices] - baseline[moving_vertices], axis=1)
    payload = {
        "label": "RAIN FIXED-TARGET GT-ONLY GEOMETRY AUDIT",
        "blend": str(args.blend.resolve()),
        "target": {
            "mesh": scarf.name,
            "vertex_group": args.target_group,
            "weight_threshold": args.weight_threshold,
            "vertex_count": int(len(target_vertices)),
            "source_triangle_count": int(len(target_triangle_indices(scarf, target_vertices))),
            "source_triangle_indices": target_triangle_indices(scarf, target_vertices).tolist(),
            "world_position_delta": {
                "max": float(target_delta.max()),
                "mean": float(target_delta.mean()),
                "p99": float(np.quantile(target_delta, 0.99)),
            },
        },
        "moving_neighbor": {
            "mesh": scarf.name,
            "vertex_group": args.moving_group,
            "vertex_count": int(len(moving_vertices)),
            "FK-Scarf3_local_X_degrees_added": args.scarf3_x_degrees,
            "world_position_delta": {
                "max": float(moving_delta.max()),
                "mean": float(moving_delta.mean()),
                "p99": float(np.quantile(moving_delta, 0.99)),
            },
            "minimum_target_neighbor_distance": min_distance(current[target_vertices], current[moving_vertices]),
        },
        "topology_identity_preserved": True,
        "rna_error_observed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
