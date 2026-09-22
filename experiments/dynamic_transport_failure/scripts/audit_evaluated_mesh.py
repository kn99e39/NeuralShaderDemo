"""Audit source versus evaluated render mesh at canonical and engineering poses."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import bpy
import numpy as np


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def mesh_record(obj: bpy.types.Object, depsgraph: bpy.types.Depsgraph) -> dict:
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        coords = np.empty((len(mesh.vertices), 3), dtype=np.float32)
        mesh.vertices.foreach_get("co", coords.reshape(-1))
        triangles = np.asarray([triangle.vertices[:] for triangle in mesh.loop_triangles], dtype=np.int32)
        return {
            "object": obj.name,
            "source_vertices": len(obj.data.vertices), "source_polygons": len(obj.data.polygons),
            "evaluated_vertices": len(mesh.vertices), "evaluated_polygons": len(mesh.polygons),
            "evaluated_triangles": len(mesh.loop_triangles),
            "material_slots": [slot.material.name if slot.material else None for slot in obj.material_slots],
            "uv_layers": [layer.name for layer in mesh.uv_layers],
            "modifier_stack": [modifier.type for modifier in obj.modifiers],
            "triangle_index_sha256": digest(triangles), "coordinate_sha256": digest(coords),
        }
    finally:
        evaluated.to_mesh_clear()


def collect(objects: list[bpy.types.Object]) -> list[dict]:
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    return [mesh_record(obj, depsgraph) for obj in objects]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", type=pathlib.Path, required=True)
    parser.add_argument("--objects", nargs="+", default=[])
    parser.add_argument("--render-visible", action="store_true", help="Audit every render-visible mesh in the opened scene.")
    parser.add_argument("--armature")
    parser.add_argument("--pose-bone")
    parser.add_argument("--engineering-rotation-degrees", type=float, default=12.0)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
    if args.render_visible:
        objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and not obj.hide_render]
    else:
        objects = [bpy.data.objects[name] for name in args.objects]
    if not objects:
        raise ValueError("select objects explicitly or pass --render-visible")
    canonical = collect(objects)
    engineering = canonical
    if args.armature and args.pose_bone:
        armature = bpy.data.objects[args.armature]
        pose_bone = armature.pose.bones.get(args.pose_bone)
        if pose_bone is None:
            raise ValueError(f"pose bone not found: {args.pose_bone}")
        pose_bone.rotation_mode = "XYZ"
        pose_bone.rotation_euler.x += np.deg2rad(args.engineering_rotation_degrees)
        engineering = collect(objects)
    comparison = [
        {
            "object": before["object"],
            "evaluated_vertex_count_stable": before["evaluated_vertices"] == after["evaluated_vertices"],
            "evaluated_triangle_count_stable": before["evaluated_triangles"] == after["evaluated_triangles"],
            "triangle_index_order_stable": before["triangle_index_sha256"] == after["triangle_index_sha256"],
            "material_slots_stable": before["material_slots"] == after["material_slots"],
            "uv_layers_stable": before["uv_layers"] == after["uv_layers"],
            "coordinates_changed": before["coordinate_sha256"] != after["coordinate_sha256"],
        }
        for before, after in zip(canonical, engineering)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "label": "ENGINEERING VALIDATION ONLY", "blend": str(args.blend.resolve()), "objects": [obj.name for obj in objects],
        "pose_control": {"armature": args.armature, "bone": args.pose_bone, "rotation_degrees": args.engineering_rotation_degrees} if args.armature and args.pose_bone else None,
        "canonical": canonical, "engineering_pose": engineering, "comparison": comparison,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
