"""Derive a whole-domain rigid-control scene without changing its internals."""

from __future__ import annotations

import argparse
import json
import pathlib

import bpy
from mathutils import Matrix, Vector


def matrix_payload(matrix: Matrix) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--metadata", type=pathlib.Path, required=True)
    parser.add_argument("--armature", required=True)
    parser.add_argument("--translation", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--rotation-degrees-z", type=float, default=0.0)
    parser.add_argument(
        "--application",
        choices=("armature", "all_objects"),
        default="armature",
        help="Use the production armature object to move the whole skinned domain once.",
    )
    args = parser.parse_args()

    bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
    armature = bpy.data.objects.get(args.armature)
    if armature is None or armature.type != "ARMATURE":
        raise ValueError(f"expected armature: {args.armature}")
    domain = [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and not obj.hide_render]
    if not domain:
        raise ValueError("no render-visible domain meshes")

    rotation = Matrix.Rotation(float(args.rotation_degrees_z) * 3.141592653589793 / 180.0, 4, "Z")
    transform = Matrix.Translation(Vector(args.translation)) @ rotation
    # The armature modifier already applies its object-world change to each
    # skinned domain mesh.  Moving both armature and meshes would apply a
    # translation twice.  The default therefore transforms the production
    # armature object once; it is a whole-domain rigid motion, not a pose edit.
    transformed = [armature] if args.application == "armature" else [*domain, armature]
    for obj in transformed:
        obj.matrix_world = transform @ obj.matrix_world
    bpy.context.view_layer.update()

    output = args.output.resolve()
    metadata = args.metadata.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    metadata.write_text(
        json.dumps(
            {
                "label": "WHOLE-DOMAIN RIGID CORRESPONDENCE CONTROL",
                "input_blend": str(args.blend.resolve()),
                "output_blend": str(output),
                "armature": armature.name,
                "domain_meshes": [obj.name for obj in domain],
                "world_transform": matrix_payload(transform),
                "translation": list(args.translation),
                "rotation_degrees_z": float(args.rotation_degrees_z),
                "application": args.application,
                "internal_pose_mutation": False,
                "topology_mutation": False,
                "material_mutation": False,
                "correspondence_expectation": "current_position == world_transform @ canonical_position, subject only to evaluated modifier interpolation",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
