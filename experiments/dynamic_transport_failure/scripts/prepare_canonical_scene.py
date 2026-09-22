"""Create an isolated canonical scene without changing source topology.

The saved file is an ignored run input.  It retains source meshes, material
graphs, UVs, and armature modifiers; only non-target objects are hidden from
render and the static-gate camera resolution is set.  It is deliberately not a
deformation script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import bpy
from mathutils import Vector


ASSET_SELECTIONS = {
    "character": {
        "source_id": "character_einar_v1",
        "mesh_prefixes": ("GEO-einar_", "GEO-arm_", "GEO-shoulder_", "GEO-upper_arm_", "GEO-forearm_", "GEO-wrist", "GEO-thumb"),
        "armature": "RIG-einar",
    },
    "cloth": {
        "source_id": "cloth_rain_v2_garment_set",
        "mesh_names": ("GEO-rain_scarf", "GEO-rain_top"),
        "armature": "RIG-rain",
    },
}


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_selected_mesh(obj: bpy.types.Object, selection: dict) -> bool:
    if obj.type != "MESH" or len(obj.data.polygons) == 0:
        return False
    if obj.name in selection.get("mesh_names", ()):
        return True
    return obj.name.startswith(selection.get("mesh_prefixes", ()))


def target_bounds(objects: list[bpy.types.Object]) -> tuple[Vector, float]:
    points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    low = Vector((min(point.x for point in points), min(point.y for point in points), min(point.z for point in points)))
    high = Vector((max(point.x for point in points), max(point.y for point in points), max(point.z for point in points)))
    return (low + high) * 0.5, max((high - low).length * 0.5, 0.1)


def point_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def install_fixed_review_rig(
    scene: bpy.types.Scene, objects: list[bpy.types.Object], asset: str
) -> dict:
    """Install the sole camera/lighting setup used by all states of one asset."""

    for obj in list(bpy.data.objects):
        if obj.name.startswith("HQ-B1-"):
            bpy.data.objects.remove(obj, do_unlink=True)
    center, radius = target_bounds(objects)
    if asset == "character":
        # The character evidence is an arm--torso interaction, not a full-body
        # turntable.  Crop the fixed review camera to the upper torso/arms while
        # leaving the exact same camera untouched for P0--P4.
        bound_points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
        full_height = max(point.z for point in bound_points) - min(point.z for point in bound_points)
        center.z += 0.18 * full_height
        radius *= 0.62
    camera_data = bpy.data.cameras.new("HQ-B1-Camera")
    camera = bpy.data.objects.new("HQ-B1-Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = center + Vector((0.65 * radius, -2.75 * radius, 0.30 * radius))
    camera_data.lens = 58
    point_at(camera, center + Vector((0.0, 0.0, 0.05 * radius)))
    scene.camera = camera

    lights = (
        ("HQ-B1-Key", (1.30, -1.15, 1.65), 60.0, 1.0),
        ("HQ-B1-Fill", (-1.25, -1.00, 0.55), 15.0, 1.4),
        ("HQ-B1-Rim", (0.15, 1.10, 1.75), 35.0, 0.8),
    )
    for name, offset, energy, size in lights:
        data = bpy.data.lights.new(name, "AREA")
        data.energy = energy
        data.shape = "DISK"
        data.size = max(size * radius, 0.25)
        light = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(light)
        light.location = center + radius * Vector(offset)
        point_at(light, center)
    return {"camera": camera.name, "target_center": list(center), "target_radius": radius, "lights": [item[0] for item in lights]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", choices=sorted(ASSET_SELECTIONS), required=True)
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--metadata", type=pathlib.Path, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    args = parser.parse_args()

    selection = ASSET_SELECTIONS[args.asset]
    source = args.source.resolve()
    output = args.output.resolve()
    metadata_path = args.metadata.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=str(source))

    included = []
    selected_objects = []
    for obj in bpy.context.scene.objects:
        selected = is_selected_mesh(obj, selection)
        if obj.type == "MESH":
            obj.hide_render = not selected
            if selected:
                selected_objects.append(obj)
                included.append(
                    {
                        "name": obj.name,
                        "vertices": len(obj.data.vertices),
                        "polygons": len(obj.data.polygons),
                        "materials": [slot.material.name if slot.material else None for slot in obj.material_slots],
                        "uv_layers": [layer.name for layer in obj.data.uv_layers],
                        "modifiers": [modifier.type for modifier in obj.modifiers],
                    }
                )
    if not included:
        raise RuntimeError(f"no target meshes selected for {args.asset}")
    armature = bpy.data.objects.get(selection["armature"])
    if armature is None or armature.type != "ARMATURE":
        raise RuntimeError(f"expected armature not found: {selection['armature']}")
    # An armature never renders but must stay present because the original
    # modifiers refer to it.  The source file, armature, modifiers, material
    # graphs and texture paths are otherwise preserved untouched.
    armature.hide_render = True
    scene = bpy.context.scene
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    review_rig = install_fixed_review_rig(scene, selected_objects, args.asset)
    # Production rigs can retain image datablocks for deliberately hidden props.
    # Keep a record of a missing *unused* source instead of preventing a saved
    # target scene; selected material nodes still pack before Blender reports
    # the unrelated missing path.  Static-gate inspection must reject any
    # missing image referenced by an included mesh's material graph.
    pack_warning = None
    try:
        bpy.ops.file.pack_all()
    except RuntimeError as exc:
        pack_warning = str(exc)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    metadata_path.write_text(
        json.dumps(
            {
                "asset": args.asset,
                "asset_id": selection["source_id"],
                "source_blend": str(source),
                "source_sha256": sha256(source),
                "output_blend": str(output),
                "review_rig": review_rig,
                "resolution": [args.resolution, args.resolution],
                "included_meshes": included,
                "topology_mutation": False,
                "material_mutation": False,
                "pose_mutation": False,
                "pack_warning": pack_warning,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
