"""Import an FBX in a clean BPy scene and summarize renderable meshes."""

from __future__ import annotations

import argparse
import json
import pathlib

import bpy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fbx", type=pathlib.Path, required=True)
    args = parser.parse_args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(args.fbx.resolve()))
    objects = []
    for obj in sorted(bpy.context.scene.objects, key=lambda item: item.name):
        if obj.type != "MESH":
            continue
        objects.append(
            {
                "name": obj.name,
                "vertices": len(obj.data.vertices),
                "polygons": len(obj.data.polygons),
                "uv_layers": [layer.name for layer in obj.data.uv_layers],
                "materials": [slot.material.name if slot.material else None for slot in obj.material_slots],
                "shape_keys": list(obj.data.shape_keys.key_blocks.keys()) if obj.data.shape_keys else [],
            }
        )
    print(json.dumps({"blender": bpy.app.version_string, "meshes": objects}, indent=2))


if __name__ == "__main__":
    main()
