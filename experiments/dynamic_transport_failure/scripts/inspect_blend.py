"""Print a deterministic, JSON-friendly inventory of a Blender asset.

Run with the official RNA custom-BPy Python, not a system Python:
``.venv/bin/python inspect_blend.py --blend /path/to/file.blend``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re

import bpy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", type=pathlib.Path, required=True)
    parser.add_argument("--name-regex", help="only report objects whose name matches this regex")
    args = parser.parse_args()
    bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
    objects = []
    name_pattern = re.compile(args.name_regex) if args.name_regex else None
    for obj in sorted(bpy.context.scene.objects, key=lambda item: item.name):
        if name_pattern and not name_pattern.search(obj.name):
            continue
        record = {"name": obj.name, "type": obj.type, "hidden_render": bool(obj.hide_render)}
        if obj.type == "MESH":
            record.update(
                vertices=len(obj.data.vertices),
                polygons=len(obj.data.polygons),
                materials=[slot.material.name if slot.material else None for slot in obj.material_slots],
                modifiers=[modifier.type for modifier in obj.modifiers],
                uv_layers=[layer.name for layer in obj.data.uv_layers],
            )
        elif obj.type == "ARMATURE":
            record["bones"] = len(obj.data.bones)
        objects.append(record)
    print(json.dumps({"blender": bpy.app.version_string, "objects": objects}, indent=2))


if __name__ == "__main__":
    main()
