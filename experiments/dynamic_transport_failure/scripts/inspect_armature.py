"""List selected bone names from an armature in a Blender file."""

from __future__ import annotations

import argparse
import pathlib
import re

import bpy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", type=pathlib.Path, required=True)
    parser.add_argument("--armature", required=True)
    parser.add_argument("--name-regex", default=".*")
    args = parser.parse_args()
    bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
    armature = bpy.data.objects.get(args.armature)
    if armature is None or armature.type != "ARMATURE":
        raise ValueError(f"armature not found: {args.armature}")
    pattern = re.compile(args.name_regex, re.IGNORECASE)
    for bone in armature.data.bones:
        if pattern.search(bone.name):
            print(bone.name)


if __name__ == "__main__":
    main()
