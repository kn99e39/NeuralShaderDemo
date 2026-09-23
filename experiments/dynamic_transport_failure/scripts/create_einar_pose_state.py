"""Create a fixed Einar arm--torso production-rig trajectory state."""

from __future__ import annotations

import argparse
import json
import math
import pathlib

import bpy


STATE_SPECS = {
    "P0": {"upper_arm_x": 0.0, "forearm_x": 0.0, "meaning": "separated canonical configuration"},
    "P1": {"upper_arm_x": 18.0, "forearm_x": 0.0, "meaning": "moderate upper-arm articulation with limited interaction"},
    "P2": {"upper_arm_x": 35.0, "forearm_x": 10.0, "meaning": "meaningful arm--torso approach"},
    "P3": {"upper_arm_x": 52.0, "forearm_x": 22.0, "meaning": "near-contact shoulder/armpit configuration"},
    "P4": {"upper_arm_x": 68.0, "forearm_x": 35.0, "meaning": "closest stable arm--torso configuration"},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", type=pathlib.Path, required=True)
    parser.add_argument("--state", choices=sorted(STATE_SPECS), required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--metadata", type=pathlib.Path, required=True)
    parser.add_argument("--armature", default="RIG-einar")
    parser.add_argument("--upper-arm-bone", default="FK-UpperArm.R")
    parser.add_argument("--forearm-bone", default="FK-Forearm.R")
    args = parser.parse_args()
    spec = STATE_SPECS[args.state]
    bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
    armature = bpy.data.objects.get(args.armature)
    if armature is None or armature.type != "ARMATURE":
        raise ValueError(f"expected armature: {args.armature}")
    upper_arm = armature.pose.bones.get(args.upper_arm_bone)
    forearm = armature.pose.bones.get(args.forearm_bone)
    if upper_arm is None or forearm is None:
        raise ValueError("expected production FK upper-arm and forearm controls")
    for bone, degrees in ((upper_arm, spec["upper_arm_x"]), (forearm, spec["forearm_x"])):
        bone.rotation_mode = "XYZ"
        bone.rotation_euler.x += math.radians(degrees)
    bpy.context.view_layer.update()
    output, metadata = args.output.resolve(), args.metadata.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    metadata.write_text(
        json.dumps(
            {
                "label": "PREDECLARED EINAR ARM-TORSO TRAJECTORY",
                "state": args.state,
                "input_blend": str(args.blend.resolve()),
                "output_blend": str(output),
                "armature": armature.name,
                "controls": [
                    {"bone": upper_arm.name, "axis": "local X", "degrees_added": spec["upper_arm_x"]},
                    {"bone": forearm.name, "axis": "local X", "degrees_added": spec["forearm_x"]},
                ],
                "meaning": spec["meaning"],
                "predeclared": True,
                "mesh_topology_mutation": False,
                "material_mutation": False,
                "camera_or_light_mutation": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
