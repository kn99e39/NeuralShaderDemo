"""Create one predeclared production-Rain scarf/top deformation state.

The source is reopened for every state.  Only the documented FK scarf control
is changed; mesh data, materials, AOV instrumentation, camera and lights are
carried through unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

import bpy


STATE_SPECS = {
    "F0": {"scarf2_x_degrees": 0.0, "scarf3_x_degrees": 0.0, "meaning": "canonical open configuration"},
    "F1": {"scarf2_x_degrees": 32.0, "scarf3_x_degrees": -20.0, "meaning": "moderate two-link production-rig flexion"},
    "F2": {"scarf2_x_degrees": 64.0, "scarf3_x_degrees": -48.0, "meaning": "deep two-link fold / cavity progression"},
    "F3": {"scarf2_x_degrees": 86.0, "scarf3_x_degrees": -72.0, "meaning": "strong two-link fold / near-contact progression"},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", type=pathlib.Path, required=True)
    parser.add_argument("--state", choices=sorted(STATE_SPECS), required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--metadata", type=pathlib.Path, required=True)
    parser.add_argument("--armature", default="RIG-rain")
    parser.add_argument("--bone", default="FK-Scarf2")
    parser.add_argument("--tip-bone", default="FK-Scarf3")
    args = parser.parse_args()
    spec = STATE_SPECS[args.state]
    bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
    armature = bpy.data.objects.get(args.armature)
    if armature is None or armature.type != "ARMATURE":
        raise ValueError(f"expected armature: {args.armature}")
    bone = armature.pose.bones.get(args.bone)
    if bone is None:
        raise ValueError(f"expected production scarf control: {args.bone}")
    tip_bone = armature.pose.bones.get(args.tip_bone)
    if tip_bone is None:
        raise ValueError(f"expected production scarf tip control: {args.tip_bone}")
    bone.rotation_mode = "XYZ"
    # FK-Scarf2 points along its local Y chain direction.  Its local X axis
    # is the production flexion axis; local Y only twists the scarf and cannot
    # create the specified fold/cavity trajectory.
    bone.rotation_euler.x += math.radians(spec["scarf2_x_degrees"])
    tip_bone.rotation_mode = "XYZ"
    tip_bone.rotation_euler.x += math.radians(spec["scarf3_x_degrees"])
    bpy.context.view_layer.update()

    output = args.output.resolve()
    metadata = args.metadata.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    metadata.write_text(
        json.dumps(
            {
                "label": "PREDECLARED RAIN FOLD TRAJECTORY",
                "state": args.state,
                "input_blend": str(args.blend.resolve()),
                "output_blend": str(output),
                "armature": armature.name,
                "bone": bone.name,
                "tip_bone": tip_bone.name,
                "rotation_axis": "local X (production flexion axis)",
                "rotation_degrees_added": spec["scarf2_x_degrees"],
                "tip_rotation_degrees_added": spec["scarf3_x_degrees"],
                "meaning": spec["meaning"],
                "domain_meshes": ["GEO-rain_scarf", "GEO-rain_top"],
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
