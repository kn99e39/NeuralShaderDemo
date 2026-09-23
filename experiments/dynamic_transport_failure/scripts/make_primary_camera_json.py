"""Export the canonical scene's fixed review camera with a fixed RNA light."""

from __future__ import annotations

import argparse
import json
import pathlib

import bpy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", type=pathlib.Path, required=True)
    parser.add_argument("--light-source", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--light-frame-index", type=int, default=0)
    args = parser.parse_args()
    bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
    camera = bpy.context.scene.camera
    if camera is None or camera.type != "CAMERA":
        raise ValueError("canonical scene has no active review camera")
    light_source = json.loads(args.light_source.resolve().read_text(encoding="utf-8"))
    frames = light_source.get("frames", [])
    if not 0 <= args.light_frame_index < len(frames):
        raise ValueError("light frame index outside source")
    source_frame = frames[args.light_frame_index]
    payload = {
        "camera_angle_x": float(camera.data.angle_x),
        "frames": [
            {
                "file_path": "primary_fixed_geometry_state_0",
                "transform_matrix": [[float(value) for value in row] for row in camera.matrix_world],
                "pl_pos": source_frame["pl_pos"],
                "pl_intensity": source_frame["pl_intensity"],
            }
        ],
        "fixed_experimental_variables": {
            "camera": "canonical scene HQ-B1-Camera",
            "light": f"held-out review light frame {args.light_frame_index}",
            "intent": "one front garment/scarf framing shared without change by F0-F3",
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
