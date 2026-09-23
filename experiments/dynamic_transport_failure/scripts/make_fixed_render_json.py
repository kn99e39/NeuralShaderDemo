"""Freeze one already-reviewed camera/light state for geometry trajectories."""

from __future__ import annotations

import argparse
import json
import pathlib


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    args = parser.parse_args()
    payload = json.loads(args.input.resolve().read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
    if not 0 <= args.frame_index < len(frames):
        raise ValueError("frame index outside reviewed camera sequence")
    selected = dict(frames[args.frame_index])
    selected["file_path"] = "fixed_geometry_state_0"
    payload["frames"] = [selected]
    payload["fixed_experimental_variables"] = {
        "source_camera_light_review": str(args.input.resolve()),
        "source_frame_index": args.frame_index,
        "camera": "fixed",
        "light": "fixed",
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
