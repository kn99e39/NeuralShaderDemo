"""Render and compare standard/current versus transported canonical position AOV."""

from __future__ import annotations

import argparse
import json
import pathlib

import bpy
import numpy as np
import pyexr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--rigid-metadata", type=pathlib.Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "32"
    scene.render.film_transparent = True
    scene.render.filepath = str(output / "beauty")
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    for node in scene.node_tree.nodes:
        if node.type == "OUTPUT_FILE":
            node.base_path = str(output)
    bpy.ops.render.render(write_still=True)
    current = pyexr.open(str(output / "position0001.exr")).get()[..., :3].astype(np.float32)
    canonical = pyexr.open(str(output / "canonical_position0001.exr")).get()[..., :3].astype(np.float32)
    beauty = pyexr.open(str(output / "beauty.exr")).get().astype(np.float32)
    mask = beauty[..., 3] > 0.0
    if not bool(mask.any()):
        raise ValueError("empty canonical identity render")
    expected = canonical
    audit_label = "CANONICAL POSITION AOV IDENTITY AUDIT"
    if args.rigid_metadata is not None:
        metadata = json.loads(args.rigid_metadata.resolve().read_text(encoding="utf-8"))
        transform = np.asarray(metadata["world_transform"], dtype=np.float32)
        if transform.shape != (4, 4):
            raise ValueError("rigid metadata must contain a 4x4 world_transform")
        homogeneous = np.concatenate(
            [canonical, np.ones((*canonical.shape[:2], 1), dtype=np.float32)], axis=-1
        )
        expected = (homogeneous @ transform.T)[..., :3]
        audit_label = "CANONICAL POSITION AOV RIGID-TRANSFORM AUDIT"
    difference = np.abs(current - expected)
    payload = {
        "label": audit_label,
        "blend": str(args.blend.resolve()),
        "resolution": [args.resolution, args.resolution],
        "covered_pixels": int(mask.sum()),
        "mae": float(difference[mask].mean()),
        "max_abs_error": float(difference[mask].max()),
        "p99_abs_error": float(np.quantile(difference[mask], 0.99)),
        "comparison": "current position versus transformed canonical AOV" if args.rigid_metadata else "current position versus canonical AOV",
        "rigid_metadata": str(args.rigid_metadata.resolve()) if args.rigid_metadata else None,
    }
    (output / "canonical_position_identity_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
