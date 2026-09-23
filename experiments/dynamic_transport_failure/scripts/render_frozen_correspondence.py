"""Render frozen RNA with canonical feature identity and current render inputs.

Run this wrapper from the official RNA repository with its BPy environment.
It subclasses only at the adapter boundary and does not edit official source.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import h5py
import numpy as np
import torch as th

import bpy

OFFICIAL_ROOT = pathlib.Path(__file__).resolve().parents[3] / "external" / "relightable-neural-assets"
sys.path.insert(0, str(OFFICIAL_ROOT))

from rna import config, renderers  # noqa: E402
from utils import exr, ops  # noqa: E402


CANONICAL_AOV_FILE = "canonical_position0001"


class FrozenCorrespondenceRenderer(renderers.NeuralSurfaceHairRenderer):
    def __init__(
        self,
        *args,
        canonical_aabb_min: np.ndarray,
        canonical_aabb_max: np.ndarray,
        trace_file: pathlib.Path | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.canonical_aabb_min = th.tensor(canonical_aabb_min, dtype=th.float32)
        self.canonical_aabb_max = th.tensor(canonical_aabb_max, dtype=th.float32)
        self.trace_file = trace_file

    def trace(self, message: str) -> None:
        if self.trace_file is None:
            return
        with self.trace_file.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def render_features(self, resolution, prepass=False, frame=None):
        self.trace(f"render_features_start prepass={prepass} frame={frame}")
        super().render_features(resolution, prepass=prepass, frame=frame)
        # The official renderer creates the geometry buffers in its pre-pass,
        # then uses those same files after the direct-visibility render.  There
        # is no second feature render in the main pass, so capture the custom
        # AOV here as well.  (The initial version inverted this condition and
        # therefore left the canonical AOV outside tmprndr/.)
        if frame is not None and prepass:
            source = CANONICAL_AOV_FILE + ".exr"
            destination = renderers.TMP_RENDER_DIR + CANONICAL_AOV_FILE + "_" + str(frame) + ".exr"
            if not os.path.exists(source):
                raise FileNotFoundError(
                    f"{source} missing; prepare the blend with add_canonical_position_aov.py"
                )
            os.replace(source, destination)
            self.trace(f"canonical_aov_saved frame={frame}")

    def process_deep_buffers(self, mask, visibility=True, frame=999):
        """Use current explicit inputs but canonical AOV for TriPlane lookup."""
        self.trace(f"process_start frame={frame} pixels={int(mask.sum())}")
        prefix = renderers.TMP_RENDER_DIR
        canonical_pixels = th.from_numpy(exr.read(prefix + CANONICAL_AOV_FILE + "_" + str(frame) + ".exr"))
        tangent_pixels = th.from_numpy(exr.read(prefix + "tangent0001_" + str(frame) + ".exr"))
        normals_pixels = th.from_numpy(exr.read(prefix + "normal0001_" + str(frame) + ".exr"))
        camera_dir_pixels = th.from_numpy(exr.read(prefix + "camera_dir0001_" + str(frame) + ".exr"))
        uv_pixels = th.from_numpy(exr.read(prefix + "uv0001_" + str(frame) + ".exr"))
        if visibility:
            direct = th.from_numpy(exr.read(prefix + "diffuse_direct" + str(frame) + ".exr"))
            visibility_pixels = direct.sum(axis=-1, keepdims=True).reshape(-1).gt(0.0).float()
        else:
            visibility_pixels = th.ones(canonical_pixels.shape[0] * canonical_pixels.shape[1])
        canonical_pixels = canonical_pixels.reshape(-1, 3)
        tangent_pixels = tangent_pixels.reshape(-1, 3)
        normals_pixels = normals_pixels.reshape(-1, 3)
        camera_dir_pixels = camera_dir_pixels.reshape(-1, 3)
        uv_pixels = uv_pixels[..., :2].reshape(-1, 2)
        canonical_normalized = ops.normalize_positions(
            canonical_pixels, self.canonical_aabb_min, self.canonical_aabb_max
        )
        self.trace(f"process_ready frame={frame}")
        return (
            canonical_normalized[mask],
            None,
            camera_dir_pixels[mask],
            tangent_pixels[mask],
            normals_pixels[mask],
            canonical_pixels[mask],
            visibility_pixels[mask],
        )


def load_camera_json(path: pathlib.Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload["frames"]
    return (
        [frame["pl_pos"] for frame in frames],
        [frame["transform_matrix"] for frame in frames],
        frames[0]["pl_intensity"][0],
        payload["camera_angle_x"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--blend", type=pathlib.Path, required=True)
    parser.add_argument("--canonical-dataset", type=pathlib.Path, required=True)
    parser.add_argument("--camera-json", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path)
    parser.add_argument("--frames", type=int, nargs="+")
    parser.add_argument("--output-dir", type=pathlib.Path)
    parser.add_argument("--model", action="store_true")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trace-file", type=pathlib.Path)
    args = parser.parse_args()
    if not args.model and not args.reference:
        raise ValueError("select --model and/or --reference")
    conf = config.get_render_config(str(args.config.resolve()))
    if args.frames is not None:
        conf.frames = args.frames
    if args.output_dir is not None:
        conf.output_dir = str(args.output_dir.resolve())
    pathlib.Path(conf.output_dir).mkdir(parents=True, exist_ok=True)
    if args.trace_file is not None:
        args.trace_file.parent.mkdir(parents=True, exist_ok=True)
        args.trace_file.write_text("main_start\n", encoding="utf-8")
    with h5py.File(args.canonical_dataset.resolve(), "r") as dataset:
        aabb_min, aabb_max = dataset.attrs["aabb_min"], dataset.attrs["aabb_max"]
    renderer_params = dict(conf.renderer_params)
    renderer_params["blend_file"] = str(args.blend.resolve())
    if args.checkpoint is not None:
        renderer_params["checkpoint"] = str(args.checkpoint.resolve())
    renderer = FrozenCorrespondenceRenderer(
        conf,
        canonical_aabb_min=aabb_min,
        canonical_aabb_max=aabb_max,
        trace_file=args.trace_file,
        device=args.device,
        **renderer_params,
    )
    light_positions, camera_matrices, intensity, horizontal_fov = load_camera_json(args.camera_json.resolve())
    renderer.prepare(light_positions, camera_matrices, intensity, horizontal_fov)
    renderer.trace("prepare_done")
    if args.model:
        renderer.trace("model_render_start")
        renderer.render_animation_neural()
        renderer.trace("model_render_done")
    if args.reference:
        renderer.blender_render(conf.frames, conf.output_dir)


if __name__ == "__main__":
    main()
