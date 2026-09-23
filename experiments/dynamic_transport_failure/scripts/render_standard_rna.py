"""Render an unmodified official RNA checkpoint with explicit CLI inputs."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

OFFICIAL_ROOT = pathlib.Path(__file__).resolve().parents[3] / "external" / "relightable-neural-assets"
sys.path.insert(0, str(OFFICIAL_ROOT))

from rna import config, renderers  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--blend", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--camera-json", type=pathlib.Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+")
    parser.add_argument("--output-dir", type=pathlib.Path)
    parser.add_argument("--model", action="store_true")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.model and not args.reference:
        raise ValueError("select --model and/or --reference")
    conf = config.get_render_config(str(args.config.resolve()))
    if args.frames is not None:
        conf.frames = args.frames
    if args.output_dir is not None:
        conf.output_dir = str(args.output_dir.resolve())
    pathlib.Path(conf.output_dir).mkdir(parents=True, exist_ok=True)
    params = dict(conf.renderer_params)
    params["blend_file"] = str(args.blend.resolve())
    params["checkpoint"] = str(args.checkpoint.resolve())
    renderer = renderers.NeuralSurfaceHairRenderer(conf, device=args.device, **params)
    payload = json.loads(args.camera_json.resolve().read_text(encoding="utf-8"))
    frames = payload["frames"]
    renderer.prepare(
        [frame["pl_pos"] for frame in frames],
        [frame["transform_matrix"] for frame in frames],
        frames[0]["pl_intensity"][0],
        payload["camera_angle_x"],
    )
    if args.model:
        renderer.render_animation_neural()
    if args.reference:
        renderer.blender_render(conf.frames, conf.output_dir)


if __name__ == "__main__":
    main()
