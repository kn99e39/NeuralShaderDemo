"""Create a deterministic GT/RNA alternating review GIF from EXR frame pairs.

This is a review artifact only: metrics are always calculated from the source
linear EXRs, never from this tonemapped animation.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pyexr
from PIL import Image


def tonemap(rgb: np.ndarray) -> np.ndarray:
    return np.clip(rgb / (1.0 + np.maximum(rgb, 0.0)), 0.0, 1.0)


def load(path: pathlib.Path) -> Image.Image:
    image = pyexr.open(str(path)).get().astype(np.float32)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"expected RGB(A) EXR: {path}")
    pixels = (tonemap(image[..., :3]) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=pathlib.Path, required=True)
    parser.add_argument("--gt-directory", type=pathlib.Path, help="Defaults to --directory.")
    parser.add_argument("--rna-directory", type=pathlib.Path, help="Defaults to --directory.")
    parser.add_argument("--gt-prefix", default="ref_")
    parser.add_argument("--rna-prefix", default="rna_")
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--duration-ms", type=int, default=500)
    args = parser.parse_args()
    if args.duration_ms <= 0:
        raise ValueError("duration must be positive")
    directory = args.directory.resolve()
    gt_directory = args.gt_directory.resolve() if args.gt_directory else directory
    rna_directory = args.rna_directory.resolve() if args.rna_directory else directory
    images: list[Image.Image] = []
    for frame in args.frames:
        images.extend(
            [
                load(gt_directory / f"{args.gt_prefix}{frame}.exr"),
                load(rna_directory / f"{args.rna_prefix}{frame}.exr"),
            ]
        )
    if not images:
        raise ValueError("at least one frame is required")
    if any(image.size != images[0].size for image in images):
        raise ValueError("all review images must have the same resolution")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=args.duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )


if __name__ == "__main__":
    main()
