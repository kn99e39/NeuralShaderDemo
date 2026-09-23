"""Create a labeled reference-versus-frozen-RNA GIF across deformation states.

The GIF is a visual review artifact.  Quantitative results must continue to be
calculated from the original linear EXR files, never from this tonemapped GIF.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pyexr
from PIL import Image, ImageDraw


def tonemap(rgb: np.ndarray) -> np.ndarray:
    return np.clip(rgb / (1.0 + np.maximum(rgb, 0.0)), 0.0, 1.0)


def load(path: pathlib.Path) -> Image.Image:
    image = pyexr.open(str(path)).get().astype(np.float32)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"expected RGB(A) EXR: {path}")
    pixels = (tonemap(image[..., :3]) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def label(image: Image.Image, text: str) -> None:
    draw = ImageDraw.Draw(image)
    draw.text((12, 12), text, fill="black", stroke_width=3, stroke_fill="black")
    draw.text((12, 12), text, fill="white")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state-dir",
        type=pathlib.Path,
        action="append",
        required=True,
        help="State render directory, in display order (for example .../F0).",
    )
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--duration-ms", type=int, default=1200)
    args = parser.parse_args()
    if args.duration_ms <= 0:
        raise ValueError("duration must be positive")

    frames: list[Image.Image] = []
    for state_dir in args.state_dir:
        state_dir = state_dir.resolve()
        reference = load(state_dir / "ref_0.exr")
        rna = load(state_dir / "rna_0.exr")
        if reference.size != rna.size:
            raise ValueError(f"mismatched dimensions in {state_dir}")
        frame = Image.new("RGB", (reference.width * 2, reference.height))
        frame.paste(reference, (0, 0))
        frame.paste(rna, (reference.width, 0))
        label(frame, f"{state_dir.name}  reference")
        draw = ImageDraw.Draw(frame)
        draw.text(
            (reference.width + 12, 12),
            "frozen RNA",
            fill="black",
            stroke_width=3,
            stroke_fill="black",
        )
        draw.text((reference.width + 12, 12), "frozen RNA", fill="white")
        frames.append(frame)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )


if __name__ == "__main__":
    main()
