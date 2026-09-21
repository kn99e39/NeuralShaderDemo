"""Build a single GT/frozen/refit/error comparison panel for the G4 control."""

from __future__ import annotations

import argparse
import pathlib

import imageio.v3 as iio
import matplotlib
import numpy as np
from PIL import Image, ImageDraw

from metrics import foreground_rgb, read_exr, tonemap


def load(path: pathlib.Path) -> np.ndarray:
    image, _ = foreground_rgb(read_exr(path))
    return tonemap(image)


def to_u8(image: np.ndarray) -> np.ndarray:
    return np.asarray(np.clip(image, 0.0, 1.0) * 255.0, dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=pathlib.Path)
    parser.add_argument("--frozen", required=True, type=pathlib.Path)
    parser.add_argument("--refit", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    reference = load(args.reference)
    frozen = load(args.frozen)
    refit = load(args.refit)
    frozen_error = np.mean(np.abs(reference - frozen), axis=-1)
    refit_error = np.mean(np.abs(reference - refit), axis=-1)
    shared_scale = max(float(np.quantile(frozen_error, 0.99)), 1e-6)
    colormap = matplotlib.colormaps["magma"]
    frozen_heat = colormap(np.clip(frozen_error / shared_scale, 0.0, 1.0))[..., :3]
    refit_heat = colormap(np.clip(refit_error / shared_scale, 0.0, 1.0))[..., :3]

    panel = np.concatenate((reference, frozen, refit, frozen_heat, refit_heat), axis=1)
    height, width = reference.shape[:2]
    header_height = max(32, height // 20)
    canvas = Image.new("RGB", (panel.shape[1], height + header_height), "black")
    canvas.paste(Image.fromarray(to_u8(panel)), (0, header_height))
    draw = ImageDraw.Draw(canvas)
    labels = ("G4 GT", "Frozen G0 RNA", "RNA refit on G4", "Frozen error", "Refit error")
    for index, label in enumerate(labels):
        draw.text((index * width + 8, 8), label, fill="white")
    draw.text((3 * width + 8, header_height - 13), "shared error scale", fill="white")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.output, np.asarray(canvas))


if __name__ == "__main__":
    main()
