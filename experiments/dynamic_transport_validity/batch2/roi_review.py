"""Export predeclared localized error crops for Batch 2 deformation families."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import imageio.v3 as iio
import matplotlib
import numpy as np
from PIL import Image, ImageDraw

EXPERIMENT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from shared.metrics import foreground_rgb, read_exr, tonemap


# Normalized (left, top, right, bottom) regions fixed before final RNA renders.
# Each encloses the prospective fold/contact/compound-cavity rather than selecting
# a post-hoc high-error patch.
FAMILY_ROIS = {
    "A": (0.40, 0.27, 0.72, 0.75),
    "B": (0.42, 0.22, 0.60, 0.80),
    "C": (0.24, 0.22, 0.70, 0.76),
}


def psnr(reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    values = np.square(reference - prediction)[mask]
    return float(-10.0 * math.log10(max(float(np.mean(values)), 1e-12)))


def u8(image: np.ndarray) -> np.ndarray:
    return np.asarray(np.clip(image, 0.0, 1.0) * 255.0, dtype=np.uint8)


def pair(state_dir: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    refs = sorted(state_dir.glob("ref_*.exr"))
    rnas = sorted(state_dir.glob("rna_*.exr"))
    if len(refs) != 1 or len(rnas) != 1:
        raise ValueError(f"expected one ref/rna pair in {state_dir}")
    reference_raw = read_exr(refs[0])
    prediction_raw = read_exr(rnas[0])
    reference, ref_alpha = foreground_rgb(reference_raw)
    prediction, pred_alpha = foreground_rgb(prediction_raw)
    return tonemap(reference), tonemap(prediction), np.logical_or(ref_alpha, pred_alpha)


def roi_bounds(shape: tuple[int, int], family: str) -> tuple[int, int, int, int]:
    height, width = shape
    left, top, right, bottom = FAMILY_ROIS[family]
    return (
        round(left * width), round(top * height), round(right * width), round(bottom * height)
    )


def export_crop(
    output: pathlib.Path,
    family: str,
    state: str,
    reference: np.ndarray,
    prediction: np.ndarray,
    bounds: tuple[int, int, int, int],
) -> None:
    left, top, right, bottom = bounds
    ref = reference[top:bottom, left:right]
    pred = prediction[top:bottom, left:right]
    error = np.mean(np.abs(ref - pred), axis=-1)
    scale = max(float(np.quantile(error, 0.99)), 1e-6)
    heat = matplotlib.colormaps["magma"](np.clip(error / scale, 0.0, 1.0))[..., :3]
    content = np.concatenate((ref, pred, heat), axis=1)
    header = max(28, ref.shape[0] // 12)
    canvas = Image.new("RGB", (content.shape[1], content.shape[0] + header), "black")
    canvas.paste(Image.fromarray(u8(content)), (0, header))
    draw = ImageDraw.Draw(canvas)
    width = ref.shape[1]
    draw.text((6, 6), f"{family}{state} ROI — GT", fill="white")
    draw.text((width + 6, 6), "Frozen RNA", fill="white")
    draw.text((2 * width + 6, 6), "Absolute error", fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output, np.asarray(canvas))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--family", choices=sorted(FAMILY_ROIS), required=True)
    parser.add_argument("--states", nargs="+", required=True)
    args = parser.parse_args()

    records = []
    for state in args.states:
        reference, prediction, foreground = pair(args.root / args.family / state)
        left, top, right, bottom = roi_bounds(reference.shape[:2], args.family)
        roi = np.zeros(foreground.shape, dtype=bool)
        roi[top:bottom, left:right] = True
        roi_foreground = np.logical_and(roi, foreground)
        rest_foreground = np.logical_and(~roi, foreground)
        absolute = np.mean(np.abs(reference - prediction), axis=-1)
        record = {
            "family": args.family,
            "state": state,
            "roi_normalized": FAMILY_ROIS[args.family],
            "roi_bounds_pixels": [left, top, right, bottom],
            "roi_foreground_pixels": int(np.sum(roi_foreground)),
            "rest_foreground_pixels": int(np.sum(rest_foreground)),
            "roi_mae": float(np.mean(absolute[roi_foreground])),
            "rest_mae": float(np.mean(absolute[rest_foreground])),
            "roi_psnr": psnr(reference, prediction, roi_foreground),
            "rest_psnr": psnr(reference, prediction, rest_foreground),
        }
        records.append(record)
        export_crop(
            args.root / "review" / args.family / state / "ROI_GT_RNA_error.png",
            args.family,
            state,
            reference,
            prediction,
            (left, top, right, bottom),
        )
    metrics_dir = args.root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"batch2_{args.family}_roi.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
