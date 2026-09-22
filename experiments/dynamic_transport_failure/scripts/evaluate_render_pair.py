"""Measure and review one full-resolution GT/RNA EXR pair."""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pyexr
from PIL import Image
from scipy.ndimage import gaussian_filter


def load_exr(path: pathlib.Path) -> np.ndarray:
    image = pyexr.open(str(path)).get().astype(np.float32)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"expected RGB(A) EXR: {path}")
    if image.shape[2] == 3:
        image = np.dstack([image, np.ones(image.shape[:2], dtype=np.float32)])
    return image[..., :4]


def ssim(reference: np.ndarray, estimate: np.ndarray, data_range: float) -> float:
    """Channel-mean Gaussian-window SSIM for linear RGB arrays."""
    k1, k2 = 0.01, 0.03
    c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2
    values = []
    for channel in range(3):
        x, y = reference[..., channel], estimate[..., channel]
        mux, muy = gaussian_filter(x, 1.5), gaussian_filter(y, 1.5)
        varx = gaussian_filter(x * x, 1.5) - mux * mux
        vary = gaussian_filter(y * y, 1.5) - muy * muy
        cov = gaussian_filter(x * y, 1.5) - mux * muy
        score = ((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux * mux + muy * muy + c1) * (varx + vary + c2))
        values.append(float(np.mean(score)))
    return float(np.mean(values))


def tonemap(rgb: np.ndarray) -> np.ndarray:
    return np.clip(rgb / (1.0 + np.maximum(rgb, 0.0)), 0.0, 1.0)


def save_preview(path: pathlib.Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(path)


def parse_roi(value: str) -> tuple[str, tuple[int, int, int, int]]:
    name, coordinates = value.split(":", 1)
    x0, y0, x1, y1 = (int(item) for item in coordinates.split(","))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"invalid ROI: {value}")
    return name, (x0, y0, x1, y1)


def measure(reference: np.ndarray, estimate: np.ndarray) -> dict:
    if reference.shape != estimate.shape:
        raise ValueError(f"image shape mismatch: {reference.shape} vs {estimate.shape}")
    mask = (reference[..., 3] > 0.0) | (estimate[..., 3] > 0.0)
    if not bool(mask.any()):
        raise ValueError("both renders have empty alpha")
    gt, rna = reference[..., :3], estimate[..., :3]
    error = np.abs(gt - rna)
    valid_error = error[mask]
    peak = max(1.0, float(np.max(np.abs(gt[mask]))), float(np.max(np.abs(rna[mask]))))
    mse = float(np.mean((gt[mask] - rna[mask]) ** 2))
    return {
        "pixels": int(mask.sum()),
        "mae": float(np.mean(valid_error)),
        "psnr": None if mse == 0.0 else float(20.0 * np.log10(peak) - 10.0 * np.log10(mse)),
        "psnr_is_infinite": mse == 0.0,
        "ssim": ssim(gt * mask[..., None], rna * mask[..., None], peak),
        "data_range": peak,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=pathlib.Path, required=True)
    parser.add_argument("--rna", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--roi", action="append", default=[], help="name:x0,y0,x1,y1; repeatable")
    args = parser.parse_args()
    gt, rna = load_exr(args.gt.resolve()), load_exr(args.rna.resolve())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    error = np.abs(gt[..., :3] - rna[..., :3])
    pyexr.write(str(output / f"{args.label}_absolute_error.exr"), error)
    preview = np.concatenate([tonemap(gt[..., :3]), tonemap(rna[..., :3])], axis=1)
    save_preview(output / f"{args.label}_gt_rna.png", preview)
    save_preview(output / f"{args.label}_absolute_error.png", tonemap(error * 4.0))
    metrics = {"label": args.label, "gt": str(args.gt.resolve()), "rna": str(args.rna.resolve()), "full_frame": measure(gt, rna), "lpips": "not computed: no verified local LPIPS model was supplied"}
    for raw_roi in args.roi:
        name, (x0, y0, x1, y1) = parse_roi(raw_roi)
        if not (0 <= x0 < x1 <= gt.shape[1] and 0 <= y0 < y1 <= gt.shape[0]):
            raise ValueError(f"ROI outside image bounds: {raw_roi}")
        metrics.setdefault("rois", {})[name] = measure(gt[y0:y1, x0:x1], rna[y0:y1, x0:x1])
        roi_preview = np.concatenate([tonemap(gt[y0:y1, x0:x1, :3]), tonemap(rna[y0:y1, x0:x1, :3])], axis=1)
        save_preview(output / f"{args.label}_{name}_gt_rna.png", roi_preview)
        save_preview(output / f"{args.label}_{name}_absolute_error.png", tonemap(error[y0:y1, x0:x1] * 4.0))
    (output / f"{args.label}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
