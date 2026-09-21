"""Build human-readable PNG panels, an aggregate table, and sweep videos."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

import imageio.v2 as iio
import matplotlib
import numpy as np
from PIL import Image, ImageDraw

from metrics import foreground_rgb, read_exr, tonemap


def to_u8(image: np.ndarray) -> np.ndarray:
    return np.asarray(np.clip(image, 0.0, 1.0) * 255.0, dtype=np.uint8)


def load_pair(state_dir: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    reference_paths = sorted(state_dir.glob("ref_*.exr"))
    prediction_paths = sorted(state_dir.glob("rna_*.exr"))
    if len(reference_paths) != 1 or len(prediction_paths) != 1:
        raise ValueError(f"expected one ref/rna EXR pair in {state_dir}")
    reference, _ = foreground_rgb(read_exr(reference_paths[0]))
    prediction, _ = foreground_rgb(read_exr(prediction_paths[0]))
    return tonemap(reference), tonemap(prediction)


def labeled_panel(
    reference: np.ndarray,
    prediction: np.ndarray,
    state_label: str,
    prediction_label: str = "Frozen RNA",
) -> np.ndarray:
    absolute = np.mean(np.abs(reference - prediction), axis=-1)
    absolute_scale = max(float(np.quantile(absolute, 0.99)), 1e-6)
    heat = matplotlib.colormaps["magma"](
        np.clip(absolute / absolute_scale, 0.0, 1.0)
    )[..., :3]
    panel = np.concatenate((reference, prediction, heat), axis=1)
    header_height = max(28, reference.shape[0] // 24)
    canvas = Image.new("RGB", (panel.shape[1], panel.shape[0] + header_height), "black")
    canvas.paste(Image.fromarray(to_u8(panel)), (0, header_height))
    draw = ImageDraw.Draw(canvas)
    width = reference.shape[1]
    draw.text((8, 7), f"{state_label} — GT", fill="white")
    draw.text((width + 8, 7), prediction_label, fill="white")
    draw.text((2 * width + 8, 7), "Absolute error (state-scaled)", fill="white")
    return np.asarray(canvas)


def export_state(
    state_dir: pathlib.Path,
    output_dir: pathlib.Path,
    state_label: str,
    prediction_label: str = "Frozen RNA",
) -> np.ndarray:
    reference, prediction = load_pair(state_dir)
    absolute_rgb = np.abs(reference - prediction)
    absolute = np.mean(absolute_rgb, axis=-1)
    relative = np.mean(absolute_rgb / np.maximum(reference, 0.03), axis=-1)
    absolute_scale = max(float(np.quantile(absolute, 0.99)), 1e-6)
    relative_scale = max(float(np.quantile(relative, 0.99)), 1e-6)
    absolute_heat = matplotlib.colormaps["magma"](
        np.clip(absolute / absolute_scale, 0.0, 1.0)
    )[..., :3]
    relative_heat = matplotlib.colormaps["viridis"](
        np.clip(relative / relative_scale, 0.0, 1.0)
    )[..., :3]
    state_output = output_dir / state_label
    state_output.mkdir(parents=True, exist_ok=True)
    iio.imwrite(state_output / "reference_GT.png", to_u8(reference))
    iio.imwrite(state_output / "frozen_RNA.png", to_u8(prediction))
    iio.imwrite(state_output / "side_by_side.png", to_u8(np.concatenate((reference, prediction), axis=1)))
    iio.imwrite(state_output / "absolute_error.png", to_u8(absolute_heat))
    iio.imwrite(state_output / "relative_error.png", to_u8(relative_heat))
    panel = labeled_panel(reference, prediction, state_label, prediction_label)
    iio.imwrite(state_output / "GT_RNA_error.png", panel)
    return panel


def write_video(path: pathlib.Path, panels: list[np.ndarray], fps: int = 12) -> None:
    if not panels:
        return
    sequence = list(range(len(panels))) + list(range(len(panels) - 2, -1, -1))
    frames = []
    hold = max(1, fps)
    for index in sequence:
        frames.extend([panels[index]] * hold)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.mimwrite(path, frames, fps=fps, codec="libx264", quality=8, macro_block_size=2)


def aggregate_metrics(metrics_dir: pathlib.Path, output: pathlib.Path) -> None:
    records = []
    for path in sorted(metrics_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "psnr" in data:
            records.append(data)
    if not records:
        return
    preferred = [
        "asset", "state", "deformation", "psnr", "psnr_foreground", "ssim",
        "ssim_foreground", "lpips", "mean_absolute_rgb_error",
        "mean_absolute_rgb_error_foreground", "changed_visibility_fraction",
        "error_in_changed_region", "error_elsewhere", "visibility_comparison_mode",
        "visibility_matched_fraction", "visibility_match_p95_distance",
    ]
    fields = preferred + sorted(set().union(*(r.keys() for r in records)) - set(preferred))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=pathlib.Path)
    parser.add_argument("--group", required=True)
    parser.add_argument("--states", nargs="+", required=True)
    parser.add_argument("--video-name", required=True)
    parser.add_argument("--prediction-label", default="Frozen RNA")
    args = parser.parse_args()
    group_dir = args.root / args.group
    output_dir = args.root / "review" / args.group
    panels = [
        export_state(group_dir / state, output_dir, state, args.prediction_label)
        for state in args.states
    ]
    write_video(args.root / "videos" / args.video_name, panels)
    aggregate_metrics(args.root / "metrics", args.root / "metrics" / "all_metrics.csv")


if __name__ == "__main__":
    main()
