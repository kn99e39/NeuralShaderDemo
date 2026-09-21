"""Quantitative accounting and review exports for RNA/GT image pairs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib

import imageio.v3 as iio
import matplotlib
import numpy as np
import pyexr
from scipy.ndimage import gaussian_filter, zoom
from scipy.spatial import cKDTree


_LPIPS_CACHE = None


def read_exr(path: pathlib.Path) -> np.ndarray:
    image = np.asarray(pyexr.read(str(path)), dtype=np.float32)
    if image.ndim == 2:
        image = image[..., None]
    return image


def tonemap(image: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    image = np.clip(image, 0.0, None)
    return np.power(image / (1.0 + image), 1.0 / gamma)


def foreground_rgb(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb = image[..., :3]
    if image.shape[-1] >= 4:
        alpha = image[..., 3] > 1e-4
    else:
        alpha = np.any(np.abs(rgb) > 1e-8, axis=-1)
    return rgb, alpha


def psnr(reference: np.ndarray, prediction: np.ndarray, mask=None) -> float:
    squared = np.square(reference - prediction)
    if mask is not None:
        squared = squared[mask]
    mse = float(np.mean(squared)) if squared.size else float("nan")
    return float(-10.0 * math.log10(max(mse, 1e-12)))


def ssim_map(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Standard Gaussian-window SSIM for unit-range RGB images."""

    sigma = 1.5
    c1, c2 = 0.01**2, 0.03**2
    mu_x = gaussian_filter(reference, (sigma, sigma, 0), mode="reflect")
    mu_y = gaussian_filter(prediction, (sigma, sigma, 0), mode="reflect")
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    variance_x = gaussian_filter(reference * reference, (sigma, sigma, 0), mode="reflect") - mu_x2
    variance_y = gaussian_filter(prediction * prediction, (sigma, sigma, 0), mode="reflect") - mu_y2
    covariance = gaussian_filter(reference * prediction, (sigma, sigma, 0), mode="reflect") - mu_xy
    numerator = (2.0 * mu_xy + c1) * (2.0 * covariance + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (variance_x + variance_y + c2)
    return np.mean(numerator / np.maximum(denominator, 1e-12), axis=-1)


def resize_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == target_shape:
        return mask
    factors = (target_shape[0] / mask.shape[0], target_shape[1] / mask.shape[1])
    return zoom(mask.astype(np.float32), factors, order=0) > 0.5


def resize_vector_nearest(
    image: np.ndarray, target_shape: tuple[int, int]
) -> np.ndarray:
    """Select pixel-centre samples without averaging persistent identities."""

    if image.shape[:2] == target_shape:
        return image
    rows = np.minimum(
        ((np.arange(target_shape[0]) + 0.5) * image.shape[0] / target_shape[0]).astype(int),
        image.shape[0] - 1,
    )
    columns = np.minimum(
        ((np.arange(target_shape[1]) + 0.5) * image.shape[1] / target_shape[1]).astype(int),
        image.shape[1] - 1,
    )
    return image[rows[:, None], columns[None, :]]


def visibility_mask(path: pathlib.Path, target_shape: tuple[int, int]) -> np.ndarray:
    visibility = read_exr(path)[..., :3]
    mask = np.sum(np.abs(visibility), axis=-1) > 1e-8
    return resize_mask(mask, target_shape)


def aligned_visibility_masks(args, foreground: np.ndarray) -> dict | None:
    """Compare light visibility at matched canonical surface identities.

    The deformation moves a material point to another pixel, so an image-space
    XOR is not a valid surface attribution mask.  When canonical-position AOVs
    are supplied, nearest-neighbour matching is performed in RNA's normalized
    canonical lookup space and unmatched/newly-camera-visible samples are
    excluded from the visibility statistic.
    """

    if not (args.canonical_visibility and args.current_visibility):
        return None
    canonical_visibility = visibility_mask(args.canonical_visibility, foreground.shape)
    current_visibility = visibility_mask(args.current_visibility, foreground.shape)

    if not (args.canonical_position_reference and args.canonical_position_current):
        matched = foreground.copy()
        changed = np.logical_and(
            np.logical_xor(canonical_visibility, current_visibility), matched
        )
        return {
            "mode": "image-space-fallback",
            "matched": matched,
            "changed": changed,
            "newly_shadowed": np.logical_and.reduce(
                (canonical_visibility, ~current_visibility, matched)
            ),
            "newly_exposed": np.logical_and.reduce(
                (~canonical_visibility, current_visibility, matched)
            ),
            "matched_fraction": 1.0,
            "median_distance": None,
            "p95_distance": None,
        }

    reference_positions = resize_vector_nearest(
        read_exr(args.canonical_position_reference)[..., :3], foreground.shape
    )
    current_positions = resize_vector_nearest(
        read_exr(args.canonical_position_current)[..., :3], foreground.shape
    )

    reference_valid = np.any(np.abs(reference_positions) > 1e-8, axis=-1)
    current_valid = np.logical_and(
        foreground, np.any(np.abs(current_positions) > 1e-8, axis=-1)
    )
    reference_points = reference_positions[reference_valid]
    if reference_points.size == 0 or not np.any(current_valid):
        raise ValueError("no valid canonical-position samples for visibility alignment")
    tree = cKDTree(reference_points)
    distances, indices = tree.query(current_positions[current_valid], workers=-1)
    accepted = distances <= args.correspondence_tolerance

    matched = np.zeros(foreground.shape, dtype=bool)
    matched[current_valid] = accepted
    canonical_at_current = np.zeros(foreground.shape, dtype=bool)
    source_visibility = canonical_visibility[reference_valid]
    current_flat_indices = np.flatnonzero(current_valid)
    canonical_at_current.flat[current_flat_indices[accepted]] = source_visibility[
        indices[accepted]
    ]
    changed = np.logical_and(
        np.logical_xor(canonical_at_current, current_visibility), matched
    )
    return {
        "mode": "canonical-surface-nearest-neighbour",
        "matched": matched,
        "changed": changed,
        "newly_shadowed": np.logical_and.reduce(
            (canonical_at_current, ~current_visibility, matched)
        ),
        "newly_exposed": np.logical_and.reduce(
            (~canonical_at_current, current_visibility, matched)
        ),
        "matched_fraction": float(np.mean(matched[foreground])),
        "median_distance": float(np.median(distances[accepted])) if np.any(accepted) else None,
        "p95_distance": float(np.quantile(distances[accepted], 0.95)) if np.any(accepted) else None,
    }


def save_attribution_masks(output_dir: pathlib.Path, masks: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("matched", "changed", "newly_shadowed", "newly_exposed"):
        image = np.asarray(masks[name], dtype=np.uint8) * 255
        iio.imwrite(output_dir / f"{name}.png", image)
    categories = np.zeros((*masks["changed"].shape, 3), dtype=np.uint8)
    categories[masks["newly_shadowed"]] = (255, 80, 40)
    categories[masks["newly_exposed"]] = (40, 180, 255)
    iio.imwrite(output_dir / "visibility_change_categories.png", categories)


def optional_lpips(reference: np.ndarray, prediction: np.ndarray) -> float | None:
    global _LPIPS_CACHE
    try:
        import lpips
        import torch
    except ImportError:
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if _LPIPS_CACHE is None:
        _LPIPS_CACHE = lpips.LPIPS(net="alex").to(device)
    metric = _LPIPS_CACHE
    ref = torch.from_numpy(reference.transpose(2, 0, 1)).unsqueeze(0).to(device)
    pred = torch.from_numpy(prediction.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        return float(metric(2.0 * ref - 1.0, 2.0 * pred - 1.0).item())


def save_review(
    output: pathlib.Path,
    reference: np.ndarray,
    prediction: np.ndarray,
    changed_visibility: np.ndarray | None,
) -> None:
    error = np.mean(np.abs(reference - prediction), axis=-1)
    scale = max(float(np.quantile(error, 0.99)), 1e-6)
    heat = matplotlib.colormaps["magma"](np.clip(error / scale, 0.0, 1.0))[..., :3]
    if changed_visibility is not None:
        overlay = reference.copy()
        boundary = np.logical_xor(
            changed_visibility,
            np.logical_and.reduce(
                [
                    np.roll(changed_visibility, 1, axis=0),
                    np.roll(changed_visibility, -1, axis=0),
                    np.roll(changed_visibility, 1, axis=1),
                    np.roll(changed_visibility, -1, axis=1),
                ]
            ),
        )
        overlay[boundary] = (0.0, 1.0, 1.0)
    else:
        overlay = reference
    panel = np.concatenate((reference, prediction, heat, overlay), axis=1)
    iio.imwrite(output, np.asarray(np.clip(panel, 0.0, 1.0) * 255.0, dtype=np.uint8))


def evaluate(args) -> dict:
    ref_raw = read_exr(args.reference)
    pred_raw = read_exr(args.prediction)
    ref_rgb, ref_alpha = foreground_rgb(ref_raw)
    pred_rgb, pred_alpha = foreground_rgb(pred_raw)
    if ref_rgb.shape != pred_rgb.shape:
        raise ValueError(f"image shape mismatch: {ref_rgb.shape} vs {pred_rgb.shape}")
    foreground = np.logical_or(ref_alpha, pred_alpha)
    reference = tonemap(ref_rgb)
    prediction = tonemap(pred_rgb)
    absolute = np.mean(np.abs(reference - prediction), axis=-1)
    ssim = ssim_map(reference, prediction)

    visibility = aligned_visibility_masks(args, foreground)
    changed = visibility["changed"] if visibility is not None else None
    matched = visibility["matched"] if visibility is not None else foreground
    if visibility is not None and args.mask_output_dir:
        save_attribution_masks(args.mask_output_dir, visibility)

    result = {
        "asset": args.asset,
        "state": args.state,
        "deformation": args.deformation,
        "psnr": psnr(reference, prediction),
        "psnr_foreground": psnr(reference, prediction, foreground),
        "ssim": float(np.mean(ssim)),
        "ssim_foreground": float(np.mean(ssim[foreground])),
        "lpips": optional_lpips(reference, prediction) if args.lpips else None,
        "mean_absolute_rgb_error": float(np.mean(absolute)),
        "mean_absolute_rgb_error_foreground": float(np.mean(absolute[foreground])),
        "changed_visibility_fraction": (
            float(np.mean(changed[matched])) if changed is not None and np.any(matched) else None
        ),
        "error_in_changed_region": (
            float(np.mean(absolute[changed])) if changed is not None and np.any(changed) else None
        ),
        "error_elsewhere": (
            float(np.mean(absolute[np.logical_and(matched, ~changed)]))
            if changed is not None and np.any(np.logical_and(matched, ~changed))
            else None
        ),
        "visibility_comparison_mode": visibility["mode"] if visibility else None,
        "visibility_matched_fraction": visibility["matched_fraction"] if visibility else None,
        "visibility_match_median_distance": visibility["median_distance"] if visibility else None,
        "visibility_match_p95_distance": visibility["p95_distance"] if visibility else None,
        "reference": str(args.reference),
        "prediction": str(args.prediction),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    save_review(args.review, reference, prediction, changed)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--deformation", required=True)
    parser.add_argument("--reference", required=True, type=pathlib.Path)
    parser.add_argument("--prediction", required=True, type=pathlib.Path)
    parser.add_argument("--canonical-visibility", type=pathlib.Path)
    parser.add_argument("--current-visibility", type=pathlib.Path)
    parser.add_argument("--canonical-position-reference", type=pathlib.Path)
    parser.add_argument("--canonical-position-current", type=pathlib.Path)
    parser.add_argument("--correspondence-tolerance", type=float, default=0.003)
    parser.add_argument("--mask-output-dir", type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--review", required=True, type=pathlib.Path)
    parser.add_argument("--lpips", action="store_true")
    args = parser.parse_args()
    result = evaluate(args)
    writer = csv.DictWriter(sys.stdout, fieldnames=result.keys())
    writer.writeheader()
    writer.writerow(result)


if __name__ == "__main__":
    import sys

    main()
