"""Measure F0-to-F1 direct-visibility transitions for the Rain cyan artifact.

The correspondence is deliberately limited to a nearest canonical-position
match within the stable scarf/top provenance AOV.  It is an AOV-assisted
screen-space diagnostic, not a triangle/UV correspondence proof.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pyexr
from PIL import Image
from scipy.spatial import cKDTree


def load(path: pathlib.Path) -> np.ndarray:
    return pyexr.open(str(path)).get().astype(np.float32)


def alpha_or_ones(image: np.ndarray) -> np.ndarray:
    return image[..., 3] > 0.0 if image.shape[-1] > 3 else np.ones(image.shape[:2], dtype=bool)


def surface_codes(provenance: np.ndarray) -> np.ndarray:
    """Return 0=scarf, 1=top, -1=unlabelled from the stable RGB AOV."""
    rgb = provenance[..., :3]
    codes = np.full(rgb.shape[:2], -1, dtype=np.int8)
    labelled = np.max(rgb, axis=-1) > 0.1
    codes[labelled & (rgb[..., 0] >= rgb[..., 1])] = 0
    codes[labelled & (rgb[..., 1] > rgb[..., 0])] = 1
    return codes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--f0-reference", type=pathlib.Path, required=True)
    parser.add_argument("--f0-rna", type=pathlib.Path, required=True)
    parser.add_argument("--f0-canonical", type=pathlib.Path, required=True)
    parser.add_argument("--f0-provenance", type=pathlib.Path, required=True)
    parser.add_argument("--f0-direct", type=pathlib.Path, required=True)
    parser.add_argument("--f1-reference", type=pathlib.Path, required=True)
    parser.add_argument("--f1-rna", type=pathlib.Path, required=True)
    parser.add_argument("--f1-canonical", type=pathlib.Path, required=True)
    parser.add_argument("--f1-provenance", type=pathlib.Path, required=True)
    parser.add_argument("--f1-direct", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--match-radius", type=float, default=0.005)
    parser.add_argument("--direct-epsilon", type=float, default=0.0)
    args = parser.parse_args()
    if args.match_radius <= 0.0:
        raise ValueError("match radius must be positive")

    f0_ref, f0_rna, f0_can, f0_prov, f0_direct = (load(path) for path in (
        args.f0_reference, args.f0_rna, args.f0_canonical, args.f0_provenance, args.f0_direct
    ))
    f1_ref, f1_rna, f1_can, f1_prov, f1_direct = (load(path) for path in (
        args.f1_reference, args.f1_rna, args.f1_canonical, args.f1_provenance, args.f1_direct
    ))
    shape = f1_ref.shape[:2]
    if any(image.shape[:2] != shape for image in (f0_ref, f0_rna, f0_can, f0_prov, f0_direct, f1_rna, f1_can, f1_prov, f1_direct)):
        raise ValueError("all F0/F1 input images must share a resolution")

    f1_valid = alpha_or_ones(f1_ref) | alpha_or_ones(f1_rna)
    error = np.abs(f1_ref[..., :3] - f1_rna[..., :3]).mean(axis=-1)
    error_p95 = float(np.quantile(error[f1_valid], 0.95))
    cyan = (
        f1_valid
        & (f1_rna[..., 1] > 0.25)
        & (f1_rna[..., 2] > 0.25)
        & (f1_rna[..., 0] < 0.6 * np.minimum(f1_rna[..., 1], f1_rna[..., 2]))
        & (error >= error_p95)
    )
    if not cyan.any():
        raise ValueError("cyan artifact rule selected no F1 pixels")

    f0_valid = alpha_or_ones(f0_ref) | alpha_or_ones(f0_rna)
    f0_surface, f1_surface = surface_codes(f0_prov), surface_codes(f1_prov)
    f0_points = f0_can[..., :3].reshape(-1, 3)
    f0_valid_flat = f0_valid.reshape(-1)
    f0_surface_flat = f0_surface.reshape(-1)
    f1_points = f1_can[..., :3][cyan]
    f1_surface_selected = f1_surface[cyan]
    if np.any(f1_surface_selected < 0):
        raise ValueError("cyan artifact contains pixels without stable provenance")

    trees: dict[int, tuple[cKDTree, np.ndarray]] = {}
    for code in (0, 1):
        indices = np.flatnonzero(f0_valid_flat & (f0_surface_flat == code))
        if not len(indices):
            raise ValueError(f"F0 has no valid provenance pixels for surface {code}")
        trees[code] = (cKDTree(f0_points[indices]), indices)

    nearest_distance = np.full(len(f1_points), np.inf, dtype=np.float32)
    nearest_flat_index = np.full(len(f1_points), -1, dtype=np.int64)
    for code in (0, 1):
        selected = f1_surface_selected == code
        distances, local_indices = trees[code][0].query(f1_points[selected])
        nearest_distance[selected] = distances.astype(np.float32)
        nearest_flat_index[selected] = trees[code][1][local_indices]
    matched = nearest_distance <= args.match_radius

    f0_direct_flat = f0_direct[..., :3].sum(axis=-1).reshape(-1)
    f1_direct_selected = f1_direct[..., :3].sum(axis=-1)[cyan]
    f0_visible = np.zeros(len(f1_points), dtype=bool)
    f0_visible[matched] = f0_direct_flat[nearest_flat_index[matched]] > args.direct_epsilon
    f1_visible = f1_direct_selected > args.direct_epsilon
    transitions = {
        "visible_to_visible": matched & f0_visible & f1_visible,
        "visible_to_occluded": matched & f0_visible & ~f1_visible,
        "occluded_to_visible": matched & ~f0_visible & f1_visible,
        "occluded_to_occluded": matched & ~f0_visible & ~f1_visible,
    }

    visual = np.zeros((*shape, 3), dtype=np.uint8)
    flat = visual.reshape(-1, 3)
    cyan_indices = np.flatnonzero(cyan.reshape(-1))
    colors = {
        "visible_to_visible": (0, 220, 0),
        "visible_to_occluded": (255, 220, 0),
        "occluded_to_visible": (0, 160, 255),
        "occluded_to_occluded": (230, 0, 0),
    }
    for name, mask in transitions.items():
        flat[cyan_indices[mask]] = colors[name]
    flat[cyan_indices[~matched]] = (255, 255, 255)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(visual, mode="RGB").save(output / "F1_cyan_artifact_F0_to_F1_direct_visibility.png")
    payload = {
        "label": "RAIN F0 TO F1 SAME-SURFACE DIRECT-VISIBILITY DIAGNOSTIC",
        "selection": {
            "cyan_rule": "RNA G,B > 0.25; RNA R < 0.6*min(G,B); mean absolute RGB error >= F1 valid-pixel p95",
            "error_p95": error_p95,
            "artifact_pixels": int(cyan.sum()),
        },
        "correspondence": {
            "method": "F1 canonical-position nearest-neighbour into F0, constrained by stable scarf/top provenance AOV",
            "not_a_triangle_or_uv_correspondence": True,
            "match_radius": args.match_radius,
            "matched_pixels": int(matched.sum()),
            "unmatched_pixels": int((~matched).sum()),
            "nearest_distance_median": float(np.median(nearest_distance)),
            "nearest_distance_p95": float(np.quantile(nearest_distance, 0.95)),
        },
        "visibility": {
            "definition": "sum(RGB diffuse_direct) > direct_epsilon",
            "direct_epsilon": args.direct_epsilon,
            "f1_visible_artifact_pixels": int(f1_visible.sum()),
            "f1_occluded_artifact_pixels": int((~f1_visible).sum()),
            "transitions_among_matched": {name: int(mask.sum()) for name, mask in transitions.items()},
        },
        "visual_legend": {
            "green": "matched: F0 visible -> F1 visible",
            "yellow": "matched: F0 visible -> F1 occluded",
            "blue": "matched: F0 occluded -> F1 visible",
            "red": "matched: F0 occluded -> F1 occluded",
            "white": "no same-surface canonical F0 match within radius",
        },
    }
    (output / "F1_cyan_artifact_F0_to_F1_direct_visibility.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
