"""Measure GT direct-light visibility transitions along a locked Rain R path.

No neural checkpoint is loaded.  Points are paired only by a nearest
canonical-position match within the stable scarf/top provenance AOV, so the
result is a geometry/direct-light diagnostic rather than exact mesh tracking.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pyexr
from scipy.spatial import cKDTree


def load(path: pathlib.Path) -> np.ndarray:
    return pyexr.open(str(path)).get().astype(np.float32)


def surface_codes(provenance: np.ndarray) -> np.ndarray:
    rgb = provenance[..., :3]
    result = np.full(rgb.shape[:2], -1, dtype=np.int8)
    labelled = np.max(rgb, axis=-1) > 0.1
    result[labelled & (rgb[..., 0] >= rgb[..., 1])] = 0
    result[labelled & (rgb[..., 1] > rgb[..., 0])] = 1
    return result


def transition_summary(anchor_visible: np.ndarray, current_visible: np.ndarray, matched: np.ndarray) -> dict:
    return {
        "matched": int(matched.sum()),
        "unmatched": int((~matched).sum()),
        "visible_to_visible": int((matched & anchor_visible & current_visible).sum()),
        "visible_to_occluded": int((matched & anchor_visible & ~current_visible).sum()),
        "occluded_to_visible": int((matched & ~anchor_visible & current_visible).sum()),
        "occluded_to_occluded": int((matched & ~anchor_visible & ~current_visible).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state",
        action="append",
        nargs=4,
        metavar=("NAME", "CANONICAL_EXR", "PROVENANCE_EXR", "DIRECT_EXR"),
        required=True,
        help="repeat for each ordered state; the first state is the canonical anchor",
    )
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--match-radius", type=float, default=0.005)
    parser.add_argument("--direct-epsilon", type=float, default=0.0)
    args = parser.parse_args()
    if args.match_radius <= 0.0:
        raise ValueError("match radius must be positive")
    if len(args.state) < 2:
        raise ValueError("supply an anchor and at least one comparison state")

    prepared = []
    for name, canonical_path, provenance_path, direct_path in args.state:
        canonical, provenance, direct = (load(pathlib.Path(path)) for path in (canonical_path, provenance_path, direct_path))
        if canonical.shape[:2] != provenance.shape[:2] or canonical.shape[:2] != direct.shape[:2]:
            raise ValueError(f"AOV resolution mismatch for {name}")
        codes = surface_codes(provenance)
        valid = (codes >= 0) & np.isfinite(canonical[..., :3]).all(axis=-1)
        prepared.append({"name": name, "canonical": canonical[..., :3], "codes": codes, "visible": direct[..., :3].sum(axis=-1) > args.direct_epsilon, "valid": valid})

    anchor = prepared[0]
    anchor_indices = np.flatnonzero(anchor["valid"].reshape(-1))
    anchor_points = anchor["canonical"].reshape(-1, 3)[anchor_indices]
    anchor_codes = anchor["codes"].reshape(-1)[anchor_indices]
    anchor_visible = anchor["visible"].reshape(-1)[anchor_indices]
    output = {
        "label": "RAIN R TRAJECTORY GT DIRECT-VISIBILITY DIAGNOSTIC",
        "regime": "GT geometry/direct-light only; no neural checkpoint loaded",
        "correspondence": {
            "anchor": anchor["name"],
            "method": "anchor canonical-position nearest-neighbour constrained by stable scarf/top provenance AOV",
            "not_a_triangle_or_uv_correspondence": True,
            "match_radius": args.match_radius,
        },
        "visibility_definition": "sum(RGB diffuse_direct) > direct_epsilon",
        "direct_epsilon": args.direct_epsilon,
        "states": {},
    }
    for current in prepared:
        current_points = current["canonical"].reshape(-1, 3)
        current_codes = current["codes"].reshape(-1)
        current_valid = current["valid"].reshape(-1)
        nearest_distance = np.full(len(anchor_points), np.inf, dtype=np.float32)
        nearest_index = np.full(len(anchor_points), -1, dtype=np.int64)
        for code in (0, 1):
            anchor_mask = anchor_codes == code
            candidates = np.flatnonzero(current_valid & (current_codes == code))
            if not len(candidates):
                continue
            distances, local_indices = cKDTree(current_points[candidates]).query(anchor_points[anchor_mask])
            nearest_distance[anchor_mask] = distances.astype(np.float32)
            nearest_index[anchor_mask] = candidates[local_indices]
        matched = nearest_distance <= args.match_radius
        current_visible = np.zeros(len(anchor_points), dtype=bool)
        current_visible[matched] = current["visible"].reshape(-1)[nearest_index[matched]]
        per_surface = {}
        for code, label in ((0, "scarf"), (1, "top")):
            subset = anchor_codes == code
            per_surface[label] = transition_summary(anchor_visible[subset], current_visible[subset], matched[subset])
        output["states"][current["name"]] = {
            "nearest_distance_median": float(np.median(nearest_distance)),
            "nearest_distance_p95": float(np.quantile(nearest_distance, 0.95)),
            "all_surfaces": transition_summary(anchor_visible, current_visible, matched),
            "by_anchor_surface": per_surface,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
