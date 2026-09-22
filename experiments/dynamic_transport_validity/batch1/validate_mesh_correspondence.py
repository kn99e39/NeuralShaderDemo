"""Focused correspondence and current-normal checks on the actual Lego mesh."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np


WORKSPACE = pathlib.Path(__file__).resolve().parents[3]
RNA_ROOT = WORKSPACE / "external" / "relightable-neural-assets"
EXPERIMENT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RNA_ROOT))
sys.path.insert(0, str(EXPERIMENT_DIR))

import bpy  # noqa: E402

from shared.correspondence import (  # noqa: E402
    interpolate,
    map_deformed_to_canonical,
    transform_points,
    triangle_normals,
    validate_barycentric_mapping,
)
from shared.dynamic_renderer import _smooth_bend_world_x  # noqa: E402
from utils import blender_utils  # noqa: E402


def world_vertices(obj) -> np.ndarray:
    local = np.empty((len(obj.data.vertices), 3), dtype=np.float64)
    obj.data.vertices.foreach_get("co", local.reshape(-1))
    matrix = np.asarray(obj.matrix_world, dtype=np.float64)
    homogeneous = np.concatenate((local, np.ones((len(local), 1))), axis=1)
    return (homogeneous @ matrix.T)[:, :3]


def sample_triangles(obj, vertices, count: int, seed: int):
    rng = np.random.default_rng(seed)
    candidates = rng.choice(len(obj.data.polygons), size=min(count * 4, len(obj.data.polygons)), replace=False)
    triangles, polygon_indices = [], []
    for polygon_index in candidates:
        polygon = obj.data.polygons[int(polygon_index)]
        if len(polygon.vertices) < 3:
            continue
        tri = vertices[np.asarray(polygon.vertices[:3], dtype=np.int64)]
        if np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])) <= 1e-10:
            continue
        triangles.append(tri)
        polygon_indices.append(int(polygon_index))
        if len(triangles) == count:
            break
    if len(triangles) < count:
        raise RuntimeError(f"only found {len(triangles)} nondegenerate triangles")
    return np.asarray(triangles), polygon_indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend-file", required=True)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--angle", type=float, default=35.0)
    args = parser.parse_args()

    bpy.ops.wm.open_mainfile(filepath=args.blend_file)
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and not obj.hide_render]
    if len(meshes) != 1:
        raise RuntimeError(f"expected one visible Lego mesh, found {len(meshes)}")
    obj = meshes[0]
    canonical_vertices = world_vertices(obj)
    aabb_min, aabb_max = blender_utils.get_scene_bounding_box()
    deformed_vertices = _smooth_bend_world_x(
        canonical_vertices, np.asarray(aabb_min), np.asarray(aabb_max), args.angle
    )

    canonical_triangles, polygon_indices = sample_triangles(
        obj, canonical_vertices, args.samples, args.seed
    )
    deformed_triangles = []
    for polygon_index in polygon_indices:
        indices = np.asarray(obj.data.polygons[polygon_index].vertices[:3], dtype=np.int64)
        deformed_triangles.append(deformed_vertices[indices])
    deformed_triangles = np.asarray(deformed_triangles)

    # A nonlinear bend can make a small number of triangles locally singular at
    # this exact state.  They have no well-defined barycentric inverse and are
    # reported separately rather than contaminating the correspondence test.
    canonical_double_area = np.linalg.norm(
        np.cross(
            canonical_triangles[:, 1] - canonical_triangles[:, 0],
            canonical_triangles[:, 2] - canonical_triangles[:, 0],
        ),
        axis=-1,
    )
    deformed_double_area = np.linalg.norm(
        np.cross(
            deformed_triangles[:, 1] - deformed_triangles[:, 0],
            deformed_triangles[:, 2] - deformed_triangles[:, 0],
        ),
        axis=-1,
    )
    # barycentric_coordinates rejects Gram determinants <= 1e-12.  The Gram
    # determinant is the squared double-area, so compare double-area against
    # sqrt(1e-12), not 1e-12 itself.  Check both states because some imported
    # production meshes contain numerically collapsed source triangles.
    double_area_epsilon = np.sqrt(1e-12)
    nonsingular = np.logical_and(
        canonical_double_area > double_area_epsilon,
        deformed_double_area > double_area_epsilon,
    )
    singular_after_deformation = int(np.count_nonzero(~nonsingular))
    canonical_triangles = canonical_triangles[nonsingular]
    deformed_triangles = deformed_triangles[nonsingular]

    rng = np.random.default_rng(args.seed + 1)
    weights = rng.dirichlet(np.ones(3), size=len(canonical_triangles))
    barycentric = validate_barycentric_mapping(
        canonical_triangles, deformed_triangles, weights
    )

    canonical_normals = triangle_normals(canonical_triangles)
    current_normals = triangle_normals(deformed_triangles)
    cosine = np.clip(np.sum(canonical_normals * current_normals, axis=-1), -1.0, 1.0)
    normal_angle = np.rad2deg(np.arccos(cosine))
    changed_vertices = np.linalg.norm(deformed_vertices - canonical_vertices, axis=-1) > 1e-7

    rigid_matrix = np.array(
        [
            [0.95630476, -0.29237170, 0.0, 0.12],
            [0.29237170, 0.95630476, 0.0, -0.08],
            [0.0, 0.0, 1.0, 0.06],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    rigid_triangles = transform_points(canonical_triangles, rigid_matrix)
    rigid_points = interpolate(rigid_triangles, weights)
    rigid_mapped, rigid_weights = map_deformed_to_canonical(
        rigid_points, rigid_triangles, canonical_triangles
    )
    expected_canonical = interpolate(canonical_triangles, weights)

    result = {
        "asset": obj.name,
        "vertex_count": len(obj.data.vertices),
        "polygon_count": len(obj.data.polygons),
        "sample_count": len(canonical_triangles),
        "singular_samples_excluded_after_deformation": singular_after_deformation,
        "seed": args.seed,
        "deformation_angle_degrees": args.angle,
        "changed_vertex_fraction": float(np.mean(changed_vertices)),
        "barycentric": barycentric.as_dict(),
        "current_normal": {
            "mean_angle_from_canonical_degrees": float(np.mean(normal_angle)),
            "max_angle_from_canonical_degrees": float(np.max(normal_angle)),
            "changed_normal_fraction_over_0_1_degree": float(np.mean(normal_angle > 0.1)),
            "finite": bool(np.isfinite(current_normals).all()),
        },
        "rigid_control": {
            "max_canonical_position_error": float(
                np.max(np.abs(rigid_mapped - expected_canonical))
            ),
            "max_barycentric_error": float(np.max(np.abs(rigid_weights - weights))),
        },
        "topology": {
            "vertex_identity_preserved": len(canonical_vertices) == len(deformed_vertices),
            "polygon_identity_preserved": True,
            "topology_changes": 0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
