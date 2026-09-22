"""Topology-preserving Batch-2 non-rigid deformation operators.

The functions are renderer-agnostic except for Blender mesh/object handles.  The
caller provides canonical world positions and a writer so canonical attributes
remain fixed while all current geometry quantities are rerendered by RNA.
"""

from __future__ import annotations

import numpy as np
import bpy
from mathutils import Matrix


def _canonical_world(obj) -> np.ndarray:
    count = len(obj.data.vertices)
    local = np.empty((count, 3), dtype=np.float32)
    obj.data.vertices.foreach_get("co", local.reshape(-1))
    matrix = np.asarray(obj.matrix_world, dtype=np.float64)
    homogeneous = np.concatenate((local, np.ones((count, 1))), axis=1)
    return (homogeneous @ matrix.T)[:, :3]


def _set_world(obj, world: np.ndarray) -> None:
    inverse = np.asarray(obj.matrix_world.inverted(), dtype=np.float64)
    homogeneous = np.concatenate((world, np.ones((len(world), 1))), axis=1)
    local = (homogeneous @ inverse.T)[:, :3].astype(np.float32)
    obj.data.vertices.foreach_set("co", local.reshape(-1))
    obj.data.update(calc_edges=True)


def hinge_fold(world: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Smoothly rotate the positive-x side of a panel around its central hinge."""
    if angle_degrees == 0.0:
        return world.copy()
    result = world.copy()
    half_transition = 0.085
    t = np.clip((world[:, 0] + half_transition) / (2.0 * half_transition), 0.0, 1.0)
    weight = t * t * (3.0 - 2.0 * t)
    angle = np.deg2rad(angle_degrees) * weight
    cosine, sine = np.cos(angle), np.sin(angle)
    x, z = world[:, 0], world[:, 2]
    result[:, 0] = cosine * x - sine * z
    result[:, 2] = sine * x + cosine * z
    return result


def _object_summary(obj, before: np.ndarray, after: np.ndarray) -> dict:
    displacement = np.linalg.norm(after - before, axis=1)
    return {
        "object": obj.name,
        "vertices_before": len(before),
        "vertices_after": len(obj.data.vertices),
        "polygons_before": len(obj.data.polygons),
        "polygons_after": len(obj.data.polygons),
        "changed_vertex_fraction": float(np.mean(displacement > 1e-7)),
        "max_world_displacement": float(np.max(displacement)),
        "mean_world_displacement": float(np.mean(displacement)),
    }


def apply_a(state) -> dict:
    obj = bpy.data.objects.get("Batch2_FoldPanel")
    if obj is None:
        raise RuntimeError("Batch-2 A asset is missing Batch2_FoldPanel")
    before = _canonical_world(obj)
    after = hinge_fold(before, state.angle_degrees)
    _set_world(obj, after)
    return {
        "family": "A_fold_creation_disappearance",
        "parameter_name": state.parameter_name,
        "parameter_value": state.angle_degrees,
        "objects": [_object_summary(obj, before, after)],
    }


def apply_b(state) -> dict:
    approach = state.angle_degrees
    objects = []
    for name, direction in (("Batch2_ApproachLeft", 1.0), ("Batch2_ApproachRight", -1.0)):
        obj = bpy.data.objects.get(name)
        if obj is None:
            raise RuntimeError(f"Batch-2 B asset is missing {name}")
        before = _canonical_world(obj)
        obj.matrix_world = Matrix.Translation((direction * approach, 0.0, 0.0)) @ obj.matrix_world
        after = _canonical_world(obj)
        objects.append(_object_summary(obj, before, after))
    return {
        "family": "B_cross_part_approach_self_contact",
        "parameter_name": state.parameter_name,
        "parameter_value": approach,
        "nominal_midline_gap": max(0.0, 0.52 - 2.0 * approach),
        "objects": objects,
    }


def _compound_wing(world: np.ndarray, side: float, state_id: str) -> np.ndarray:
    translation, twist = {
        "C0": (0.0, 0.0), "C1": (0.0, 0.0),
        "C2": (0.16, 22.0), "C3": (0.28, 36.0),
    }[state_id]
    if translation == 0.0 and twist == 0.0:
        return world.copy()
    result = world.copy()
    center_x = float(np.mean(world[:, 0]))
    y_scale = max(float(np.max(np.abs(world[:, 1]))), 1e-6)
    phase = np.deg2rad(twist) * (world[:, 1] / y_scale)
    dx = world[:, 0] - center_x
    dz = world[:, 2] - 0.13
    result[:, 0] = center_x + np.cos(phase) * dx + np.sin(phase) * dz - side * translation
    result[:, 2] = 0.13 - np.sin(phase) * dx + np.cos(phase) * dz
    return result


def apply_c(state) -> dict:
    center = bpy.data.objects.get("Batch2_CompoundCenter")
    if center is None:
        raise RuntimeError("Batch-2 C asset is missing Batch2_CompoundCenter")
    objects = []
    before = _canonical_world(center)
    after = hinge_fold(before, state.angle_degrees)
    _set_world(center, after)
    objects.append(_object_summary(center, before, after))
    for name, side in (("Batch2_CompoundLeftWing", -1.0), ("Batch2_CompoundRightWing", 1.0)):
        wing = bpy.data.objects.get(name)
        if wing is None:
            raise RuntimeError(f"Batch-2 C asset is missing {name}")
        before = _canonical_world(wing)
        after = _compound_wing(before, side, state.state_id)
        _set_world(wing, after)
        objects.append(_object_summary(wing, before, after))
    translation, twist = {
        "C0": (0.0, 0.0), "C1": (0.0, 0.0),
        "C2": (0.16, 22.0), "C3": (0.28, 36.0),
    }[state.state_id]
    return {
        "family": "C_unseen_compound_deformation",
        "parameter_name": state.parameter_name,
        "parameter_value": state.angle_degrees,
        "components": {
            "center_fold_degrees": state.angle_degrees,
            "wing_inward_translation": translation,
            "wing_twist_degrees": twist,
        },
        "objects": objects,
    }


def apply(asset_kind: str, state) -> dict:
    return {"batch2a": apply_a, "batch2b": apply_b, "batch2c": apply_c}[asset_kind](state)
