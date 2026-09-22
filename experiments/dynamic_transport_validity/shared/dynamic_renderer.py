"""Correspondence-preserving frozen-RNA rendering on deformed geometry.

This module subclasses the published renderer only to replace the triplane lookup
coordinate with a rasterized canonical-position attribute.  The checkpoint, model,
network inputs other than position, visibility branch, and lighting implementation
remain the official RNA implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import shutil
import sys
from dataclasses import dataclass

import numpy as np


WORKSPACE = pathlib.Path(__file__).resolve().parents[3]
RNA_ROOT = WORKSPACE / "external" / "relightable-neural-assets"
EXPERIMENT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RNA_ROOT))
sys.path.insert(0, str(EXPERIMENT_DIR))

from shared.rna_compat import install_historical_module_aliases  # noqa: E402


install_historical_module_aliases()

import bpy  # noqa: E402
import torch as torch  # noqa: E402
from mathutils import Matrix  # noqa: E402
from rna import config, renderers  # noqa: E402
from utils import blender_utils as blender_utils  # noqa: E402
from utils import exr  # noqa: E402
from batch2.deformations import apply as apply_batch2_deformation  # noqa: E402


CANONICAL_ATTRIBUTE = "rna_canonical_position"
CANONICAL_AOV = "canonical_position_aov"
CANONICAL_FILE_STEM = "canonical_position"


@dataclass(frozen=True)
class DeformationState:
    state_id: str
    angle_degrees: float
    label: str
    parameter_name: str = "angle_degrees"


STATES = {
    "G0": DeformationState("G0", 0.0, "canonical"),
    "G1": DeformationState("G1", 15.0, "moderate"),
    "G2": DeformationState("G2", 35.0, "strong"),
    "RIGID": DeformationState("RIGID", 0.0, "rigid-transform-control"),
    "TRANSLATE": DeformationState("TRANSLATE", 0.0, "translation-control"),
}

SYNTHETIC_STATES = {
    "G0": DeformationState("G0", 0.0, "flat"),
    "G1": DeformationState("G1", 20.0, "mild-fold"),
    "G2": DeformationState("G2", 45.0, "moderate-fold"),
    "G3": DeformationState("G3", 75.0, "strong-fold"),
    "G4": DeformationState("G4", 120.0, "near-cavity"),
    "RIGID": DeformationState("RIGID", 0.0, "rigid-transform-control"),
}


# Batch 2 uses three authored, deformation-capable assets.  ``parameter_name``
# makes B-family gap closure distances explicit in run metadata.
BATCH2_A_STATES = {
    "A0": DeformationState("A0", 0.0, "open-near-flat"),
    "A1": DeformationState("A1", 55.0, "moderate-fold"),
    "A2": DeformationState("A2", 125.0, "strong-fold-cavity"),
    "A3": DeformationState("A3", 0.0, "fold-released"),
}
BATCH2_B_STATES = {
    "B0": DeformationState("B0", 0.00, "separated", "inward_translation"),
    "B1": DeformationState("B1", 0.20, "close-approach", "inward_translation"),
    "B2": DeformationState("B2", 0.31, "near-contact", "inward_translation"),
    "B3": DeformationState("B3", 0.35, "stable-contact", "inward_translation"),
    "B4": DeformationState("B4", 0.00, "contact-released", "inward_translation"),
}
BATCH2_C_STATES = {
    "C0": DeformationState("C0", 0.0, "open-canonical"),
    "C1": DeformationState("C1", 50.0, "single-center-fold"),
    "C2": DeformationState("C2", 85.0, "fold-twist-approach"),
    "C3": DeformationState("C3", 115.0, "strong-compound-near-contact"),
}
BATCH2_STATE_TABLES = {
    "batch2a": BATCH2_A_STATES,
    "batch2b": BATCH2_B_STATES,
    "batch2c": BATCH2_C_STATES,
}

def _scene_meshes():
    return [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and not obj.hide_render
    ]


def _normalized_world_vertices(obj, aabb_min: np.ndarray, aabb_max: np.ndarray):
    count = len(obj.data.vertices)
    local = np.empty((count, 3), dtype=np.float32)
    obj.data.vertices.foreach_get("co", local.reshape(-1))
    matrix = np.asarray(obj.matrix_world, dtype=np.float64)
    homogeneous = np.concatenate(
        (local.astype(np.float64), np.ones((count, 1), dtype=np.float64)), axis=1
    )
    world = (homogeneous @ matrix.T)[:, :3]
    extent = np.asarray(aabb_max, dtype=np.float64) - np.asarray(
        aabb_min, dtype=np.float64
    )
    if np.any(extent <= 0.0):
        raise ValueError(f"degenerate canonical scene AABB: {aabb_min}, {aabb_max}")
    # Match utils.ops.normalize_positions exactly: RNA trains in [0, 1], not [-1, 1].
    normalized = (world - aabb_min) / extent
    return normalized.astype(np.float32), world


def _install_canonical_position_aov(aabb_min: np.ndarray, aabb_max: np.ndarray):
    """Attach canonical positions to vertices and route them to a float EXR AOV."""

    meshes = _scene_meshes()
    if not meshes:
        raise RuntimeError("scene has no visible mesh")

    for obj in meshes:
        mesh = obj.data
        old = mesh.attributes.get(CANONICAL_ATTRIBUTE)
        if old is not None:
            mesh.attributes.remove(old)
        attribute = mesh.attributes.new(
            name=CANONICAL_ATTRIBUTE, type="FLOAT_VECTOR", domain="POINT"
        )
        normalized, _ = _normalized_world_vertices(obj, aabb_min, aabb_max)
        attribute.data.foreach_set("vector", normalized.reshape(-1))

    view_layer = bpy.context.scene.view_layers[0]
    if CANONICAL_AOV not in {item.name for item in view_layer.aovs}:
        view_layer.aovs.add()
        view_layer.aovs[-1].name = CANONICAL_AOV

    materials = {
        slot.material
        for obj in meshes
        for slot in obj.material_slots
        if slot.material is not None
    }
    for material in materials:
        material.use_nodes = True
        tree = material.node_tree
        for node in list(tree.nodes):
            if node.type == "OUTPUT_AOV" and node.name == CANONICAL_AOV:
                tree.nodes.remove(node)
            elif node.type == "ATTRIBUTE" and node.name == CANONICAL_ATTRIBUTE:
                tree.nodes.remove(node)
        attribute_node = tree.nodes.new("ShaderNodeAttribute")
        attribute_node.name = CANONICAL_ATTRIBUTE
        attribute_node.attribute_name = CANONICAL_ATTRIBUTE
        output_node = tree.nodes.new("ShaderNodeOutputAOV")
        output_node.name = CANONICAL_AOV
        output_node.label = CANONICAL_AOV
        tree.links.new(attribute_node.outputs["Vector"], output_node.inputs["Color"])

    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    render_layers = next(
        (node for node in tree.nodes if node.type == "R_LAYERS"), None
    )
    if render_layers is None:
        render_layers = tree.nodes.new("CompositorNodeRLayers")
    old_output = tree.nodes.get(CANONICAL_FILE_STEM + "_out")
    if old_output is not None:
        tree.nodes.remove(old_output)
    output = tree.nodes.new("CompositorNodeOutputFile")
    output.name = CANONICAL_FILE_STEM + "_out"
    output.label = CANONICAL_FILE_STEM
    output.format.file_format = "OPEN_EXR"
    output.format.color_mode = "RGB"
    output.format.color_depth = "32"
    output.base_path = ""
    output.file_slots[0].path = CANONICAL_FILE_STEM
    output.file_slots[0].use_node_format = True
    bpy.context.view_layer.update()
    socket = render_layers.outputs.get(CANONICAL_AOV)
    if socket is None:
        raise RuntimeError(f"Cycles did not expose registered AOV {CANONICAL_AOV}")
    tree.links.new(socket, output.inputs[0])


def _smooth_bend_world_x(
    world: np.ndarray,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    angle_degrees: float,
    hinge_fraction: float = 0.50,
    transition_fraction: float = 0.18,
) -> np.ndarray:
    """Bend the upper part of an asset while keeping a smooth hinge band."""

    if angle_degrees == 0.0:
        return world.copy()
    result = world.copy()
    y_min, y_max = float(aabb_min[1]), float(aabb_max[1])
    hinge = y_min + hinge_fraction * (y_max - y_min)
    transition = transition_fraction * (y_max - y_min)
    parameter = np.clip((world[:, 1] - hinge) / transition, 0.0, 1.0)
    weight = parameter * parameter * (3.0 - 2.0 * parameter)
    angles = np.deg2rad(angle_degrees) * weight
    cosine, sine = np.cos(angles), np.sin(angles)
    pivot_z = 0.5 * (float(aabb_min[2]) + float(aabb_max[2]))
    delta_y = world[:, 1] - hinge
    delta_z = world[:, 2] - pivot_z
    result[:, 1] = hinge + cosine * delta_y - sine * delta_z
    result[:, 2] = pivot_z + sine * delta_y + cosine * delta_z
    return result


def _apply_intrinsic_deformation(
    state: DeformationState, aabb_min: np.ndarray, aabb_max: np.ndarray
) -> dict:
    summaries = []
    for obj in _scene_meshes():
        mesh = obj.data
        vertex_count = len(mesh.vertices)
        polygon_count = len(mesh.polygons)
        normalized, canonical_world = _normalized_world_vertices(obj, aabb_min, aabb_max)
        del normalized
        deformed_world = _smooth_bend_world_x(
            canonical_world, aabb_min, aabb_max, state.angle_degrees
        )
        inverse = np.asarray(obj.matrix_world.inverted(), dtype=np.float64)
        homogeneous = np.concatenate(
            (deformed_world, np.ones((vertex_count, 1), dtype=np.float64)), axis=1
        )
        deformed_local = (homogeneous @ inverse.T)[:, :3].astype(np.float32)
        mesh.vertices.foreach_set("co", deformed_local.reshape(-1))
        mesh.update(calc_edges=True)
        displacement = np.linalg.norm(deformed_world - canonical_world, axis=1)
        summaries.append(
            {
                "object": obj.name,
                "vertices_before": vertex_count,
                "vertices_after": len(mesh.vertices),
                "polygons_before": polygon_count,
                "polygons_after": len(mesh.polygons),
                "changed_vertex_fraction": float(np.mean(displacement > 1e-7)),
                "max_world_displacement": float(np.max(displacement)),
                "mean_world_displacement": float(np.mean(displacement)),
            }
        )
    return {"objects": summaries}


def _hinge_fold_world_y(world: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Continuously fold a thin connected sheet about the x=0 hinge band."""

    result = world.copy()
    half_transition = 0.08
    t = np.clip(
        (world[:, 0] + half_transition) / (2.0 * half_transition), 0.0, 1.0
    )
    weight = t * t * (3.0 - 2.0 * t)
    angle = np.deg2rad(angle_degrees) * weight
    cosine, sine = np.cos(angle), np.sin(angle)
    x, z = world[:, 0], world[:, 2]
    result[:, 0] = cosine * x - sine * z
    result[:, 2] = sine * x + cosine * z
    return result


def _apply_synthetic_deformation(state: DeformationState) -> dict:
    summaries = []
    for obj in _scene_meshes():
        mesh = obj.data
        vertex_count, polygon_count = len(mesh.vertices), len(mesh.polygons)
        local = np.empty((vertex_count, 3), dtype=np.float32)
        mesh.vertices.foreach_get("co", local.reshape(-1))
        matrix = np.asarray(obj.matrix_world, dtype=np.float64)
        homogeneous = np.concatenate(
            (local.astype(np.float64), np.ones((vertex_count, 1))), axis=1
        )
        canonical_world = (homogeneous @ matrix.T)[:, :3]
        deformed_world = _hinge_fold_world_y(canonical_world, state.angle_degrees)
        inverse = np.asarray(obj.matrix_world.inverted(), dtype=np.float64)
        deformed_homogeneous = np.concatenate(
            (deformed_world, np.ones((vertex_count, 1))), axis=1
        )
        deformed_local = (deformed_homogeneous @ inverse.T)[:, :3].astype(np.float32)
        mesh.vertices.foreach_set("co", deformed_local.reshape(-1))
        mesh.update(calc_edges=True)
        displacement = np.linalg.norm(deformed_world - canonical_world, axis=1)
        summaries.append(
            {
                "object": obj.name,
                "vertices_before": vertex_count,
                "vertices_after": len(mesh.vertices),
                "polygons_before": polygon_count,
                "polygons_after": len(mesh.polygons),
                "changed_vertex_fraction": float(np.mean(displacement > 1e-7)),
                "max_world_displacement": float(np.max(displacement)),
                "mean_world_displacement": float(np.mean(displacement)),
            }
        )
    return {"objects": summaries}


def _apply_rigid_control() -> dict:
    angle = math.radians(17.0)
    rotation = Matrix.Rotation(angle, 4, "Z")
    translation = Matrix.Translation((0.12, -0.08, 0.06))
    objects = []
    for obj in _scene_meshes():
        vertex_count, polygon_count = len(obj.data.vertices), len(obj.data.polygons)
        obj.matrix_world = translation @ rotation @ obj.matrix_world
        objects.append(
            {
                "object": obj.name,
                "vertices_before": vertex_count,
                "vertices_after": len(obj.data.vertices),
                "polygons_before": polygon_count,
                "polygons_after": len(obj.data.polygons),
                "rotation_degrees_world_z": 17.0,
                "translation": [0.12, -0.08, 0.06],
            }
        )
    return {"objects": objects}


def _apply_translation_control() -> dict:
    translation_vector = (0.12, -0.08, 0.06)
    translation = Matrix.Translation(translation_vector)
    objects = []
    for obj in _scene_meshes():
        vertex_count, polygon_count = len(obj.data.vertices), len(obj.data.polygons)
        obj.matrix_world = translation @ obj.matrix_world
        objects.append(
            {
                "object": obj.name,
                "vertices_before": vertex_count,
                "vertices_after": len(obj.data.vertices),
                "polygons_before": polygon_count,
                "polygons_after": len(obj.data.polygons),
                "rotation_degrees_world_z": 0.0,
                "translation": list(translation_vector),
            }
        )
    return {"objects": objects}


def _rigid_control_matrix() -> np.ndarray:
    angle = np.deg2rad(17.0)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [cosine, -sine, 0.0, 0.12],
            [sine, cosine, 0.0, -0.08],
            [0.0, 0.0, 1.0, 0.06],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _translation_control_matrix() -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = (0.12, -0.08, 0.06)
    return matrix


def _transform_evaluation_frame(light_positions, camera_matrices, state_id: str):
    """Apply the object rigid transform to cameras/lights as a frame control."""

    matrix = (
        _translation_control_matrix()
        if state_id == "TRANSLATE"
        else _rigid_control_matrix()
    )
    transformed_lights = []
    for position in light_positions:
        homogeneous = np.append(np.asarray(position, dtype=np.float64), 1.0)
        transformed_lights.append((matrix @ homogeneous)[:3].tolist())
    transformed_cameras = [
        (matrix @ np.asarray(camera, dtype=np.float64)).tolist()
        for camera in camera_matrices
    ]
    return transformed_lights, transformed_cameras


class CorrespondenceSurfaceRenderer(renderers.NeuralSurfaceHairRenderer):
    def __init__(self, *args, state: DeformationState, lookup: str, asset_kind: str, **kwargs):
        self.state = state
        self.lookup = lookup
        self.asset_kind = asset_kind
        self.canonical_aabb_min = None
        self.canonical_aabb_max = None
        self.deformation_summary = None
        super().__init__(*args, **kwargs)
        self._restore_dynamic_scene()

    def _restore_dynamic_scene(self):
        aabb_min, aabb_max = blender_utils.get_scene_bounding_box()
        self.canonical_aabb_min = np.asarray(aabb_min, dtype=np.float64)
        self.canonical_aabb_max = np.asarray(aabb_max, dtype=np.float64)
        _install_canonical_position_aov(
            self.canonical_aabb_min, self.canonical_aabb_max
        )
        if self.state.state_id == "RIGID":
            self.deformation_summary = _apply_rigid_control()
        elif self.state.state_id == "TRANSLATE":
            self.deformation_summary = _apply_translation_control()
        elif self.asset_kind == "synthetic":
            self.deformation_summary = _apply_synthetic_deformation(self.state)
        elif self.asset_kind in BATCH2_STATE_TABLES:
            self.deformation_summary = apply_batch2_deformation(self.asset_kind, self.state)
        else:
            self.deformation_summary = _apply_intrinsic_deformation(
                self.state, self.canonical_aabb_min, self.canonical_aabb_max
            )

    def render_features(self, resolution, prepass=False, frame=None):
        super().render_features(resolution, prepass=prepass, frame=frame)
        if frame is not None:
            source = pathlib.Path(CANONICAL_FILE_STEM + "0001.exr")
            target = pathlib.Path(renderers.TMP_RENDER_DIR) / (
                CANONICAL_FILE_STEM + "0001_" + str(frame) + ".exr"
            )
            if not source.exists():
                raise RuntimeError(f"canonical correspondence AOV was not written: {source}")
            source.replace(target)

    def render_visibility(self, resolution, frame=None):
        super().render_visibility(resolution, frame=frame)
        # The official implementation reverts the .blend to restore materials.
        # Reinstall only the experiment AOV and deterministic deformation afterward.
        self._restore_dynamic_scene()

    def process_deep_buffers(self, mask, visibility=True, frame=999):
        official = super().process_deep_buffers(mask, visibility=visibility, frame=frame)
        if self.lookup == "naive":
            return official
        canonical_path = pathlib.Path(renderers.TMP_RENDER_DIR) / (
            CANONICAL_FILE_STEM + "0001_" + str(frame) + ".exr"
        )
        canonical = torch.from_numpy(exr.read(str(canonical_path))).reshape(-1, 3)
        canonical = canonical[mask]
        if not torch.isfinite(canonical).all():
            raise RuntimeError("canonical lookup AOV contains non-finite values")
        if self.state.state_id in {"G0", "A0", "A3", "B0", "B4", "C0"}:
            ordinary = official[0]
            error = torch.abs(canonical - ordinary)
            self.identity_correspondence = {
                "samples": int(canonical.shape[0]),
                "max_abs_error": float(error.max().item()),
                "mean_abs_error": float(error.mean().item()),
            }
        return (canonical, *official[1:])


def _write_run_metadata(
    renderer, output_dir: pathlib.Path, args, conf, evaluation_manifest: dict
) -> None:
    metadata = {
        "state": renderer.state.state_id,
        "asset_kind": renderer.asset_kind,
        "state_label": renderer.state.label,
        "angle_degrees": renderer.state.angle_degrees,
        "lookup": renderer.lookup,
        "deformation_parameter_name": renderer.state.parameter_name,
        "canonical_aabb_min": renderer.canonical_aabb_min.tolist(),
        "canonical_aabb_max": renderer.canonical_aabb_max.tolist(),
        "deformation": renderer.deformation_summary,
        "identity_correspondence": getattr(renderer, "identity_correspondence", None),
        "checkpoint": conf.renderer_params.checkpoint,
        "blend_file": conf.renderer_params.blend_file,
        "resolution": list(conf.renderer_params.resolution),
        "aa_samples": int(conf.renderer_params.aa_samples),
        "reference_samples": int(conf.renderer_params.reference_samples),
        "frames": list(conf.frames),
        "evaluation_camera_light_manifest": evaluation_manifest,
        "model_frozen": True,
        "cli": {
            key: str(value) if isinstance(value, pathlib.Path) else value
            for key, value in vars(args).items()
        },
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def _preserve_diagnostic_buffers(output_dir: pathlib.Path, frames) -> None:
    for frame in frames:
        sources = {
            "visibility.exr": pathlib.Path(renderers.TMP_RENDER_DIR)
            / f"diffuse_direct{frame}.exr",
            "diffuse_direct_feature.exr": pathlib.Path(renderers.TMP_RENDER_DIR)
            / f"diffuse_direct0001_{frame}.exr",
            "glossy_direct_feature.exr": pathlib.Path(renderers.TMP_RENDER_DIR)
            / f"glossy_direct0001_{frame}.exr",
            "current_normal.exr": pathlib.Path(renderers.TMP_RENDER_DIR)
            / f"normal0001_{frame}.exr",
            "current_tangent.exr": pathlib.Path(renderers.TMP_RENDER_DIR)
            / f"tangent0001_{frame}.exr",
            "current_camera_direction.exr": pathlib.Path(renderers.TMP_RENDER_DIR)
            / f"camera_dir0001_{frame}.exr",
            "current_position.exr": pathlib.Path(renderers.TMP_RENDER_DIR)
            / f"position0001_{frame}.exr",
            "canonical_position.exr": pathlib.Path(renderers.TMP_RENDER_DIR)
            / f"{CANONICAL_FILE_STEM}0001_{frame}.exr",
        }
        for target_name, source in sources.items():
            if source.exists():
                shutil.copy2(source, output_dir / target_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--asset-kind", choices=("lego", "synthetic", "batch2a", "batch2b", "batch2c"), default="lego")
    parser.add_argument("--state", choices=sorted(set(STATES) | set(SYNTHETIC_STATES) | set(BATCH2_A_STATES) | set(BATCH2_B_STATES) | set(BATCH2_C_STATES)), required=True)
    parser.add_argument("--lookup", choices=("canonical", "naive"), default="canonical")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", action="store_true")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument(
        "--coupled-rigid-frame",
        action="store_true",
        help="transform cameras and point lights with the RIGID object control",
    )
    args = parser.parse_args()
    if not (args.model or args.reference):
        raise ValueError("select --model and/or --reference")

    conf = config.get_render_config(args.config)
    if args.checkpoint:
        conf.renderer_params.checkpoint = args.checkpoint
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    conf.output_dir = str(output_dir)
    jsn = json.loads(pathlib.Path(conf.json).read_text(encoding="utf-8"))
    frames = jsn["frames"]
    conf.light_type = "PointLight"
    light_positions = [item["pl_pos"] for item in frames]
    camera_matrices = [item["transform_matrix"] for item in frames]
    if args.coupled_rigid_frame:
        if args.state not in ("RIGID", "TRANSLATE"):
            raise ValueError(
                "--coupled-rigid-frame is valid only for RIGID or TRANSLATE"
            )
        light_positions, camera_matrices = _transform_evaluation_frame(
            light_positions, camera_matrices, args.state
        )
    light_intensity = frames[0]["pl_intensity"][0]

    state_table = BATCH2_STATE_TABLES.get(args.asset_kind, SYNTHETIC_STATES if args.asset_kind == "synthetic" else STATES)
    if args.state not in state_table:
        raise ValueError(f"state {args.state} is not declared for {args.asset_kind}")
    renderer = CorrespondenceSurfaceRenderer(
        conf,
        **conf.renderer_params,
        device=args.device,
        state=state_table[args.state],
        lookup=args.lookup,
        asset_kind=args.asset_kind,
    )
    renderer.prepare(
        light_positions,
        camera_matrices,
        light_intensity,
        jsn["camera_angle_x"],
    )
    if args.model:
        renderer.render_animation_neural()
        _preserve_diagnostic_buffers(output_dir, conf.frames)
    if args.reference:
        renderer.blender_render(conf.frames, str(output_dir))
    _write_run_metadata(renderer, output_dir, args, conf, jsn)


if __name__ == "__main__":
    main()
