"""Build visually legible Batch-2 RNA assets in Blender.

These are intentionally authored as small sculptural cloth/flap assets rather
than a diagnostic plane.  Each mesh is a closed, finite-thickness grid, so the
published RNA data pipeline sees an ordinary surface asset with a nonzero AABB.
The corresponding non-rigid motions live in ``dynamic_renderer.py`` and preserve
all vertex and polygon identities.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections.abc import Callable

import numpy as np


WORKSPACE = pathlib.Path(__file__).resolve().parents[2]
RNA_ROOT = WORKSPACE / "external" / "relightable-neural-assets"
sys.path.insert(0, str(RNA_ROOT))

import bpy  # noqa: E402
from scripts import setup_blender_scene  # noqa: E402
from utils import ops  # noqa: E402


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for blocks in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for block in list(blocks):
            blocks.remove(block)


def make_solid_grid(
    name: str,
    surface: Callable[[float, float], tuple[float, float, float]],
    *,
    nu: int,
    nv: int,
    thickness: float,
    thickness_axis: tuple[float, float, float],
    material,
):
    """Create a closed quad-grid shell with persistent vertex/face order."""

    offset = 0.5 * thickness * np.asarray(thickness_axis, dtype=np.float64)
    vertices: list[tuple[float, float, float]] = []
    points: list[np.ndarray] = []
    for v_index in range(nv):
        v = v_index / (nv - 1)
        for u_index in range(nu):
            u = u_index / (nu - 1)
            points.append(np.asarray(surface(u, v), dtype=np.float64))
    for sign in (-1.0, 1.0):
        vertices.extend(tuple(point + sign * offset) for point in points)

    layer = nu * nv
    faces: list[tuple[int, int, int, int]] = []
    for v_index in range(nv - 1):
        for u_index in range(nu - 1):
            index = v_index * nu + u_index
            faces.append((index, index + nu, index + nu + 1, index + 1))
            top = layer + index
            faces.append((top, top + 1, top + nu + 1, top + nu))
    for u_index in range(nu - 1):
        low_a, low_b = u_index, u_index + 1
        high_a, high_b = (nv - 1) * nu + u_index, (nv - 1) * nu + u_index + 1
        faces.append((low_a, low_b, layer + low_b, layer + low_a))
        faces.append((high_a, layer + high_a, layer + high_b, high_b))
    for v_index in range(nv - 1):
        left_a, left_b = v_index * nu, (v_index + 1) * nu
        right_a, right_b = v_index * nu + nu - 1, (v_index + 1) * nu + nu - 1
        faces.append((left_a, layer + left_a, layer + left_b, left_b))
        faces.append((right_a, right_b, layer + right_b, layer + right_a))

    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    mesh.materials.append(material)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    return obj


def make_material(name: str, base: tuple[float, float, float], roughness: float, metallic: float = 0.0):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    shader = tree.nodes.new("ShaderNodeBsdfPrincipled")
    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 5.5
    noise.inputs["Detail"].default_value = 4.0
    ramp = tree.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.28
    ramp.color_ramp.elements[0].color = (*np.asarray(base) * 0.36, 1.0)
    ramp.color_ramp.elements[1].position = 0.72
    ramp.color_ramp.elements[1].color = (*np.minimum(np.asarray(base) * 1.25, 1.0), 1.0)
    tree.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    tree.links.new(ramp.outputs["Color"], shader.inputs["Base Color"])
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Metallic"].default_value = metallic
    if "Sheen Weight" in shader.inputs:
        shader.inputs["Sheen Weight"].default_value = 0.18
    tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def add_fold_asset() -> None:
    velvet = make_material("Batch2VelvetAmber", (0.58, 0.085, 0.025), 0.34, 0.0)

    def surface(u: float, v: float):
        x = -1.12 + 2.24 * u
        y = -0.88 + 1.76 * v
        z = 0.045 * np.sin(2.0 * np.pi * y / 1.76) * (1.0 - 0.25 * (x / 1.12) ** 2)
        return x, y, z

    make_solid_grid(
        "Batch2_FoldPanel", surface, nu=89, nv=65, thickness=0.055,
        thickness_axis=(0.0, 0.0, 1.0), material=velvet,
    )


def _leaf_surface(center_x: float, phase: float = 0.0):
    def surface(u: float, v: float):
        signed_u = 2.0 * u - 1.0
        signed_v = 2.0 * v - 1.0
        width = 0.10 + 0.34 * np.power(max(0.0, 1.0 - signed_v * signed_v), 0.52)
        x = center_x + signed_u * width
        z = 0.03 + 0.92 * signed_v
        y = 0.095 * (1.0 - signed_u * signed_u) * (1.0 - signed_v * signed_v)
        y += 0.018 * np.sin(4.0 * np.pi * v + phase) * (1.0 - signed_u * signed_u)
        return x, y, z

    return surface


def add_approach_asset() -> None:
    jade = make_material("Batch2ApproachJade", (0.055, 0.33, 0.19), 0.23, 0.05)
    gold = make_material("Batch2ApproachCopper", (0.48, 0.12, 0.025), 0.29, 0.15)
    make_solid_grid(
        "Batch2_ApproachLeft", _leaf_surface(-0.70, 0.0), nu=61, nv=89,
        thickness=0.052, thickness_axis=(0.0, 1.0, 0.0), material=jade,
    )
    make_solid_grid(
        "Batch2_ApproachRight", _leaf_surface(0.70, np.pi), nu=61, nv=89,
        thickness=0.052, thickness_axis=(0.0, 1.0, 0.0), material=gold,
    )


def _wing_surface(center_x: float, side: float):
    def surface(u: float, v: float):
        signed_u = 2.0 * u - 1.0
        signed_v = 2.0 * v - 1.0
        x = center_x + 0.43 * signed_u
        y = 0.78 * signed_v
        z = 0.13 + 0.12 * (1.0 - signed_u * signed_u) * (1.0 - signed_v * signed_v)
        z += side * 0.035 * signed_u * signed_v
        return x, y, z

    return surface


def add_compound_asset() -> None:
    center_material = make_material("Batch2CompoundPlum", (0.31, 0.025, 0.17), 0.31, 0.0)
    left_material = make_material("Batch2CompoundTeal", (0.02, 0.28, 0.31), 0.27, 0.08)
    right_material = make_material("Batch2CompoundOchre", (0.54, 0.21, 0.025), 0.30, 0.05)

    def center_surface(u: float, v: float):
        x = -0.58 + 1.16 * u
        y = -0.82 + 1.64 * v
        z = 0.045 * np.sin(np.pi * x / 1.16) * np.sin(2.0 * np.pi * y / 1.64)
        return x, y, z

    make_solid_grid(
        "Batch2_CompoundCenter", center_surface, nu=73, nv=65, thickness=0.055,
        thickness_axis=(0.0, 0.0, 1.0), material=center_material,
    )
    make_solid_grid(
        "Batch2_CompoundLeftWing", _wing_surface(-0.96, -1.0), nu=53, nv=57,
        thickness=0.048, thickness_axis=(0.0, 0.0, 1.0), material=left_material,
    )
    make_solid_grid(
        "Batch2_CompoundRightWing", _wing_surface(0.96, 1.0), nu=53, nv=57,
        thickness=0.048, thickness_axis=(0.0, 0.0, 1.0), material=right_material,
    )


def configure_scene() -> tuple[np.ndarray, list[float]]:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.film_transparent = True
    scene.render.resolution_percentage = 100
    scene.cycles.max_bounces = 32
    scene.cycles.use_light_tree = False
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.004, 0.004, 0.006, 1.0)
    background.inputs["Strength"].default_value = 0.035

    camera_data = bpy.data.cameras.new("EvaluationCamera")
    camera = bpy.data.objects.new("EvaluationCamera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    matrix = ops.get_look_at(
        np.array([2.55, -3.45, 2.25]), np.array([0.0, 0.0, 0.05]), np.array([0.0, 0.0, 1.0])
    )
    camera.matrix_world = matrix
    camera_data.lens_unit = "FOV"
    camera_data.angle = np.deg2rad(46.0)
    scene.camera = camera
    light_position = [-1.85, -2.55, 3.15]
    setup_blender_scene.add_aovs(hair=False)
    setup_blender_scene.add_shading_aov_nodes(hair=False)
    setup_blender_scene.add_compositor_nodes(hair=False)
    return np.asarray(matrix), light_position


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("A", "B", "C"), required=True)
    parser.add_argument("--blend-output", required=True, type=pathlib.Path)
    parser.add_argument("--json-output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    clear_scene()
    {"A": add_fold_asset, "B": add_approach_asset, "C": add_compound_asset}[args.family]()
    camera_matrix, light_position = configure_scene()
    args.blend_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.blend_output.resolve()))
    evaluation = {
        "camera_angle_x": float(np.deg2rad(46.0)),
        "frames": [{
            "file_ext": ".png",
            "file_path": "batch2_" + args.family + "_canonical",
            "pl_intensity": [900.0, 900.0, 900.0],
            "pl_pos": light_position,
            "transform_matrix": camera_matrix.tolist(),
        }],
    }
    args.json_output.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
