"""Create the deterministic connected folding-sheet control scene."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

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
    for datablocks in (bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for block in list(datablocks):
            datablocks.remove(block)


def create_sheet(
    nx: int = 65,
    ny: int = 33,
    *,
    thickness: float = 0.03,
    roughness: float = 0.72,
    fold_angle_degrees: float = 0.0,
):
    def fold_point(x: float, y: float, z: float) -> tuple[float, float, float]:
        if fold_angle_degrees == 0.0:
            return x, y, z
        half_transition = 0.08
        t = np.clip((x + half_transition) / (2.0 * half_transition), 0.0, 1.0)
        weight = t * t * (3.0 - 2.0 * t)
        angle = np.deg2rad(fold_angle_degrees) * weight
        return (
            np.cos(angle) * x - np.sin(angle) * z,
            y,
            np.sin(angle) * x + np.cos(angle) * z,
        )

    vertices = []
    for z in (-0.5 * thickness, 0.5 * thickness):
        for iy in range(ny):
            y = -0.7 + 1.4 * iy / (ny - 1)
            for ix in range(nx):
                x = -1.0 + 2.0 * ix / (nx - 1)
                vertices.append(fold_point(x, y, z))

    layer_size = nx * ny
    faces = []
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            a = iy * nx + ix
            faces.append((a, a + nx, a + 1 + nx, a + 1))
            top = layer_size + a
            faces.append((top, top + 1, top + 1 + nx, top + nx))

    # Close the four boundary strips so RNA always sees a finite 3D AABB and
    # the folding control is a connected, thin solid rather than a degenerate
    # zero-thickness plane.
    for ix in range(nx - 1):
        a, b = ix, ix + 1
        faces.append((a, b, layer_size + b, layer_size + a))
        a = (ny - 1) * nx + ix
        b = a + 1
        faces.append((a, layer_size + a, layer_size + b, b))
    for iy in range(ny - 1):
        a, b = iy * nx, (iy + 1) * nx
        faces.append((a, layer_size + a, layer_size + b, b))
        a, b = iy * nx + nx - 1, (iy + 1) * nx + nx - 1
        faces.append((a, b, layer_size + b, layer_size + a))

    mesh = bpy.data.meshes.new("ConnectedFoldingSheetMesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update(calc_edges=True)
    sheet = bpy.data.objects.new("ConnectedFoldingSheet", mesh)
    bpy.context.scene.collection.objects.link(sheet)

    uv_layer = mesh.uv_layers.new(name="UVMap")
    for polygon in mesh.polygons:
        for loop_index in polygon.loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            x, y, _ = vertices[vertex_index]
            uv_layer.data[loop_index].uv = ((x + 1.0) * 0.5, (y + 0.7) / 1.4)

    material = bpy.data.materials.new(
        "SheetDiffuseRough" if roughness >= 0.5 else "SheetModeratelyGlossy"
    )
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    principled = tree.nodes.new("ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = (0.62, 0.22, 0.08, 1.0)
    principled.inputs["Roughness"].default_value = roughness
    tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    sheet.data.materials.append(material)
    return sheet


def add_preview_camera_and_light() -> tuple[np.ndarray, list[float]]:
    camera_data = bpy.data.cameras.new("EvaluationCamera")
    camera = bpy.data.objects.new("EvaluationCamera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    matrix = ops.get_look_at(
        np.array([2.6, -3.0, 2.4]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    camera.matrix_world = matrix
    camera_data.lens_unit = "FOV"
    camera_data.angle = np.deg2rad(48.0)
    bpy.context.scene.camera = camera

    light_position = [-2.0, -1.5, 3.0]
    light_data = bpy.data.lights.new("EvaluationKey", type="AREA")
    light_data.energy = 800.0
    light_data.shape = "DISK"
    light_data.size = 1.0
    light = bpy.data.objects.new("EvaluationKey", light_data)
    bpy.context.scene.collection.objects.link(light)
    light.location = light_position
    return np.asarray(matrix), light_position


def configure_scene() -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.film_transparent = True
    scene.render.resolution_percentage = 100
    scene.cycles.max_bounces = 32
    scene.cycles.use_light_tree = False
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = (0.01, 0.01, 0.01, 1.0)
        background.inputs["Strength"].default_value = 0.05
    setup_blender_scene.add_aovs(hair=False)
    setup_blender_scene.add_shading_aov_nodes(hair=False)
    setup_blender_scene.add_compositor_nodes(hair=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend-output", required=True, type=pathlib.Path)
    parser.add_argument("--json-output", required=True, type=pathlib.Path)
    parser.add_argument("--roughness", type=float, default=0.72)
    parser.add_argument("--fold-angle", type=float, default=0.0)
    args = parser.parse_args()
    clear_scene()
    create_sheet(roughness=args.roughness, fold_angle_degrees=args.fold_angle)
    camera_matrix, light_position = add_preview_camera_and_light()
    configure_scene()
    args.blend_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.blend_output.resolve()))
    evaluation = {
        "camera_angle_x": float(np.deg2rad(48.0)),
        "frames": [
            {
                "file_ext": ".png",
                "file_path": "synthetic_G0",
                "pl_intensity": [800.0, 800.0, 800.0],
                "pl_pos": light_position,
                "transform_matrix": camera_matrix.tolist(),
            }
        ],
    }
    args.json_output.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
