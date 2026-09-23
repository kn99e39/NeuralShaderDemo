"""Add a transported canonical-world-position AOV to an RNA scene copy.

The point-domain attribute is populated once in the canonical pose.  Blender's
armature evaluation carries it with the mesh, while its value remains the
canonical world position.  Rasterization then gives barycentric interpolation
at a deformed pixel without querying the frozen TriPlane at a deformed point.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import bpy
import numpy as np
from mathutils import Vector


ATTRIBUTE_NAME = "rna_canonical_world_position"
AOV_NAME = "canonical_position_aov"
COMPOSITOR_NODE_NAME = "canonical_position_out"


def ensure_aov(view_layer: bpy.types.ViewLayer) -> None:
    for aov in list(view_layer.aovs):
        if aov.name == AOV_NAME:
            view_layer.aovs.remove(aov)
    aov = view_layer.aovs.add()
    aov.name = AOV_NAME


def set_canonical_attribute(obj: bpy.types.Object) -> tuple[int, list[list[float]]]:
    mesh = obj.data
    existing = mesh.attributes.get(ATTRIBUTE_NAME)
    if existing is not None:
        mesh.attributes.remove(existing)
    # A FLOAT_VECTOR is a spatial attribute and Blender converts it through
    # the object's current transform in shader evaluation.  A FLOAT_COLOR is
    # a non-spatial four-float payload, so RGB can carry canonical XYZ without
    # acquiring a whole-object rigid motion.
    attribute = mesh.color_attributes.new(ATTRIBUTE_NAME, "FLOAT_COLOR", "POINT")
    values = np.empty((len(mesh.vertices), 3), dtype=np.float32)
    for index, vertex in enumerate(mesh.vertices):
        values[index] = obj.matrix_world @ vertex.co
    colors = np.ones((len(mesh.vertices), 4), dtype=np.float32)
    colors[:, :3] = values
    attribute.data.foreach_set("color", colors.reshape(-1))
    return len(mesh.vertices), values.tolist()


def install_material_aov(material: bpy.types.Material) -> None:
    if not material.use_nodes or material.node_tree is None:
        raise ValueError(f"rendered material has no shader nodes: {material.name}")
    tree = material.node_tree
    for node in list(tree.nodes):
        if node.name == AOV_NAME or node.name == f"{AOV_NAME}_attribute":
            tree.nodes.remove(node)
    attribute = tree.nodes.new("ShaderNodeAttribute")
    attribute.name = f"{AOV_NAME}_attribute"
    attribute.attribute_name = ATTRIBUTE_NAME
    output = tree.nodes.new("ShaderNodeOutputAOV")
    output.name = AOV_NAME
    output.label = AOV_NAME
    # Color is non-spatial.  The attribute is a FLOAT_COLOR payload carrying
    # canonical XYZ in RGB, not a geometric Vector socket.
    tree.links.new(attribute.outputs["Color"], output.inputs["Color"])


def install_compositor_aov(scene: bpy.types.Scene) -> None:
    if not scene.use_nodes or scene.node_tree is None:
        raise ValueError("run official setup_blender_scene.py before installing canonical AOV")
    tree = scene.node_tree
    for node in list(tree.nodes):
        if node.name == COMPOSITOR_NODE_NAME:
            tree.nodes.remove(node)
    render_layer = next((node for node in tree.nodes if node.type == "R_LAYERS"), None)
    if render_layer is None or AOV_NAME not in render_layer.outputs:
        raise ValueError(f"render layer does not expose {AOV_NAME}")
    output = tree.nodes.new("CompositorNodeOutputFile")
    output.name = COMPOSITOR_NODE_NAME
    output.label = AOV_NAME
    output.format.file_format = "OPEN_EXR"
    output.format.color_mode = "RGB"
    output.format.color_depth = "16"
    output.base_path = ""
    output.file_slots[0].path = "canonical_position"
    tree.links.new(render_layer.outputs[AOV_NAME], output.inputs[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--metadata", type=pathlib.Path, required=True)
    args = parser.parse_args()
    bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
    scene = bpy.context.scene
    ensure_aov(scene.view_layers[0])
    objects = [obj for obj in scene.objects if obj.type == "MESH" and not obj.hide_render]
    if not objects:
        raise ValueError("no render-visible meshes in canonical scene")
    records = []
    all_positions = []
    processed_materials: set[str] = set()
    for obj in objects:
        count, positions = set_canonical_attribute(obj)
        all_positions.extend(positions)
        records.append({"object": obj.name, "vertices": count, "attribute": ATTRIBUTE_NAME})
        for slot in obj.material_slots:
            if slot.material is None:
                raise ValueError(f"rendered object has empty material slot: {obj.name}")
            if slot.material.name not in processed_materials:
                install_material_aov(slot.material)
                processed_materials.add(slot.material.name)
    install_compositor_aov(scene)
    output = args.output.resolve()
    metadata = args.metadata.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    points = np.asarray(all_positions, dtype=np.float32)
    metadata.write_text(json.dumps({
        "label": "CANONICAL CORRESPONDENCE INSTRUMENTATION",
        "input_blend": str(args.blend.resolve()),
        "output_blend": str(output),
        "attribute": ATTRIBUTE_NAME,
        "aov": AOV_NAME,
        "semantics": "canonical world position at each rasterized current-surface sample; barycentrically interpolated by mesh rasterization",
        "canonical_aabb_min": points.min(axis=0).tolist(),
        "canonical_aabb_max": points.max(axis=0).tolist(),
        "meshes": records,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
