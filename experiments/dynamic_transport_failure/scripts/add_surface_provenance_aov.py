"""Add a stable Rain scarf/top provenance AOV to a canonical scene copy."""

from __future__ import annotations

import argparse
import json
import pathlib

import bpy
import numpy as np


ATTRIBUTE = "rna_surface_provenance"
AOV = "surface_provenance_aov"
NODE = "surface_provenance_out"
CODES = {
    "GEO-rain_scarf": [1.0, 0.0, 0.0, 1.0],
    "GEO-rain_top": [0.0, 1.0, 0.0, 1.0],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--metadata", type=pathlib.Path, required=True)
    args = parser.parse_args()
    bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
    scene = bpy.context.scene
    view_layer = scene.view_layers[0]
    for item in list(view_layer.aovs):
        if item.name == AOV:
            view_layer.aovs.remove(item)
    view_layer.aovs.add().name = AOV
    materials: dict[str, bpy.types.Material] = {}
    records = []
    for name, code in CODES.items():
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "MESH":
            raise ValueError(f"missing required Rain provenance mesh: {name}")
        attribute = obj.data.color_attributes.get(ATTRIBUTE)
        if attribute is not None:
            obj.data.color_attributes.remove(attribute)
        attribute = obj.data.color_attributes.new(ATTRIBUTE, "FLOAT_COLOR", "POINT")
        values = np.tile(np.asarray(code, dtype=np.float32), (len(obj.data.vertices), 1))
        attribute.data.foreach_set("color", values.reshape(-1))
        records.append({"object": name, "vertices": len(obj.data.vertices), "code": code})
        for slot in obj.material_slots:
            if slot.material is not None:
                materials[slot.material.name] = slot.material
    for material in materials.values():
        if not material.use_nodes or material.node_tree is None:
            raise ValueError(f"required material has no nodes: {material.name}")
        tree = material.node_tree
        for node in list(tree.nodes):
            if node.name in {AOV, f"{AOV}_attribute"}:
                tree.nodes.remove(node)
        attribute = tree.nodes.new("ShaderNodeAttribute")
        attribute.name = f"{AOV}_attribute"
        attribute.attribute_name = ATTRIBUTE
        output = tree.nodes.new("ShaderNodeOutputAOV")
        output.name = AOV
        tree.links.new(attribute.outputs["Color"], output.inputs["Color"])
    if not scene.use_nodes or scene.node_tree is None:
        raise ValueError("scene has no compositor")
    tree = scene.node_tree
    for node in list(tree.nodes):
        if node.name == NODE:
            tree.nodes.remove(node)
    render_layer = next((node for node in tree.nodes if node.type == "R_LAYERS"), None)
    if render_layer is None or AOV not in render_layer.outputs:
        raise ValueError(f"render layer does not expose {AOV}")
    output = tree.nodes.new("CompositorNodeOutputFile")
    output.name = NODE
    output.format.file_format = "OPEN_EXR"
    output.format.color_mode = "RGB"
    output.format.color_depth = "16"
    output.base_path = ""
    output.file_slots[0].path = "surface_provenance"
    tree.links.new(render_layer.outputs[AOV], output.inputs[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output.resolve()))
    args.metadata.write_text(json.dumps({"attribute": ATTRIBUTE, "aov": AOV, "codes": CODES, "meshes": records}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
