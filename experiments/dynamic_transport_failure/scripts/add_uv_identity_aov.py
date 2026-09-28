"""Add a stable mesh-UV identity AOV to a Rain scene copy.

This avoids BPy's unavailable CORNER color-attribute storage.  The rasterized
UV is a topology-derived material coordinate, not a nearest-position match.
Its one-to-one locality is audited before it is admitted as correspondence.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import bpy


AOV = "surface_uv_aov"
NODE = "surface_uv_out"


def script_argv() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--metadata", type=pathlib.Path, required=True)
    args = parser.parse_args(script_argv())
    bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
    scene = bpy.context.scene
    layer = scene.view_layers[0]
    for item in list(layer.aovs):
        if item.name == AOV:
            layer.aovs.remove(item)
    layer.aovs.add().name = AOV
    objects = [bpy.data.objects[name] for name in ("GEO-rain_scarf", "GEO-rain_top")]
    materials = {slot.material.name: slot.material for obj in objects for slot in obj.material_slots if slot.material}
    for material in materials.values():
        tree = material.node_tree
        if tree is None:
            raise ValueError(f"material has no nodes: {material.name}")
        for node in list(tree.nodes):
            if node.name in {AOV, f"{AOV}_texcoord"}:
                tree.nodes.remove(node)
        texcoord = tree.nodes.new("ShaderNodeTexCoord")
        texcoord.name = f"{AOV}_texcoord"
        output = tree.nodes.new("ShaderNodeOutputAOV")
        output.name = AOV
        tree.links.new(texcoord.outputs["UV"], output.inputs["Color"])
    tree = scene.node_tree
    if tree is None:
        raise ValueError("scene has no compositor")
    for node in list(tree.nodes):
        if node.name == NODE:
            tree.nodes.remove(node)
    render_layer = next(node for node in tree.nodes if node.type == "R_LAYERS")
    output = tree.nodes.new("CompositorNodeOutputFile")
    output.name = NODE
    output.format.file_format = "OPEN_EXR"
    output.format.color_mode = "RGB"
    output.format.color_depth = "16"
    output.base_path = ""
    output.file_slots[0].path = "surface_uv"
    tree.links.new(render_layer.outputs[AOV], output.inputs[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output.resolve()))
    args.metadata.write_text(json.dumps({"label": "RAIN UV SURFACE IDENTITY AOV", "aov": AOV, "identity": "mesh UV; use only after one-to-one locality audit", "objects": [{"name": obj.name, "uv_layers": [layer.name for layer in obj.data.uv_layers], "polygons": len(obj.data.polygons)} for obj in objects]}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
