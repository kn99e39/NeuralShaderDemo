"""Render Rain GT plus fixed-target AOVs without constructing an RNA model."""
from __future__ import annotations
import argparse
import pathlib
import sys
import bpy

PASS_OUTPUTS = {"current_normal": "Normal", "current_position": "Position", "diffuse_direct": "DiffDir"}

def argv() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]

def output_node(tree: bpy.types.NodeTree, name: str, socket: str, directory: pathlib.Path) -> None:
    for node in list(tree.nodes):
        if node.name == name:
            tree.nodes.remove(node)
    layer = next(node for node in tree.nodes if node.type == "R_LAYERS")
    if socket not in layer.outputs:
        raise ValueError(f"render layer does not expose {socket}")
    node = tree.nodes.new("CompositorNodeOutputFile")
    node.name = name
    node.base_path = str(directory)
    node.format.file_format = "OPEN_EXR"
    node.format.color_mode = "RGB"
    node.format.color_depth = "32"
    node.file_slots[0].path = name
    tree.links.new(layer.outputs[socket], node.inputs[0])

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    args = parser.parse_args(argv())
    bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
    scene = bpy.context.scene
    layer = scene.view_layers[0]
    layer.use_pass_normal = True
    layer.use_pass_position = True
    layer.use_pass_diffuse_direct = True
    if not scene.use_nodes or scene.node_tree is None:
        raise ValueError("expected compositor AOV instrumentation")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene.cycles.samples = args.samples
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "32"
    scene.render.filepath = str(output / "ref_0.exr")
    for stem, socket in PASS_OUTPUTS.items():
        output_node(scene.node_tree, stem, socket, output)
    for node in scene.node_tree.nodes:
        if node.type == "OUTPUT_FILE" and node.name in {"surface_uv_out", "canonical_position_out", "surface_provenance_out"}:
            node.base_path = str(output)
    bpy.ops.render.render(write_still=True)

if __name__ == "__main__":
    main()