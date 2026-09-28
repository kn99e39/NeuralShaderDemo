"""Install stable triangle-ID and barycentric AOVs on an RNA scene copy.

The two AOVs together identify a material point as
``(object, source-polygon-index, barycentric-coordinates)``.  Unlike a
nearest-world-position query, this identity stays tied to the source mesh
topology when an armature deforms it.  Values are carried in non-spatial color
attributes so they are rasterized using the mesh's actual triangle corners.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import bpy
import numpy as np


TRIANGLE_ATTRIBUTE = "rna_triangle_identity"
BARYCENTRIC_ATTRIBUTE = "rna_barycentric"
TRIANGLE_AOV = "triangle_identity_aov"
BARYCENTRIC_AOV = "barycentric_aov"
TRIANGLE_NODE = "triangle_identity_out"
BARYCENTRIC_NODE = "barycentric_out"


def script_argv() -> list[str]:
    """Support both ``bpy`` Python and ``blender --python ... --`` invocations."""
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]


def ensure_aov(view_layer: bpy.types.ViewLayer, name: str) -> None:
    for item in list(view_layer.aovs):
        if item.name == name:
            view_layer.aovs.remove(item)
    view_layer.aovs.add().name = name


def triangulate_source_mesh(obj: bpy.types.Object) -> None:
    """Triangulate only the experiment scene copy through Blender core ops.

    ``bpy.ops`` is available in both the full Blender executable and the
    lightweight BPy wheel used by the official RNA environment; ``bmesh`` is
    not guaranteed by that wheel.
    """
    if all(len(polygon.vertices) == 3 for polygon in obj.data.polygons):
        return
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris(quad_method="FIXED", ngon_method="BEAUTY")
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.data.update()

def populate_identity(mesh: bpy.types.Mesh, object_code: int) -> dict[str, int]:

    for name in (TRIANGLE_ATTRIBUTE, BARYCENTRIC_ATTRIBUTE):
        existing = mesh.color_attributes.get(name)
        if existing is not None:
            mesh.color_attributes.remove(existing)
    triangle = mesh.color_attributes.new(TRIANGLE_ATTRIBUTE, "FLOAT_COLOR", "CORNER")
    barycentric = mesh.color_attributes.new(BARYCENTRIC_ATTRIBUTE, "FLOAT_COLOR", "CORNER")
    triangle_values = np.ones((len(mesh.loops), 4), dtype=np.float32)
    barycentric_values = np.zeros((len(mesh.loops), 4), dtype=np.float32)
    barycentric_values[:, 3] = 1.0
    triangles = 0
    for polygon in mesh.polygons:
        if len(polygon.vertices) != 3:
            # Do not fabricate an identity for an n-gon.  The fixed-target
            # selector accepts only triangles with exact source topology.
            for loop_index in polygon.loop_indices:
                triangle_values[loop_index, :3] = (-1.0, -1.0, -1.0)
            continue
        triangles += 1
        for local_corner, loop_index in enumerate(polygon.loop_indices):
            triangle_values[loop_index, :3] = (float(object_code), float(polygon.index), 0.0)
            barycentric_values[loop_index, local_corner] = 1.0
    triangle.data.foreach_set("color", triangle_values.reshape(-1))
    barycentric.data.foreach_set("color", barycentric_values.reshape(-1))
    return {"vertices": len(mesh.vertices), "source_polygons": len(mesh.polygons), "triangles": triangles, "object_code": object_code}


def install_material_output(material: bpy.types.Material, attribute_name: str, aov_name: str) -> None:
    if not material.use_nodes or material.node_tree is None:
        raise ValueError(f"rendered material has no node tree: {material.name}")
    tree = material.node_tree
    for node in list(tree.nodes):
        if node.name in {aov_name, f"{aov_name}_attribute"}:
            tree.nodes.remove(node)
    attribute = tree.nodes.new("ShaderNodeAttribute")
    attribute.name = f"{aov_name}_attribute"
    attribute.attribute_name = attribute_name
    output = tree.nodes.new("ShaderNodeOutputAOV")
    output.name = aov_name
    tree.links.new(attribute.outputs["Color"], output.inputs["Color"])


def install_compositor_output(scene: bpy.types.Scene, aov_name: str, node_name: str, file_stem: str) -> None:
    if not scene.use_nodes or scene.node_tree is None:
        raise ValueError("scene has no compositor node tree")
    tree = scene.node_tree
    for node in list(tree.nodes):
        if node.name == node_name:
            tree.nodes.remove(node)
    render_layer = next((node for node in tree.nodes if node.type == "R_LAYERS"), None)
    if render_layer is None or aov_name not in render_layer.outputs:
        raise ValueError(f"render layer does not expose {aov_name}")
    output = tree.nodes.new("CompositorNodeOutputFile")
    output.name = node_name
    output.format.file_format = "OPEN_EXR"
    output.format.color_mode = "RGB"
    output.format.color_depth = "16"
    output.base_path = ""
    output.file_slots[0].path = file_stem
    tree.links.new(render_layer.outputs[aov_name], output.inputs[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--metadata", type=pathlib.Path, required=True)
    parser.add_argument("--objects", nargs="+", default=["GEO-rain_scarf", "GEO-rain_top"])
    args = parser.parse_args(script_argv())
    bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
    scene = bpy.context.scene
    ensure_aov(scene.view_layers[0], TRIANGLE_AOV)
    ensure_aov(scene.view_layers[0], BARYCENTRIC_AOV)
    materials: dict[str, bpy.types.Material] = {}
    records: list[dict[str, object]] = []
    for object_code, name in enumerate(args.objects, start=1):
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "MESH":
            raise ValueError(f"missing required mesh: {name}")
        record: dict[str, object] = {"object": name}
        triangulate_source_mesh(obj)
        record.update(populate_identity(obj.data, object_code))
        records.append(record)
        for slot in obj.material_slots:
            if slot.material is not None:
                materials[slot.material.name] = slot.material
    for material in materials.values():
        install_material_output(material, TRIANGLE_ATTRIBUTE, TRIANGLE_AOV)
        install_material_output(material, BARYCENTRIC_ATTRIBUTE, BARYCENTRIC_AOV)
    install_compositor_output(scene, TRIANGLE_AOV, TRIANGLE_NODE, "triangle_identity")
    install_compositor_output(scene, BARYCENTRIC_AOV, BARYCENTRIC_NODE, "barycentric")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output.resolve()))
    args.metadata.write_text(
        json.dumps(
            {
                "label": "RAIN TOPOLOGY-DERIVED SURFACE IDENTITY INSTRUMENTATION",
                "input_blend": str(args.blend.resolve()),
                "output_blend": str(args.output.resolve()),
                "identity": "object code + explicitly triangulated source triangle index + rasterized barycentric coordinates",
                "topology_note": "A deterministic triangulation is applied only to this experiment scene copy; vertex indices, UVs, weights, materials, and surface geometry are preserved.",
                "triangle_attribute": TRIANGLE_ATTRIBUTE,
                "barycentric_attribute": BARYCENTRIC_ATTRIBUTE,
                "triangle_aov": TRIANGLE_AOV,
                "barycentric_aov": BARYCENTRIC_AOV,
                "objects": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
