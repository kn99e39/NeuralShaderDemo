"""Reviewer preview of one scene variant in G0 (not timed).

blender --factory-startup -b --python render_preview.py -- <protocol.json> <scene.blend> <out.png> <camera> <spp> [state]
Render contract as in the protocol, written at 50% size as PNG.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_common as B  # noqa: E402
import bpy  # noqa: E402


def main() -> None:
    args = B.script_args()
    protocol_path, scene_path, out_png, camera, spp = args[:5]
    state = args[5] if len(args) > 5 else "G0"
    protocol = B.load_protocol(protocol_path)
    bpy.ops.wm.open_mainfile(filepath=scene_path, load_ui=False)
    B.enable_optix()
    scene = bpy.context.scene
    scene.camera = bpy.data.objects[camera]
    B.apply_render_contract(scene, protocol, int(spp), "original", persistent=False)
    B.Mover(protocol).set(state)
    scene.render.resolution_percentage = 50
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = out_png
    bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=True)
    print("PREVIEW", out_png, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)
