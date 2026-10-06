"""Evidence renders for the G0 -> G1 change (protocol `evidence_images`).

blender --factory-startup -b --python render_evidence.py -- <protocol.json> <scene.blend> <out_dir> [camera]

Writes G0_s0, G1_s0 and G0_s1 as multilayer EXR (Combined + Depth, float32,
uncompressed) plus PNG previews (view transform as loaded). Not timed.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_common as B  # noqa: E402
import bpy  # noqa: E402


def main() -> None:
    args = B.script_args()
    protocol_path, scene_path, out_dir = args[:3]
    camera = args[3] if len(args) > 3 else None
    protocol = B.load_protocol(protocol_path)
    ev = protocol["evidence_images"]
    bpy.ops.wm.open_mainfile(filepath=scene_path, load_ui=False)
    B.enable_optix()
    scene = bpy.context.scene
    if camera:
        scene.camera = bpy.data.objects[camera]
    B.apply_render_contract(scene, protocol, ev["spp"], "original", persistent=False)
    scene.view_layers[0].use_pass_z = True
    mover = B.Mover(protocol)
    os.makedirs(out_dir, exist_ok=True)
    for name, state, seed in (("G0_s0", "G0", 0), ("G1_s0", "G1", 0), ("G0_s1", "G0", 1)):
        mover.set(state)
        scene.cycles.seed = seed
        bpy.context.view_layer.update()
        bpy.ops.render.render(write_still=False)
        img = bpy.data.images["Render Result"]
        st = scene.render.image_settings
        st.media_type = "MULTI_LAYER_IMAGE"  # Blender 5.x: multilayer is selected by media type
        st.file_format = "OPEN_EXR_MULTILAYER"
        st.color_depth = "32"
        st.exr_codec = "NONE"
        img.save_render(os.path.join(out_dir, name + ".exr"), scene=scene)
        st.media_type = "IMAGE"
        st.file_format = "PNG"
        st.color_depth = "8"
        img.save_render(os.path.join(out_dir, name + ".png"), scene=scene)
        print("EVIDENCE", name, flush=True)
    B.R.dump_json({"scene": scene_path, "scene_sha256": B.sha256(scene_path), "spp": ev["spp"],
                   "mover": mover.describe(), "camera": scene.camera.name,
                   "settings": B.effective_settings(scene), "environment": B.environment()},
                  os.path.join(out_dir, "evidence_meta.json"))


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
