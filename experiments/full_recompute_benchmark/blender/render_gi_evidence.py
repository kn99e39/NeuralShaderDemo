"""High-spp G0/G1 evidence, optionally with the mover hidden from camera rays (not timed).

blender --factory-startup -b --python render_gi_evidence.py -- <protocol.json> <scene.blend> <out_dir> <spp> <visible|camera_invisible>

`camera_invisible` sets visible_camera = False on the mover (the `1M` instancer;
Cycles propagates it to the instanced rear car). The rear car then never
appears in a camera ray's first hit, but still casts shadows, occludes,
reflects and bounces light, so every G0/G1 difference in the image is a
transport change on surfaces that did not move. Writes G0_s0, G1_s0 and
G0_s1 (seed 1, noise floor) as multilayer EXR (Combined + Depth) plus PNG.
Render contract as in the protocol (no denoising, no adaptive sampling).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_common as B  # noqa: E402
import bpy  # noqa: E402


def main() -> None:
    protocol_path, scene_path, out_dir, spp, mode = B.script_args()[:5]
    if mode not in ("visible", "camera_invisible"):
        raise ValueError(mode)
    protocol = B.load_protocol(protocol_path)
    bpy.ops.wm.open_mainfile(filepath=scene_path, load_ui=False)
    B.enable_optix()
    scene = bpy.context.scene
    B.apply_render_contract(scene, protocol, int(spp), "original", persistent=True)
    scene.view_layers[0].use_pass_z = True
    mover = B.Mover(protocol)
    mover.obj.visible_camera = mode == "visible"
    os.makedirs(out_dir, exist_ok=True)
    for name, state, seed in (("G0_s0", "G0", 0), ("G1_s0", "G1", 0), ("G0_s1", "G0", 1)):
        mover.set(state)
        scene.cycles.seed = seed
        bpy.context.view_layer.update()
        bpy.ops.render.render(write_still=False)
        img = bpy.data.images["Render Result"]
        st = scene.render.image_settings
        st.media_type = "MULTI_LAYER_IMAGE"
        st.file_format = "OPEN_EXR_MULTILAYER"
        st.color_depth = "32"
        st.exr_codec = "ZIP"
        img.save_render(os.path.join(out_dir, name + ".exr"), scene=scene)
        st.media_type = "IMAGE"
        st.file_format = "PNG"
        st.color_depth = "8"
        img.save_render(os.path.join(out_dir, name + ".png"), scene=scene)
        print("GI_EVIDENCE", mode, spp, name, flush=True)
    B.R.dump_json({"scene": scene_path, "scene_sha256": B.sha256(scene_path), "spp": int(spp),
                   "mover_mode": mode, "mover": mover.describe(), "camera": scene.camera.name,
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
