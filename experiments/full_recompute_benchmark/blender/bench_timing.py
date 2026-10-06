"""One timing condition in one Blender process (run with --factory-startup -b).

blender --factory-startup -b --log cycles --log-level debug --log-file <log>
        --python bench_timing.py -- <protocol.json> <condition.json> <out.json>

The condition JSON names: variant, scene path, spp, bounce preset, persistent
flag, repetition counts. The script

  1. opens the scene (A, timed),
  2. applies the render contract,
  3. renders once cold in G0,
  4. renders `warmup` frames alternating G1/G0,
  5. renders `steady` frames alternating G1/G0 (every frame follows a
     mover change),
  6. renders `static` frames with no change (control),

and parses each render's slice of the Cycles debug log.
"""

from __future__ import annotations

import os
import sys
import time

T_SCRIPT = time.perf_counter()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_common as B  # noqa: E402
import bpy  # noqa: E402

R = B.R


def main() -> None:
    protocol_path, condition_path, out_path = B.script_args()[:3]
    protocol = B.load_protocol(protocol_path)
    cond = B.load_protocol(condition_path)
    logpath = cond["log_path"]

    t0 = time.perf_counter()
    bpy.ops.wm.open_mainfile(filepath=cond["scene_path"], load_ui=False)
    load_s = time.perf_counter() - t0
    mem_after_load = B.process_memory_mb()

    devices = B.enable_optix()
    scene = bpy.context.scene
    if cond.get("camera"):
        scene.camera = bpy.data.objects[cond["camera"]]
    settings = B.apply_render_contract(scene, protocol, cond["spp"], cond["bounce_preset"],
                                       cond["persistent"])
    if cond.get("compositing_override"):
        scene.render.use_compositing = True
        settings = B.effective_settings(scene)
    mover = B.Mover(protocol)

    last_stats = {"s": None, "max_mb": None}

    def on_stats(s):
        last_stats["s"] = s
        mb = R.parse_stats_mem_mb(s)
        if mb is not None and (last_stats["max_mb"] is None or mb > last_stats["max_mb"]):
            last_stats["max_mb"] = mb

    bpy.app.handlers.render_stats.append(on_stats)

    plan = [("cold", "G0")]
    st = "G0"
    for phase in ("warmup", "steady"):
        for _ in range(cond[phase]):
            st = "G1" if st == "G0" else "G0"
            plan.append((phase, st))
    plan += [("static", st)] * cond["static"]

    renders = []
    for i, (phase, state) in enumerate(plan):
        changed = mover.set(state) if phase != "cold" else False
        ta = time.perf_counter()
        bpy.context.view_layer.update()
        tb = time.perf_counter()
        pos = os.path.getsize(logpath)
        last_stats["s"] = None
        last_stats["max_mb"] = None
        bpy.ops.render.render(write_still=False)
        tc = time.perf_counter()
        parsed = _parse_with_retry(logpath, pos)
        renders.append(R.make_render_record(
            index=i, phase=phase, state=state, changed=changed,
            depsgraph_s=tb - ta, render_call_s=tc - tb,
            cycles_mem_peak_mb=last_stats["max_mb"],
            rss_mb=B.process_memory_mb().get("working_set_mb"),
            **parsed))
        r = renders[-1]
        print(f"FRB {i:3d} {phase:6s} {state} frame {r['frame_total_s']:.4f}s "
              f"pt {r['path_trace_s']:.4f}s sync {r['sync_s']:.4f}s dev {r['device_update_s']:.4f}s",
              flush=True)

    # Accounting after timing so its depsgraph work cannot touch the timed renders.
    mover.set("G0")
    bpy.context.view_layer.update()
    acct = B.scene_accounting(scene, cond["scene_path"])

    rec = {
        "schema": R.SCHEMA_ID,
        "protocol_id": protocol["protocol_id"],
        "condition": cond,
        "environment": dict(B.environment(), optix_devices=devices,
                            script_start_to_main_s=t0 - T_SCRIPT),
        "scene": {"path": cond["scene_path"], "sha256": B.sha256(cond["scene_path"]),
                  "load_s": load_s, "settings": settings, "accounting": acct,
                  "memory_after_load": mem_after_load,
                  "memory_end": B.process_memory_mb(),
                  "last_render_stats": last_stats["s"]},
        "mover": mover.describe(),
        "renders": renders,
    }
    rec["summary"] = R.summarize_renders(renders)
    R.validate_process_record(rec)
    R.dump_json(rec, out_path)
    print("FRB_DONE", out_path, flush=True)


def _read_from(path: str, pos: int) -> str:
    with open(path, "rb") as f:
        f.seek(pos)
        return f.read().decode("utf-8", "replace")


def _parse_with_retry(path: str, pos: int) -> dict:
    """The log's last lines can land a few ms after the render call returns."""
    for _ in range(100):
        try:
            return R.parse_cycles_render_log(_read_from(path, pos))
        except ValueError:
            time.sleep(0.02)
    return R.parse_cycles_render_log(_read_from(path, pos))


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
