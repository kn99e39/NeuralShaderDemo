"""Focused tests of frb_records (stdlib only; plain asserts, run as a script).

1. fps/gap   : effective FPS and 30/60-FPS gaps for known frame costs;
               non-positive or non-finite costs are rejected.
2. log parse : real Cycles debug-log slices (one cold render, one render after
               a mover change, Blender 5.1.0) give the logged values exactly;
               a slice with zero or two renders is rejected.
3. records   : derived fields (frame total, Blender overhead); unknown
               fields rejected; per-phase summary keeps the first value and
               reports the median, not the minimum.
4. schema    : validate_process_record catches a missing field, an unknown
               phase/state and a record that does not start cold.
5. serialize : JSON round trip is lossless; NaN is refused.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import frb_records as R  # noqa: E402

FX = os.path.join(HERE, "fixtures")


def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    raise AssertionError(f"{fn} did not raise {exc.__name__}")


def test_fps_gaps():
    g = R.fps_and_gaps(0.5)
    assert math.isclose(g["total_ms"], 500.0)
    assert math.isclose(g["effective_fps"], 2.0)
    assert math.isclose(g["gap_30fps"], 15.0)
    assert math.isclose(g["gap_60fps"], 30.0)
    g = R.fps_and_gaps(1.0 / 60.0)
    assert math.isclose(g["gap_60fps"], 1.0) and math.isclose(g["gap_30fps"], 0.5)
    for bad in (0.0, -1.0, float("nan"), float("inf"), None):
        _raises(lambda b=bad: R.fps_and_gaps(b))


def test_log_parse():
    moved = open(os.path.join(FX, "cycles_log_moved_render.txt"), encoding="utf-8").read()
    p = R.parse_cycles_render_log(moved)
    assert p["sync_s"] == 0.0003061
    assert p["cycles_total_s"] == 0.223534
    assert p["render_without_sync_s"] == 0.220859
    assert p["path_trace_s"] == 0.208615
    assert p["samples_rendered"] == 16
    assert p["bvh_tasks_handled"] == 0
    assert p["objects_synced"] == 105 and p["meshes_synced"] == 36
    assert math.isclose(p["device_update_s"], 0.223534 - 0.220859 - 0.0003061)

    cold = open(os.path.join(FX, "cycles_log_cold_render.txt"), encoding="utf-8").read()
    c = R.parse_cycles_render_log(cold)
    assert c["sync_s"] == 0.148368
    assert c["cycles_total_s"] == 0.571761
    assert c["render_without_sync_s"] == 0.240012
    assert c["path_trace_s"] == 0.214351
    assert c["samples_rendered"] == 16
    assert c["device_update_s"] > 0.1  # cold BVH build + device setup

    _raises(lambda: R.parse_cycles_render_log("no render here"))
    _raises(lambda: R.parse_cycles_render_log(moved + moved))
    # A static re-render has no object update: those fields come back None.
    static = "\n".join(l for l in moved.splitlines()
                       if "Total 105 objects" not in l and "Total 36 meshes" not in l and "Tasks handled" not in l)
    s = R.parse_cycles_render_log(static)
    assert s["objects_synced"] is None and s["bvh_tasks_handled"] is None


def test_stats_mem():
    assert R.parse_stats_mem_mb("Mem: 824M | Finished") == 824.0
    assert R.parse_stats_mem_mb("Mem: 1.5G | Updating Geometry BVH Mesh 9/36 | Building BVH") == 1536.0
    assert R.parse_stats_mem_mb("Mem: 512K | Initializing") == 0.5
    assert R.parse_stats_mem_mb("Finished") is None and R.parse_stats_mem_mb(None) is None


def test_records_and_summary():
    r = R.make_render_record(index=0, phase="cold", state="G0", changed=False, depsgraph_s=0.001,
                             render_call_s=0.9, cycles_total_s=0.6, path_trace_s=0.2, sync_s=0.15,
                             device_update_s=0.2, render_without_sync_s=0.25)
    assert math.isclose(r["frame_total_s"], 0.901)
    assert math.isclose(r["blender_overhead_s"], 0.3)
    _raises(lambda: R.make_render_record(index=0, bogus=1), KeyError)

    renders = [r]
    for i, ft in enumerate([0.50, 0.40, 0.30, 0.31, 0.29, 0.35]):
        renders.append(R.make_render_record(
            index=i + 1, phase="warmup" if i < 2 else "steady", state="G1" if i % 2 == 0 else "G0",
            changed=True, depsgraph_s=0.0, render_call_s=ft, cycles_total_s=ft - 0.1, path_trace_s=ft - 0.12,
            sync_s=0.0, device_update_s=0.01, render_without_sync_s=ft - 0.11))
    s = R.summarize_renders(renders)
    assert math.isclose(s["cold"]["setup_s"], 0.4)
    st = s["steady"]["frame_total_s"]
    assert st["n"] == 4 and math.isclose(st["first"], 0.30)
    assert math.isclose(st["median"], 0.305) and math.isclose(st["min"], 0.29)
    assert math.isclose(s["steady_fps"]["effective_fps"], 1 / 0.305)
    assert s["warmup"]["frame_total_s"]["n"] == 2


def _good_record():
    rr = [R.make_render_record(index=0, phase="cold", state="G0", changed=False),
          R.make_render_record(index=1, phase="steady", state="G1", changed=True)]
    return {"schema": R.SCHEMA_ID, "protocol_id": "x", "condition": {}, "environment": {},
            "scene": {}, "mover": {}, "renders": rr, "summary": {}}


def test_schema():
    R.validate_process_record(_good_record())
    bad = _good_record(); del bad["mover"]
    _raises(lambda: R.validate_process_record(bad))
    bad = _good_record(); bad["renders"][1]["phase"] = "fast"
    _raises(lambda: R.validate_process_record(bad))
    bad = _good_record(); bad["renders"][1]["state"] = "G2"
    _raises(lambda: R.validate_process_record(bad))
    bad = _good_record(); bad["renders"].reverse()
    _raises(lambda: R.validate_process_record(bad))
    bad = _good_record(); del bad["renders"][0]["sync_s"]
    _raises(lambda: R.validate_process_record(bad))


def test_serialize():
    rec = _good_record()
    rec["summary"] = R.summarize_renders([R.make_render_record(
        index=0, phase="cold", state="G0", changed=False, depsgraph_s=0.0, render_call_s=1.0,
        cycles_total_s=0.9, path_trace_s=0.5, sync_s=0.1, device_update_s=0.2, render_without_sync_s=0.6)])
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "r.json")
        R.dump_json(rec, p)
        assert R.load_json(p) == rec
        _raises(lambda: R.dump_json({"x": float("nan")}, p))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
