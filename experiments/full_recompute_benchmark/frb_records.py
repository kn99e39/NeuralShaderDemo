"""Timing records for the full-recomputation benchmark (stdlib only).

Shared by the Blender-side timing script, the host driver and the report.
Nothing here imports bpy or numpy, so it runs under Blender's Python and
under the system Python alike.

Timing phases (protocol `timing_fields`):
  A  load_s            bpy.ops.wm.open_mainfile wall time
  B  cold setup        first render's cycles_total_s - path_trace_s
  C  depsgraph_s       set mover transform + view_layer.update() wall time
  D  sync_s            Cycles "Total time spent synchronizing data"
  E  device_update_s   cycles_total_s - render_without_sync_s - sync_s
                       (BVH / IAS build or refit together with every other
                       Cycles device manager; the debug log does not split it)
  F  path_trace_s      Cycles render-scheduler "Path Tracing" wall time
  G  frame_total_s     depsgraph_s + render_call_s (the bpy render call wall
                       time, which also holds Blender's render-pipeline
                       overhead outside the Cycles session)
"""

from __future__ import annotations

import json
import math
import re
import statistics

FRAME_MS_30 = 1000.0 / 30.0
FRAME_MS_60 = 1000.0 / 60.0

# Fields every per-render timing record carries.
RENDER_RECORD_FIELDS = (
    "index",            # position in the process's render sequence
    "phase",            # cold | warmup | steady | static
    "state",            # G0 | G1
    "changed",          # mover transform changed since the previous render
    "depsgraph_s",      # C
    "render_call_s",    # wall time of bpy.ops.render.render
    "frame_total_s",    # G
    "sync_s",           # D
    "device_update_s",  # E
    "path_trace_s",     # F
    "cycles_total_s",
    "render_without_sync_s",
    "blender_overhead_s",  # render_call_s - cycles_total_s
    "samples_rendered",
    "bvh_tasks_handled",   # BLAS builds Cycles reported for this render (None = no object update)
    "objects_synced",      # "Total N objects." (None = object manager not updated)
    "meshes_synced",
    "cycles_mem_peak_mb",  # max 'Mem:' over this render's render_stats strings (Cycles device allocations)
    "rss_mb",
)

PROCESS_RECORD_FIELDS = (
    "schema",
    "protocol_id",
    "condition",
    "environment",
    "scene",
    "mover",
    "renders",
    "summary",
)

SCHEMA_ID = "frb.timing.v1"


def fps_and_gaps(total_s: float) -> dict:
    """Effective frame rate and real-time gap for one frame cost."""
    if not (isinstance(total_s, (int, float)) and total_s > 0 and math.isfinite(total_s)):
        raise ValueError(f"frame cost must be a positive finite number, got {total_s!r}")
    total_ms = total_s * 1000.0
    return {
        "total_ms": total_ms,
        "effective_fps": 1.0 / total_s,
        "gap_30fps": total_ms / FRAME_MS_30,
        "gap_60fps": total_ms / FRAME_MS_60,
    }


def summarize(values) -> dict:
    """Repetition statistics. The first value is kept separately, never dropped silently."""
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"n": 0}
    q = statistics.quantiles(vals, n=4, method="inclusive") if len(vals) >= 2 else [vals[0]] * 3
    return {
        "n": len(vals),
        "first": vals[0],
        "median": statistics.median(vals),
        "mean": statistics.fmean(vals),
        "min": min(vals),
        "max": max(vals),
        "p25": q[0],
        "p75": q[2],
    }


_FLOAT = r"([0-9]+(?:\.[0-9]*)?(?:[eE][-+]?[0-9]+)?)"
_PATTERNS = {
    "sync_s": re.compile(r"Total time spent synchronizing data:\s*" + _FLOAT),
    "cycles_total_s": re.compile(r"Total render time:\s*" + _FLOAT),
    "render_without_sync_s": re.compile(r"Render time \(without synchronization\):\s*" + _FLOAT),
    "path_trace_s": re.compile(r"\|\s*Path Tracing\s+" + _FLOAT + r"\s+" + _FLOAT),
    "samples_rendered": re.compile(r"\|\s*Rendered ([0-9]+) samples in " + _FLOAT + " seconds\s*$", re.M),
    "bvh_tasks_handled": re.compile(r"Tasks handled:\s*([0-9]+)"),
    "objects_synced": re.compile(r"Total ([0-9]+) objects\."),
    "meshes_synced": re.compile(r"Total ([0-9]+) meshes\."),
}


def parse_cycles_render_log(chunk: str) -> dict:
    """Parse the Cycles debug-log lines written by exactly one render.

    Expects the text `--log cycles --log-level debug` produced between two
    consecutive renders. Raises if the chunk does not hold exactly one
    completed render, so a mis-sliced log cannot silently produce numbers.
    """
    totals = _PATTERNS["cycles_total_s"].findall(chunk)
    if len(totals) != 1:
        raise ValueError(f"expected exactly one 'Total render time' line, found {len(totals)}")
    out: dict = {}
    for key in ("sync_s", "cycles_total_s", "render_without_sync_s"):
        m = _PATTERNS[key].findall(chunk)
        if len(m) != 1:
            raise ValueError(f"expected exactly one {key} line, found {len(m)}")
        out[key] = float(m[0])
    pt = _PATTERNS["path_trace_s"].findall(chunk)
    if len(pt) != 1:
        raise ValueError(f"expected exactly one Path Tracing summary line, found {len(pt)}")
    out["path_trace_s"] = float(pt[0][0])
    sr = _PATTERNS["samples_rendered"].findall(chunk)
    out["samples_rendered"] = int(sr[-1][0]) if sr else None
    for key in ("bvh_tasks_handled", "objects_synced", "meshes_synced"):
        m = _PATTERNS[key].findall(chunk)
        out[key] = int(m[-1]) if m else None
    out["device_update_s"] = out["cycles_total_s"] - out["render_without_sync_s"] - out["sync_s"]
    return out


_MEM = re.compile(r"Mem:\s*([0-9.]+)([KMG])")


def parse_stats_mem_mb(stats: str | None):
    """Memory figure of a Cycles render_stats string ('Mem: 824M | Finished').

    In a background Cycles render this is Cycles' own tracked device
    allocations (scene data, BVH, render buffers, path states); the caller
    keeps the maximum over one render's strings.
    """
    if not stats:
        return None
    m = _MEM.findall(stats)
    if not m:
        return None
    val, unit = m[-1]
    return float(val) * {"K": 1 / 1024.0, "M": 1.0, "G": 1024.0}[unit]


def make_render_record(**kw) -> dict:
    rec = {k: kw.get(k) for k in RENDER_RECORD_FIELDS}
    unknown = set(kw) - set(RENDER_RECORD_FIELDS)
    if unknown:
        raise KeyError(f"unknown render-record fields: {sorted(unknown)}")
    if rec["render_call_s"] is not None and rec["depsgraph_s"] is not None:
        rec["frame_total_s"] = rec["depsgraph_s"] + rec["render_call_s"]
    if rec["render_call_s"] is not None and rec["cycles_total_s"] is not None:
        rec["blender_overhead_s"] = rec["render_call_s"] - rec["cycles_total_s"]
    return rec


def summarize_renders(renders: list) -> dict:
    """Per-phase statistics plus FPS/gap of the steady geometry-changed frames."""
    out: dict = {}
    cold = [r for r in renders if r["phase"] == "cold"]
    if cold:
        c = cold[0]
        out["cold"] = {
            "frame_total_s": c["frame_total_s"],
            "cycles_total_s": c["cycles_total_s"],
            "path_trace_s": c["path_trace_s"],
            "setup_s": c["cycles_total_s"] - c["path_trace_s"],
            "sync_s": c["sync_s"],
            "device_update_s": c["device_update_s"],
            "blender_overhead_s": c["blender_overhead_s"],
        }
    keys = ("frame_total_s", "render_call_s", "depsgraph_s", "sync_s", "device_update_s",
            "path_trace_s", "cycles_total_s", "blender_overhead_s")
    for phase in ("warmup", "steady", "static"):
        rs = [r for r in renders if r["phase"] == phase]
        if rs:
            out[phase] = {k: summarize([r[k] for r in rs]) for k in keys}
    steady = out.get("steady")
    if steady and steady["frame_total_s"]["n"]:
        out["steady_fps"] = fps_and_gaps(steady["frame_total_s"]["median"])
        out["steady_fps_cycles_only"] = fps_and_gaps(steady["cycles_total_s"]["median"])
        out["steady_fps_path_trace_only"] = fps_and_gaps(steady["path_trace_s"]["median"])
    return out


def validate_process_record(rec: dict) -> None:
    missing = [k for k in PROCESS_RECORD_FIELDS if k not in rec]
    if missing:
        raise ValueError(f"process record missing fields: {missing}")
    if rec["schema"] != SCHEMA_ID:
        raise ValueError(f"schema {rec['schema']!r} != {SCHEMA_ID!r}")
    for i, r in enumerate(rec["renders"]):
        missing = [k for k in RENDER_RECORD_FIELDS if k not in r]
        if missing:
            raise ValueError(f"render {i} missing fields: {missing}")
        if r["phase"] not in ("cold", "warmup", "steady", "static"):
            raise ValueError(f"render {i} has unknown phase {r['phase']!r}")
        if r["state"] not in ("G0", "G1"):
            raise ValueError(f"render {i} has unknown state {r['state']!r}")
    phases = [r["phase"] for r in rec["renders"]]
    if phases.count("cold") != 1 or phases[0] != "cold":
        raise ValueError("a process record must start with exactly one cold render")


def dump_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, allow_nan=False)
        f.write("\n")


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
