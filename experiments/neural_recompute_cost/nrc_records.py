"""Timing records for the neural full-recompute cost batch (stdlib only).

Shared by the instrumented training/data wrappers (native Windows 8DNA venv,
WSL RNA venv), the chain driver and the report.  Nothing here imports torch,
numpy, Mitsuba or Lightning, so the accounting can be tested on its own.

Event log
---------
Every instrumented process appends JSON lines to one events file:

    {"t": <time.time()>, "pc": <perf_counter()>, "stage": <stage>, "event": <name>,
     "kind": "begin" | "end" | "point", ...extra}

A leaf phase is a begin/end pair with the same ``event`` and ``key`` (the
key distinguishes repeated phases, e.g. one per epoch or per view).  Wall
time is ``t`` (epoch seconds; comparable across processes on one host, and
across the Windows host and WSL after the recorded clock offset is applied);
``pc`` is used only for durations inside one process.

Phases (protocol ``timing.phases``)
-----------------------------------
process level : process_setup (interpreter start -> imports done),
                scene_prep (geometry load for the current state),
                dataset_init / data_load, model_init, fit, export
epoch level   : path_generation (8DNA online path samples: dataset.reload),
                resample (8DNA shuffle), optimization (first batch start ->
                last batch end), validation, sanity_validation,
                checkpoint_write (the method's own ModelCheckpoint/final
                saves), snapshot_write (instrumentation only: the per-epoch
                copy evaluated offline)
data level    : view_render (RNA H5 target rendering, one per view)

Anything inside an interval that no leaf phase covers is reported as
``other`` and never silently dropped.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

SCHEMA = "nrc_timing_record/v1"
FRAME_S_30 = 1.0 / 30.0
FRAME_S_60 = 1.0 / 60.0

# Descriptive latency categories, declared in the protocol before any
# evidence run (batch section 14).  They are labels, not success criteria.
CATEGORY_BOUNDS_S = {"A_frame_scale": 0.1, "B_human_timescale": 10.0}

EPOCH_LEAVES = ("path_generation", "resample", "optimization", "validation", "checkpoint_write", "snapshot_write")

# Top-level fields the final machine-readable record must carry (batch section 13).
RECORD_FIELDS = (
    "schema", "method", "track", "upstream_commit", "project_commit", "project_dirty", "configuration",
    "initialization", "geometry_state", "host", "start_timestamp", "phase_durations_s", "stages",
    "cumulative", "checkpoints", "quality_trace", "first_recovery", "total_schedule_s",
    "inference", "memory", "recovery_verdict",
)


# --- writing ---------------------------------------------------------------

class EventLog:
    """Append-only JSON-lines event writer (one line per event, flushed)."""

    def __init__(self, path, stage: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stage = stage
        self._fh = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def emit(self, event: str, kind: str = "point", **extra) -> dict:
        row = {"t": time.time(), "pc": time.perf_counter(), "stage": self.stage, "event": event, "kind": kind}
        row.update(extra)
        line = json.dumps(row, allow_nan=False, default=_json_default)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()
        return row

    def begin(self, event: str, **extra) -> dict:
        return self.emit(event, "begin", **extra)

    def end(self, event: str, **extra) -> dict:
        return self.emit(event, "end", **extra)

    def close(self) -> None:
        with self._lock:
            self._fh.close()


class GpuMemoryPoller(threading.Thread):
    """Polls whole-GPU memory.used (MiB) with nvidia-smi; keeps the peak.

    Under WDDM (Windows) and WSL per-process GPU memory is unavailable, so
    this is the whole-GPU figure including the CUDA/OptiX context and any
    other GPU client; the baseline before the job is recorded alongside.
    """

    def __init__(self, interval_s: float = 2.0, exe: str = "nvidia-smi"):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.exe = exe
        self.samples = 0
        self.peak_mib = None
        self.baseline_mib = self.read()
        self._halt = threading.Event()

    def read(self):
        try:
            out = subprocess.run([self.exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
            return float(out[0]) if out else None
        except (OSError, ValueError, subprocess.SubprocessError):
            return None

    def run(self) -> None:
        while not self._halt.is_set():
            v = self.read()
            if v is not None:
                self.samples += 1
                self.peak_mib = v if self.peak_mib is None else max(self.peak_mib, v)
            self._halt.wait(self.interval_s)

    def halt(self) -> dict:
        self._halt.set()
        return {"gpu_whole_baseline_mib": self.baseline_mib, "gpu_whole_peak_mib": self.peak_mib,
                "gpu_whole_peak_over_baseline_mib": (None if self.peak_mib is None or self.baseline_mib is None
                                                     else self.peak_mib - self.baseline_mib),
                "gpu_poll_samples": self.samples, "gpu_poll_interval_s": self.interval_s}


def host_peak_memory_mb():
    """Peak resident memory of this process in MB (Windows: PeakWorkingSetSize)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.WinDLL("kernel32")
        psapi = ctypes.WinDLL("psapi")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return pmc.PeakWorkingSetSize / 2 ** 20
        return None
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # Linux: KiB


def process_create_time():
    """Wall-clock creation time of this process (interpreter start), or None.

    Lets a wrapper account for interpreter start-up and imports, which happen
    before its first line runs.
    """
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        c, e, k, u = (wintypes.FILETIME() for _ in range(4))
        if not k32.GetProcessTimes(k32.GetCurrentProcess(), ctypes.byref(c), ctypes.byref(e), ctypes.byref(k), ctypes.byref(u)):
            return None
        ticks = (c.dwHighDateTime << 32) | c.dwLowDateTime  # 100 ns since 1601-01-01
        return ticks / 1e7 - 11644473600.0
    try:
        # /proc/stat's btime has 1 s resolution; /proc/uptime (10 ms) avoids that error
        start_ticks = int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19])
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        return time.time() - (uptime - start_ticks / os.sysconf("SC_CLK_TCK"))
    except (OSError, ValueError, IndexError):
        return None


# --- reading and accounting -------------------------------------------------

def load_events(path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for k in ("t", "stage", "event", "kind"):
                if k not in row:
                    raise ValueError(f"{path}:{n}: event lacks '{k}'")
            if row["kind"] not in ("begin", "end", "point"):
                raise ValueError(f"{path}:{n}: bad kind {row['kind']!r}")
            rows.append(row)
    return rows


def intervals(events: list[dict]) -> list[dict]:
    """Pair begin/end events by (stage, event, key) in order.

    Raises on an end without a begin, a begin opened twice, or an end that
    precedes its begin; an unclosed begin is returned with ``end`` None so a
    crashed process is visible rather than dropped.
    """
    open_: dict = {}
    out = []
    for e in events:
        if e["kind"] == "point":
            continue
        k = (e["stage"], e["event"], json.dumps(e.get("key"), sort_keys=True))
        if e["kind"] == "begin":
            if k in open_:
                raise ValueError(f"phase {k} opened twice")
            open_[k] = e
        else:
            b = open_.pop(k, None)
            if b is None:
                raise ValueError(f"phase {k} ended without a begin")
            if e["t"] < b["t"]:
                raise ValueError(f"phase {k} ends before it begins")
            dur = (e["pc"] - b["pc"]) if ("pc" in e and "pc" in b) else (e["t"] - b["t"])
            out.append({"stage": e["stage"], "phase": e["event"], "key": b.get("key"), "begin": b["t"], "end": e["t"],
                        "seconds": dur, "begin_event": b, "end_event": e})
    for k, b in open_.items():
        out.append({"stage": b["stage"], "phase": b["event"], "key": b.get("key"), "begin": b["t"], "end": None,
                    "seconds": None, "begin_event": b, "end_event": None})
    out.sort(key=lambda r: r["begin"])
    return out


def phase_totals(ivs: list[dict], stage: str | None = None) -> dict:
    """Sum of closed intervals per phase (optionally one stage); counts included."""
    tot: dict = {}
    for r in ivs:
        if r["seconds"] is None or (stage is not None and r["stage"] != stage):
            continue
        d = tot.setdefault(r["phase"], {"seconds": 0.0, "count": 0})
        d["seconds"] += r["seconds"]
        d["count"] += 1
    return tot


def within(ivs: list[dict], phase: str, begin: float, end: float, stage: str | None = None) -> list[dict]:
    """Closed intervals of `phase` lying inside [begin, end] (wall clock)."""
    eps = 1e-6
    return [r for r in ivs if r["phase"] == phase and r["seconds"] is not None
            and (stage is None or r["stage"] == stage) and r["begin"] >= begin - eps and r["end"] <= end + eps]


def epoch_table(ivs: list[dict], stage: str, events: list[dict] | None = None) -> list[dict]:
    """Per-epoch phase accounting from 'epoch' intervals and their leaf phases.

    other = epoch wall time - sum(leaf phases inside the epoch); it is kept
    so the per-epoch rows always add up to the epoch's wall time.  Point
    events with a matching epoch key (val metrics, checkpoint availability)
    are attached.
    """
    rows = []
    points = {}
    for e in events or []:
        if e["kind"] == "point" and e["stage"] == stage and "epoch" in e:
            points.setdefault(e["epoch"], []).append(e)
    for ep in [r for r in ivs if r["phase"] == "epoch" and r["stage"] == stage]:
        if ep["seconds"] is None:
            rows.append({"epoch": ep["key"], "incomplete": True, "begin": ep["begin"]})
            continue
        row = {"epoch": ep["key"], "begin": ep["begin"], "end": ep["end"], "wall_s": ep["seconds"]}
        leaf_sum = 0.0
        for leaf in EPOCH_LEAVES:
            s = sum(r["seconds"] for r in within(ivs, leaf, ep["begin"], ep["end"], stage))
            row[leaf + "_s"] = s
            leaf_sum += s
        row["other_s"] = ep["seconds"] - leaf_sum
        ck = within(ivs, "checkpoint_write", ep["begin"], ep["end"], stage)
        row["checkpoint_available_t"] = max((r["end"] for r in ck), default=None)
        for p in points.get(ep["key"], []):
            for k, v in p.items():
                if k not in ("t", "pc", "stage", "event", "kind", "epoch"):
                    row.setdefault(p["event"], {})[k] = v
        rows.append(row)
    rows.sort(key=lambda r: r["begin"])
    return rows


def check_epoch_accounting(rows: list[dict], tol_s: float = 0.05) -> None:
    """Leaves inside an epoch must not exceed the epoch (no double counting)."""
    for r in rows:
        if r.get("incomplete"):
            continue
        if r["other_s"] < -tol_s:
            raise ValueError(f"epoch {r['epoch']}: leaf phases exceed epoch wall time by {-r['other_s']:.3f} s")


def stage_table(chain_events: list[dict]) -> list[dict]:
    """Stage launch/exit pairs from the chain driver's events, in order, with gaps."""
    ivs = [r for r in intervals(chain_events) if r["phase"] == "stage"]
    out, prev_end = [], None
    for r in ivs:
        row = {"stage": r["key"], "launch_t": r["begin"], "exit_t": r["end"], "wall_s": r["seconds"],
               "exit_code": (r["end_event"] or {}).get("exit_code"),
               "gap_before_s": None if prev_end is None or r["begin"] is None else r["begin"] - prev_end}
        out.append(row)
        prev_end = r["end"]
    return out


def cumulative(t0: float, marks: dict) -> dict:
    """Elapsed seconds since t0 for named wall-clock marks (None stays None)."""
    out = {}
    for name, t in marks.items():
        if t is not None and t < t0:
            raise ValueError(f"mark {name} precedes t0")
        out[name] = None if t is None else t - t0
    return out


def map_checkpoints(rows: list[dict], t0: float, offset_s: float = 0.0, export_s: float = 0.0) -> list[dict]:
    """Checkpoint (epoch) -> elapsed wall time since the track's t0.

    ``offset_s`` converts the training process's clock to the t0 clock (WSL
    vs Windows; 0 on one host).  ``export_s`` is the one-off cost of turning
    a training checkpoint into the method's usable artifact, added to every
    checkpoint's usable time.
    """
    out = []
    for r in rows:
        if r.get("incomplete") or r.get("checkpoint_available_t") is None:
            continue
        avail = r["checkpoint_available_t"] + offset_s
        if avail < t0:
            raise ValueError(f"epoch {r['epoch']} checkpoint precedes t0")
        out.append({"epoch": r["epoch"], "checkpoint_available_elapsed_s": avail - t0,
                    "usable_elapsed_s": avail - t0 + export_s})
    return out


def first_recovery(trace: list[dict], key: str = "usable_elapsed_s") -> dict | None:
    """Earliest (by elapsed time) trace row whose historical rule verdict is True."""
    rows = sorted((r for r in trace if r.get(key) is not None), key=lambda r: r[key])
    for r in rows:
        if r.get("recovers") is True:
            return r
    return None


def frame_budget_ratios(seconds: float | None) -> dict:
    if seconds is None:
        return {"x_30fps_33.3ms": None, "x_60fps_16.7ms": None}
    return {"x_30fps_33.3ms": seconds / FRAME_S_30, "x_60fps_16.7ms": seconds / FRAME_S_60}


def latency_category(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    if seconds < CATEGORY_BOUNDS_S["A_frame_scale"]:
        return "A"
    if seconds < CATEGORY_BOUNDS_S["B_human_timescale"]:
        return "B"
    return "C"


# --- provenance and serialisation --------------------------------------------

def git_state(root) -> dict:
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = [l for l in subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                                       capture_output=True, text=True).stdout.splitlines() if l.strip()]
    return {"commit": head or None, "dirty": dirty}


def require_clean(state: dict, allow_dirty: bool = False) -> None:
    """Evidence runs refuse a dirty or unknown project tree; smoke runs may opt out."""
    if allow_dirty:
        return
    if not state.get("commit"):
        raise SystemExit("project commit unknown; refusing an evidence run")
    if state.get("dirty"):
        raise SystemExit("project tree is dirty; commit first (evidence runs need a clean tree):\n  "
                         + "\n  ".join(state["dirty"]))


def validate_record(rec: dict, smoke: bool = False) -> None:
    """Schema and consistency checks; only a smoke run's record may come from a dirty tree."""
    missing = [k for k in RECORD_FIELDS if k not in rec]
    if missing:
        raise ValueError(f"timing record lacks {missing}")
    if rec["schema"] != SCHEMA:
        raise ValueError(f"schema {rec['schema']!r} != {SCHEMA!r}")
    if rec["project_dirty"] and not smoke:
        raise ValueError("timing record comes from a dirty tree")
    fr = rec["first_recovery"]
    if fr is not None and rec["total_schedule_s"] is not None and fr["usable_elapsed_s"] > rec["total_schedule_s"] + 1e-6:
        raise ValueError("first recovery lies after the end of the schedule")


def dumps(payload) -> str:
    """JSON with NaN/inf refused (they would not round-trip as numbers)."""
    return json.dumps(payload, indent=2, allow_nan=False, default=_json_default)


def write_json(path, payload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    text = dumps(payload)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(text + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _json_default(value):
    if hasattr(value, "item") and callable(value.item):  # numpy / torch scalars
        return value.item()
    if hasattr(value, "tolist") and callable(value.tolist):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite float")
    raise TypeError(type(value))
