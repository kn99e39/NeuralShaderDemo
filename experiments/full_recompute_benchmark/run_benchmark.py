"""Host driver for the BMW full-recomputation benchmark (stdlib only).

python run_benchmark.py <stage> [--run-id ID]

stages:
  env        record environment + commit only
  spp        original scene, every protocol SPP, persistent on and off
  bounce     original scene, 64 spp, bounce presets low/medium/original
  composite  original scene, 64 spp, compositor on (cost of the post stack)
  evidence   G0/G1 evidence renders of the original scene
  xl-build   build the XL variants (stop rule per mode)
  xl         XL scale series (spp 1 and 64, persistent on and off)
  all        every stage above in order

Each Blender process gets its own Cycles debug log and its own JSON record
under results/full_recompute_benchmark/runs/<run-id>/.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import frb_records as R  # noqa: E402

PROTOCOL = os.path.join(HERE, "protocol", "bmw_full_recompute_v1.json")
OUT_ROOT = os.path.join(ROOT, "results", "full_recompute_benchmark")
BL = os.path.join(HERE, "blender")
RAM_STOP_MB = 48 * 1024


def git_state() -> dict:
    def g(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": g("rev-parse", "HEAD"),
            "dirty_paths": [line for line in g("status", "--porcelain").splitlines()
                            if "full_recompute_benchmark" in line]}


def nvidia_query(fields: str) -> list:
    out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return [x.strip() for x in out.split(",")]


def environment() -> dict:
    def ps(cmd):
        return subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True,
                              text=True).stdout.strip()
    name, driver, mem = nvidia_query("name,driver_version,memory.total")
    return {
        "os": platform.platform(),
        "os_caption": ps("(Get-CimInstance Win32_OperatingSystem).Caption + ' ' + (Get-CimInstance Win32_OperatingSystem).Version"),
        "cpu": ps("(Get-CimInstance Win32_Processor).Name"),
        "ram_gb": float(ps("[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,2)")),
        "gpu": name, "gpu_driver": driver, "gpu_memory_total_mib": int(mem),
        "host_python": sys.version.split()[0],
    }


class GpuPoller(threading.Thread):
    """Whole-GPU memory/temperature/clock samples every 0.2 s."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self._halt = threading.Event()

    def run(self):
        while not self._halt.is_set():
            try:
                used, temp, clk, pwr = nvidia_query("memory.used,temperature.gpu,clocks.sm,power.draw")
                self.samples.append((time.time(), int(used), int(temp), int(clk), float(pwr)))
            except Exception:
                pass
            self._halt.wait(0.2)

    def stop(self):
        self._halt.set()
        self.join()


def run_blender(protocol, script, args, log_path, timeout=3600) -> dict:
    exe = protocol["blender"]["executable"]
    base_used = int(nvidia_query("memory.used")[0])
    cmd = [exe, "--factory-startup", "-b", "--log", "cycles", "--log-level", "debug",
           "--log-file", log_path, "--python", os.path.join(BL, script), "--", *args]
    poll = GpuPoller()
    poll.start()
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
        rc, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        rc, out, err = "timeout", (e.stdout or ""), (e.stderr or "")
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        if isinstance(err, bytes):
            err = err.decode("utf-8", "replace")
    wall = time.perf_counter() - t0
    poll.stop()
    s = poll.samples
    return {
        "returncode": rc, "process_wall_s": wall, "stdout_tail": out[-4000:], "stderr_tail": err[-4000:],
        "gpu_mem_baseline_mib": base_used,
        "gpu_mem_peak_mib": max((x[1] for x in s), default=None),
        "gpu_mem_peak_delta_mib": (max(x[1] for x in s) - base_used) if s else None,
        "gpu_temp_max_c": max((x[2] for x in s), default=None),
        "gpu_sm_clock_median_mhz": sorted(x[3] for x in s)[len(s) // 2] if s else None,
        "gpu_power_max_w": max((x[4] for x in s), default=None),
        "gpu_samples": len(s),
    }


def run_timing(protocol, run_dir, cid, cond) -> dict:
    os.makedirs(run_dir, exist_ok=True)
    out_json = os.path.join(run_dir, cid + ".json")
    if os.path.exists(out_json):
        raise FileExistsError(f"{out_json} exists; use a new run id")
    log = os.path.join(run_dir, cid + ".log")
    cond = dict(cond, log_path=log, condition_id=cid)
    cpath = os.path.join(run_dir, cid + ".condition.json")
    R.dump_json(cond, cpath)
    res = run_blender(protocol, "bench_timing.py", [PROTOCOL, cpath, out_json], log)
    ok = res["returncode"] == 0 and os.path.exists(out_json)
    if ok:
        rec = R.load_json(out_json)
        rec["host"] = res
        R.dump_json(rec, out_json)
        st = rec["summary"].get("steady", {}).get("frame_total_s", {})
        print(f"[{cid}] ok  steady median {st.get('median', float('nan')):.4f}s  "
              f"gpu+{res['gpu_mem_peak_delta_mib']} MiB  wall {res['process_wall_s']:.1f}s", flush=True)
    else:
        R.dump_json({"condition": cond, "host": res, "failed": True}, os.path.join(run_dir, cid + ".failed.json"))
        print(f"[{cid}] FAILED rc={res['returncode']}\n{res['stdout_tail'][-1500:]}\n{res['stderr_tail'][-800:]}",
              flush=True)
    return {"ok": ok, "json": out_json}


def base_cond(protocol, scene_path, variant, spp, persistent, bounce="original", camera=None, **extra):
    rep = protocol["repetitions"]
    return dict(variant=variant, scene_path=scene_path, spp=spp, bounce_preset=bounce,
                persistent=persistent, camera=camera, warmup=rep["warmup"],
                steady=rep["steady_persistent_on"] if persistent else rep["steady_persistent_off"],
                static=rep["static"], **extra)


def stage_spp(protocol, run_dir):
    src = protocol["scene_source"]["path"]
    for persistent in (True, False):
        for spp in protocol["spp_values"]:
            cid = f"orig_spp{spp:03d}_{'pon' if persistent else 'poff'}"
            run_timing(protocol, run_dir, cid, base_cond(protocol, src, "original", spp, persistent))


def stage_bounce(protocol, run_dir):
    src = protocol["scene_source"]["path"]
    b = protocol["bounce_sensitivity"]
    for preset in b["presets"]:
        cid = f"orig_bounce_{preset}_spp{b['spp']:03d}_pon"
        run_timing(protocol, run_dir, cid, base_cond(protocol, src, "original", b["spp"], True, bounce=preset))


def stage_composite(protocol, run_dir):
    src = protocol["scene_source"]["path"]
    run_timing(protocol, run_dir, "orig_composite_on_spp064_pon",
               base_cond(protocol, src, "original", 64, True, compositing_override=True))


def stage_evidence(protocol, run_dir):
    out = os.path.join(run_dir, "evidence_original")
    res = run_blender(protocol, "render_evidence.py", [PROTOCOL, protocol["scene_source"]["path"], out],
                      os.path.join(run_dir, "evidence_original.log"))
    print("[evidence]", res["returncode"], res["stdout_tail"][-600:], flush=True)


def xl_paths(g, mode):
    d = os.path.join(OUT_ROOT, "scenes")
    return (os.path.join(d, f"bmw_garage_xl_{g:02d}x{g:02d}_{mode}.blend"),
            os.path.join(d, f"bmw_garage_xl_{g:02d}x{g:02d}_{mode}.manifest.json"))


def stage_xl_build(protocol, run_dir):
    os.makedirs(os.path.join(OUT_ROOT, "scenes"), exist_ok=True)
    status = {}
    for mode, levels in protocol["xl_policy"]["levels_bays_per_side"].items():
        for g in levels:
            blend, man = xl_paths(g, mode)
            if os.path.exists(blend) and os.path.exists(man):
                print(f"[xl-build] {mode} {g}x{g} exists", flush=True)
                status[f"{mode}_{g}"] = "exists"
                continue
            log = os.path.join(run_dir, f"xl_build_{g:02d}_{mode}.log")
            res = run_blender(protocol, "build_xl.py", [PROTOCOL, str(g), mode, blend, man], log,
                              timeout=7200)
            ok = res["returncode"] == 0 and os.path.exists(man)
            peak = None
            if ok:
                m = R.load_json(man)
                peak = m.get("build_memory", {}).get("peak_working_set_mb")
                m["host"] = res
                R.dump_json(m, man)
            print(f"[xl-build] {mode} {g}x{g} ok={ok} peak_ws={peak} wall={res['process_wall_s']:.0f}s", flush=True)
            if not ok or (peak and peak > RAM_STOP_MB):
                status[f"{mode}_{g}"] = "capacity_boundary"
                R.dump_json({"mode": mode, "bays_per_side": g, "host": res, "peak_working_set_mb": peak},
                            os.path.join(run_dir, f"xl_build_{g:02d}_{mode}.boundary.json"))
                break
            status[f"{mode}_{g}"] = "built"
    R.dump_json(status, os.path.join(run_dir, "xl_build_status.json"))


def stage_xl(protocol, run_dir):
    xl = protocol["xl_policy"]
    for mode, levels in xl["levels_bays_per_side"].items():
        stop = False
        for g in levels:
            blend, man = xl_paths(g, mode)
            if not os.path.exists(blend):
                print(f"[xl] {mode} {g}x{g} not built; stopping {mode}", flush=True)
                break
            for persistent in xl["scale_series_persistent"]:
                for spp in xl["scale_series_spp"]:
                    cid = f"xl{g:02d}_{mode}_spp{spp:03d}_{'pon' if persistent else 'poff'}"
                    r = run_timing(protocol, run_dir, cid,
                                   base_cond(protocol, blend, f"xl_{g}x{g}_{mode}", spp, persistent,
                                             camera="XLCamera"))
                    if not r["ok"]:
                        stop = True
                        break
                if stop:
                    break
            if stop:
                print(f"[xl] {mode} {g}x{g} failed; recorded as capacity boundary", flush=True)
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--run-id", default=time.strftime("run_%Y%m%d_%H%M%S"))
    a = ap.parse_args()
    protocol = R.load_json(PROTOCOL)
    run_dir = os.path.join(OUT_ROOT, "runs", a.run_id)
    os.makedirs(run_dir, exist_ok=True)
    man_path = os.path.join(run_dir, f"run_manifest_{a.stage}.json")
    R.dump_json({"stage": a.stage, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "git": git_state(),
                 "environment": environment(), "protocol_id": protocol["protocol_id"]}, man_path)
    stages = {"spp": stage_spp, "bounce": stage_bounce, "composite": stage_composite,
              "evidence": stage_evidence, "xl-build": stage_xl_build, "xl": stage_xl}
    order = list(stages) if a.stage == "all" else ([] if a.stage == "env" else [a.stage])
    for s in order:
        stages[s](protocol, run_dir)
    m = R.load_json(man_path)
    m["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    R.dump_json(m, man_path)


if __name__ == "__main__":
    main()
