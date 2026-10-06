"""WSL helpers shared by the chain and the evaluation scripts (Windows side, stdlib only).

Long WSL jobs run through wsl_run_detached.sh and are polled through their
status file on the Windows filesystem, so they never depend on a wsl.exe
client staying alive (AGENTS.md, worklogs 16/20).
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RNA_ROOT = ROOT / "external" / "relightable-neural-assets"
WSL_ROOT = "/mnt/c/Projects/NeuralShaderDemo"
DISTRO = "Ubuntu-22.04"
LAUNCHER = f"{WSL_ROOT}/experiments/dynamic_transport_failure/scripts/wsl_run_detached.sh"


def wsl_path(p) -> str:
    return WSL_ROOT + "/" + str(Path(p).resolve().relative_to(ROOT)).replace("\\", "/")


def wsl(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["wsl.exe", "-d", DISTRO, "-u", "root", "--exec", *args], capture_output=True, text=True, **kw)


def clock_offset() -> dict:
    """WSL time.time() minus Windows time.time() at the midpoint of one call."""
    a = time.time()
    p = wsl(["python3", "-c", "import time;print(repr(time.time()))"])
    b = time.time()
    try:
        w = float(p.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"ok": False, "stdout": p.stdout[-200:], "stderr": p.stderr[-200:]}
    return {"ok": True, "offset_s": w - 0.5 * (a + b), "round_trip_s": b - a}


def launch_detached(job: str, log_dir: Path, job_argv: list[str], workdir: Path = RNA_ROOT) -> Path:
    """Start a detached WSL job; returns its status file (absent while running)."""
    status = Path(log_dir) / f"{job}.status"
    if status.exists():
        status.unlink()
    p = wsl(["bash", LAUNCHER, job, wsl_path(log_dir), wsl_path(workdir), "--", *job_argv])
    if p.returncode != 0:
        raise RuntimeError(f"detached launch of {job} failed: {p.stdout} {p.stderr}")
    return status


def wait_status(status: Path, poll_s: float = 5.0) -> int:
    while not status.exists():
        time.sleep(poll_s)
    text = status.read_text(encoding="utf-8").strip()
    return int(re.search(r"exit=(-?\d+)", text).group(1))


def run_detached(job: str, log_dir: Path, job_argv: list[str], workdir: Path = RNA_ROOT) -> int:
    return wait_status(launch_detached(job, log_dir, job_argv, workdir))
