"""Capture the local environment without importing project-owned model code."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "8dna_deformation_replication" / "environment_probe.json"


def command(args: list[str]) -> dict[str, object]:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, check=False)
        return {
            "args": args,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except OSError as exc:
        return {"args": args, "error": repr(exc)}


def main() -> int:
    result: dict[str, object] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cuda_home": os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"),
        "nvidia_smi": command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader",
            ]
        ),
        "nvcc": command(["nvcc", "--version"]),
    }

    try:
        import torch

        result["torch"] = {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
            "device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "device_capability": (
                list(torch.cuda.get_device_capability(0))
                if torch.cuda.is_available()
                else None
            ),
        }
    except Exception as exc:
        result["torch"] = {"error": repr(exc)}

    try:
        import mitsuba as mi

        mi.set_variant("cuda_ad_rgb")
        result["mitsuba"] = {
            "version": mi.__version__,
            "variants": mi.variants(),
            "cuda_variant_selected": True,
        }
    except Exception as exc:
        result["mitsuba"] = {"error": repr(exc)}

    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
