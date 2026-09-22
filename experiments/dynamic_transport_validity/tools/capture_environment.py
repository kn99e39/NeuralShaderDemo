"""Capture exact software/hardware versions used by Batch 1."""

from __future__ import annotations

import json
import importlib.metadata
import os
import pathlib
import platform
import subprocess
import sys

import bpy
import torch


def command(*args: str) -> str:
    completed = subprocess.run(args, text=True, capture_output=True, check=False)
    return (completed.stdout + completed.stderr).strip()


def main() -> None:
    output = pathlib.Path(sys.argv[1])
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "pytorch": torch.__version__,
        "pytorch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "blender_bpy": bpy.app.version_string,
        "blender_build_hash": bpy.app.build_hash.decode("ascii", errors="replace"),
        "blender_cycles": bpy.app.build_options.cycles,
        "nvidia_smi": command("nvidia-smi"),
        "ubuntu": command("cat", "/etc/os-release"),
        "repository_commit": command("git", "rev-parse", "HEAD"),
        "uv": command("uv", "--version"),
        "installed_distributions": sorted(
            f"{distribution.metadata['Name']}=={distribution.version}"
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        ),
        "cwd": os.getcwd(),
    }
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
