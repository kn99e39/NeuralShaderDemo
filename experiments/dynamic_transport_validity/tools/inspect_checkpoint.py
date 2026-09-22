"""Write stable, human-readable metadata for the published high-quality checkpoint."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch
from omegaconf import OmegaConf


WORKSPACE = pathlib.Path(__file__).resolve().parents[3]
RNA_ROOT = WORKSPACE / "external" / "relightable-neural-assets"
EXPERIMENT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RNA_ROOT))
sys.path.insert(0, str(EXPERIMENT_DIR))

from shared.rna_compat import install_historical_module_aliases  # noqa: E402


def json_safe(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if OmegaConf.is_config(value):
        return json_safe(OmegaConf.to_container(value, resolve=True))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            "__class__": f"{type(value).__module__}.{type(value).__qualname__}",
            "repr": repr(value),
        }
    return repr(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()

    install_historical_module_aliases()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", {})
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_keys": sorted(checkpoint.keys()),
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "pytorch_lightning_version": checkpoint.get("pytorch-lightning_version"),
        "state_tensor_count": len(state),
        "state_parameter_count": int(
            sum(value.numel() for value in state.values() if hasattr(value, "numel"))
        ),
        "hyper_parameters": json_safe(checkpoint.get("hyper_parameters", {})),
        "callbacks": json_safe(checkpoint.get("callbacks", {})),
        "loop_state": json_safe(checkpoint.get("loops", {})),
        "state_shapes": {
            key: list(value.shape) for key, value in state.items() if hasattr(value, "shape")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
