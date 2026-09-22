"""Compatibility aliases for the official checkpoint's historical module names.

The published `lego.ckpt` pickles its configuration as `neumat.config`, while the
published source tree exposes the same implementation as `rna.config`.  Registering
module aliases before `torch.load` repairs deserialization without changing a class,
tensor, model parameter, or training/rendering semantic.
"""

from __future__ import annotations

import importlib
import sys


def install_historical_module_aliases() -> None:
    rna_package = importlib.import_module("rna")
    sys.modules.setdefault("neumat", rna_package)
    for module_name in (
        "config",
        "datasets",
        "encodings",
        "interfaces",
        "lights",
        "losses",
        "models",
        "neural_textures",
        "renderers",
        "timer",
    ):
        current = importlib.import_module(f"rna.{module_name}")
        sys.modules.setdefault(f"neumat.{module_name}", current)

