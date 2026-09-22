"""Run the unmodified official renderer with checkpoint-name compatibility."""

from __future__ import annotations

import pathlib
import runpy
import sys


WORKSPACE = pathlib.Path(__file__).resolve().parents[3]
RNA_ROOT = WORKSPACE / "external" / "relightable-neural-assets"
EXPERIMENT_DIR = pathlib.Path(__file__).resolve().parents[1]

sys.path.insert(0, str(RNA_ROOT))
sys.path.insert(0, str(EXPERIMENT_DIR))

from shared.rna_compat import install_historical_module_aliases  # noqa: E402


install_historical_module_aliases()
runpy.run_path(str(RNA_ROOT / "scripts" / "render.py"), run_name="__main__")
