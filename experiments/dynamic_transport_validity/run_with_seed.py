"""Run an official RNA script after setting deterministic host-side RNG seeds."""

from __future__ import annotations

import argparse
import pathlib
import random
import runpy
import sys

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("script", type=pathlib.Path)
    known, remaining = parser.parse_known_args()
    random.seed(known.seed)
    np.random.seed(known.seed)
    torch.manual_seed(known.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(known.seed)
    sys.argv = [str(known.script), *remaining]
    runpy.run_path(str(known.script), run_name="__main__")


if __name__ == "__main__":
    main()

