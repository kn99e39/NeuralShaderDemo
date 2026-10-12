"""Held-out angular-fidelity queries (protocol evaluation): 500 000 queries per state, 32 spp,
seed 51051 (T3) / 51050 (T0); never used for fitting, oracle or selection.

    windows/run.ps1 ../surface_owned_radiance/sor_heldout.py --root <batch dir>
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import sor_common as S


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    S.init()
    import torch
    from sor_targets import directions, scene_for, trace_t

    out = Path(args.root) / "heldout"
    out.mkdir(parents=True, exist_ok=True)
    for state, sd in (("T0", 51050), ("T3", 51051)):
        scene, ids, tr = scene_for(state)
        g = torch.Generator(device="cuda").manual_seed(sd)
        x, n, p = S.SurfaceSampler().positions(500_000, g)
        q = directions(x, n, p, g, tr)
        _, y, ok = trace_t(scene, ids, q, seed=sd, spp=32)
        s = torch.nonzero(ok).squeeze(1)
        torch.save({"x_local": q["x_local"][s], "part": q["part"][s], "n": q["n"][s], "w": q["w"][s], "y": y[s],
                    "spp": 32, "seed": sd}, out / f"heldout_{state}.pt")
        print(state, "kept", len(s), flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(rc)
