"""Build and save the surface units of protocol sor_v1.json (h, seed 0).

    windows/run.ps1 ../surface_owned_radiance/sor_units.py --root <batch dir>
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
    import numpy as np

    proto = S.protocol()
    a = S.build_anchors(proto["representation"]["h"], seed=0)
    out = Path(args.root) / "units"
    out.mkdir(parents=True, exist_ok=True)
    S.save_units(out / "units.npz", a)
    rec = {"h": a["h"], "units": int(len(a["pos"])),
           "per_part": {p: int((a["part"] == i).sum()) for i, p in enumerate(S.PARTS)},
           "area_per_part": {p: float(a["area"][a["part"] == i].sum()) for i, p in enumerate(S.PARTS)},
           "table_entries": int(len(a["pos"]) * proto["representation"]["res"] ** 2 * 3)}
    S.write_json(out / "units.json", rec)
    print(rec, flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(rc)
