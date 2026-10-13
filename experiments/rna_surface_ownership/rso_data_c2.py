"""Configuration C2 datasets (protocol rso_v1.json `configurations.C2_rough`): the worklog-22 RNA
H5 producer (rna_bridge.generate_h5, unchanged) on the teaset T0 with every part's roughconductor
alpha set to 0.3 (Beckmann, Mitsuba default), 128 spp, written to this batch's own directory.

    windows/run.ps1 ../rna_surface_ownership/rso_data_c2.py --out <batch dir> --split train|val
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPL = HERE.parents[0] / "8dna_deformation_replication"
sys.path.insert(0, str(REPL))

ALPHA = 0.3
SPP = 128


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", required=True, choices=("train", "val"))
    args = ap.parse_args()
    import ednalib as L
    import teaset_parts as T
    import rna_bridge

    L.init_upstream()
    base = json.loads((REPL / "protocol" / "teaset_cross_backbone_locked.json").read_text(encoding="utf-8"))
    out = Path(args.out).resolve()
    ds_dir = out / "datasets_c2"
    ds_dir.mkdir(parents=True, exist_ok=True)
    derived = dict(base)
    derived["rna_dataset_dir"] = os.path.relpath(ds_dir, L.RESULTS)
    gen = dict(base["rna_dataset"])
    gen["spp"] = SPP
    gen["material_override"] = {"all_parts_alpha": ALPHA, "distribution": "beckmann (Mitsuba default, unchanged)"}
    derived["rna_dataset"] = gen
    proto_path = out / "protocol_c2_derived.json"
    proto_path.write_text(json.dumps(derived, indent=1))

    orig = T.scene_dict

    def scene_dict_rough(res, translations=None, lighting=None, integrator=None):
        d = orig(res, translations, lighting, integrator)
        for part in T.PARTS:
            d["group0"][part]["bsdf"]["bsdf"]["alpha"] = ALPHA
        return d

    T.scene_dict = scene_dict_rough  # rna_bridge resolves T.scene_dict at call time
    rna_bridge.generate_h5(argparse.Namespace(protocol=str(proto_path), state="T0", split=args.split))
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(rc)
