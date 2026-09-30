"""Is an RNA H5 dataset or feature buffer current for a protocol? Exit 0 if yes, 1 with a reason if not.

Used by the chain before skipping a dataset or features step. A file rendered
under a superseded lighting revision, or with different generation or
light-sampling settings, must be regenerated rather than silently reused;
files written before this check existed carry no provenance and are treated
as stale.
"""

from __future__ import annotations

import argparse
import json

import ednalib as L


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--h5", required=True)
    args = ap.parse_args()
    import h5py

    proto = json.loads(open(L.EXPERIMENT / args.protocol, encoding="utf-8").read())
    if args.h5.endswith(".npz"):
        return check_features(proto, args.h5)
    try:
        with h5py.File(args.h5, "r") as f:
            attrs = {k: f.attrs[k] for k in f.attrs}
            views = f["color"].shape[0]
    except Exception as exc:  # unreadable or truncated
        print(f"unreadable: {type(exc).__name__}")
        return 1
    if "lighting" not in attrs or "generation" not in attrs:
        print("no lighting/generation provenance (written before this check)")
        return 1
    if json.loads(attrs["lighting"]) != proto["lighting"]:
        print("lighting differs from the protocol")
        return 1
    gen = json.loads(attrs["generation"])
    if gen != proto["rna_dataset"]:
        differing = sorted(k for k in set(gen) | set(proto["rna_dataset"]) if gen.get(k) != proto["rna_dataset"].get(k))
        print(f"generation settings differ: {differing}")
        return 1
    split = "val" if args.h5.endswith("_val.h5") else "train"
    if views != proto["rna_dataset"]["views"][split]:
        print(f"incomplete: {views} views, expected {proto['rna_dataset']['views'][split]}")
        return 1
    return 0


def check_features(proto: dict, path: str) -> int:
    import numpy as np

    try:
        z = np.load(path)
        keys = set(z.files)
    except Exception as exc:
        print(f"unreadable: {type(exc).__name__}")
        return 1
    if not {"lighting", "light_sampling", "light_dir", "light_weight", "light_vis"} <= keys:
        print("no area-light samples or provenance (delta-light contract)")
        return 1
    if json.loads(str(z["lighting"])) != proto["lighting"]:
        print("lighting differs from the protocol")
        return 1
    if json.loads(str(z["light_sampling"])) != proto["rna_inference"]["light_sampling"]:
        print("light sampling differs from the protocol")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
