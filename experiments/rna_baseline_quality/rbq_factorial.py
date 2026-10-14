"""Worklog-33 control D1: where does RNA's train/validation gap come from? (Windows Mitsuba)

    windows/run.ps1 ../rna_baseline_quality/rbq_factorial.py --config C1|C2 --dest <dir>

The worklog-32 runs show validation loss 7-9x the training loss while 16x capacity lowers only the
training loss. Training views use an independent light per PIXEL, validation views one light per
IMAGE, and validation cameras are unseen. This renders, with the unchanged rna_bridge.render_view
at the dataset spp, a 2x2 design fixed before any result was seen:

  camera  x  light distribution     cells (4 cameras each)
  train   x  per-pixel  (new draws) TP: in-distribution except the light samples are new
  train   x  per-image              TI: seen camera, validation-style light
  val     x  per-pixel              VP: unseen camera, training-style light
  val     x  per-image  (stored)    VI: the stored validation views (not re-rendered)

Train cameras: train views 0, 50, 100, 150 replayed from generate_h5's RNG; val cameras: val views
0, 10, 20, 30. New lights come from a separate RNG (seed 424242 + view), uniform on the upper
hemisphere exactly as hemisphere_dirs; render seeds are independent of the dataset's.
Each output npz is an RNA-H5-compatible view (all channels the loader reads), so the A
checkpoint can be evaluated on it with the same code as the validation set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPL = HERE.parents[0] / "8dna_deformation_replication"
sys.path.insert(0, str(REPL))

TRAIN_VIEWS = (0, 50, 100, 150)
VAL_VIEWS = (0, 10, 20, 30)


def replay(gen, split, state_index, want):
    import rna_bridge as B

    rng = np.random.default_rng(gen["seeds"][split] + 7919 * state_index)
    res = gen["resolution"]
    cams = {}
    for v in range(max(want) + 1):
        h = rng.random() * 0.9999
        r = np.sqrt(max(0.0, 1 - h * h))
        phi = 2 * np.pi * rng.random()
        origin = np.array(gen["camera_lookat"]) + gen["camera_radius"] * np.array([r * np.cos(phi), h, r * np.sin(phi)])
        B.hemisphere_dirs(rng, res * res if split == "train" else 1, 0.999)  # consume the stored lights
        rng.integers(1 << 20)
        if v in want:
            cams[v] = origin
    return cams


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=("C1", "C2"))
    ap.add_argument("--dest", required=True)  # not --out: PowerShell would parse it as -OutVariable
    args = ap.parse_args()
    import ednalib as L
    import rna_bridge as B
    import teaset_parts as T

    mi, dr = L.init_upstream()
    proto = json.loads((REPL / "protocol" / "teaset_cross_backbone_locked.json").read_text(encoding="utf-8"))
    gen = dict(proto["rna_dataset"])
    if args.config == "C2":
        gen["spp"] = 128
        orig = T.scene_dict

        def scene_dict_rough(res, translations=None, lighting=None, integrator=None):
            d = orig(res, translations, lighting, integrator)
            for part in T.PARTS:
                d["group0"][part]["bsdf"]["bsdf"]["alpha"] = 0.3
            return d

        T.scene_dict = scene_dict_rough
    state = "T0"
    si = list(proto["states"]).index(state)
    scene = mi.load_dict(T.scene_dict(64, proto["states"][state], None))
    res = gen["resolution"]
    out = Path(args.dest); out.mkdir(parents=True, exist_ok=True)
    cams = {"train": replay(gen, "train", si, TRAIN_VIEWS), "val": replay(gen, "val", si, VAL_VIEWS)}
    cells = [("TP", "train", "pixel"), ("TI", "train", "image"), ("VP", "val", "pixel")]
    rec = {"config": args.config, "spp": gen["spp"], "cells": {}}
    for cell, split, mode in cells:
        for v, origin in cams[split].items():
            lrng = np.random.default_rng(424242 + v + (0 if split == "train" else 1000) + (0 if mode == "pixel" else 2000))
            to_light = B.hemisphere_dirs(lrng, res * res, 0.999) if mode == "pixel" else np.repeat(B.hemisphere_dirs(lrng, 1, 0.999), res * res, 0)
            sensor = mi.load_dict(B.sensor_dict(origin, gen["camera_lookat"], gen["fov"], res))
            view = B.render_view(scene, sensor, res, to_light, gen["spp"], gen["chunk"], 700000 + 10 * v + len(rec["cells"]), gen["irradiance"])
            np.savez_compressed(out / f"{args.config}_{cell}_view{v}.npz", **{k: np.nan_to_num(view[k]).astype(np.float16) for k in view})  # float16 = the H5 contract
            rec["cells"].setdefault(cell, []).append(v)
            print(args.config, cell, v, flush=True)
    (out / f"{args.config}_factorial.json").write_text(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush(); sys.stderr.flush()
    import os

    os._exit(rc)
