"""Worklog-33 control C: reference noise and convergence of the RNA validation targets.

    windows/run.ps1 ../rna_baseline_quality/rbq_reference.py --config C1|C2 --dest <dir>
    (run from experiments/8dna_deformation_replication via its Windows Mitsuba runtime)

Protocol, fixed before any result of this script was seen:
  * views 0, 10, 20, 30 of the validation split (the worklog-32 export views);
  * each view's camera, light direction and render seed are replayed from generate_h5's RNG
    sequence (rna_bridge.generate_h5, unchanged) - view 0 is re-rendered with its ORIGINAL
    seed and spp to prove the replay reproduces the stored H5 (float16 rounding only);
  * per view: one render at the dataset spp with an independent seed (noise replicate) and one
    at 8x the dataset spp with another independent seed (convergence reference);
  * C2 = the worklog-32 rough configuration (every part alpha 0.3), patched exactly as
    rso_data_c2.py does.
Writes <out>/<cfg>_view<v>_{orig,rep,ref}.npz (color float32, alpha) and <out>/<cfg>_reference.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPL = HERE.parents[0] / "8dna_deformation_replication"
sys.path.insert(0, str(REPL))

VIEWS = (0, 10, 20, 30)
SEED_REP, SEED_REF = 900001, 900002  # independent of generate_h5's integer seeds (< 2^20)


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
    state, split = "T0", "val"
    scene = mi.load_dict(T.scene_dict(64, proto["states"][state], None))
    res = gen["resolution"]
    out = Path(args.dest); out.mkdir(parents=True, exist_ok=True)

    # replay generate_h5's RNG sequence for the validation split
    rng = np.random.default_rng(gen["seeds"][split] + 7919 * list(proto["states"]).index(state))
    plan = []
    for v in range(gen["views"][split]):
        h = rng.random() * 0.9999
        r = np.sqrt(max(0.0, 1 - h * h))
        phi = 2 * np.pi * rng.random()
        origin = np.array(gen["camera_lookat"]) + gen["camera_radius"] * np.array([r * np.cos(phi), h, r * np.sin(phi)])
        to_light = B.hemisphere_dirs(rng, 1, 0.999)
        seed = int(rng.integers(1 << 20))
        plan.append((origin, to_light, seed))

    rec = {"config": args.config, "dataset_spp": gen["spp"], "ref_spp": 8 * gen["spp"], "views": {}, "seconds": {}}
    for v in VIEWS:
        origin, to_light, seed = plan[v]
        sensor = mi.load_dict(B.sensor_dict(origin, gen["camera_lookat"], gen["fov"], res))
        tl = np.repeat(to_light, res * res, 0)
        jobs = [("rep", gen["spp"], SEED_REP + v), ("ref", 8 * gen["spp"], SEED_REF + v)]
        if v == 0:
            jobs.insert(0, ("orig", gen["spp"], seed))
        rec["views"][v] = {"origin": origin.tolist(), "to_light": to_light[0].tolist(), "dataset_seed": seed}
        for name, spp, s in jobs:
            t0 = time.time()
            view = B.render_view(scene, sensor, res, tl, spp, gen["chunk"], s, gen["irradiance"])
            rec["seconds"][f"{v}_{name}"] = time.time() - t0
            np.savez_compressed(out / f"{args.config}_view{v}_{name}.npz", color=view["color"].astype(np.float32),
                                alpha=view["alpha"].astype(np.float32))
            print(args.config, v, name, spp, f"{rec['seconds'][f'{v}_{name}']:.0f}s", flush=True)
    (out / f"{args.config}_reference.json").write_text(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush(); sys.stderr.flush()
    import os

    os._exit(rc)
