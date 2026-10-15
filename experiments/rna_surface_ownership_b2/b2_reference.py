"""Architecture Batch 2 (worklog 34): converged validation references for C2 (Windows Mitsuba).

    windows/run.ps1 ../rna_surface_ownership_b2/b2_reference.py --config C2 --views 1 2 3 ... --dest <dir>

Each listed validation view is replayed from rna_bridge.generate_h5's RNG (camera, light, seed;
proven exact in worklog 33) and rendered once at 8x the dataset spp with an independent seed
(950000 + view), by the unchanged rna_bridge.render_view. Existing outputs are skipped.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=("C1", "C2"))
    ap.add_argument("--views", type=int, nargs="+", required=True)
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
    rng = np.random.default_rng(gen["seeds"][split] + 7919 * list(proto["states"]).index(state))
    plan = []
    for v in range(gen["views"][split]):
        h = rng.random() * 0.9999
        r = np.sqrt(max(0.0, 1 - h * h))
        phi = 2 * np.pi * rng.random()
        origin = np.array(gen["camera_lookat"]) + gen["camera_radius"] * np.array([r * np.cos(phi), h, r * np.sin(phi)])
        plan.append((origin, B.hemisphere_dirs(rng, 1, 0.999), int(rng.integers(1 << 20))))
    for v in args.views:
        p = out / f"{args.config}_view{v}_ref.npz"
        if p.exists():
            print(args.config, v, "exists, skipped", flush=True)
            continue
        origin, to_light, _ = plan[v]
        sensor = mi.load_dict(B.sensor_dict(origin, gen["camera_lookat"], gen["fov"], res))
        t0 = time.time()
        view = B.render_view(scene, sensor, res, np.repeat(to_light, res * res, 0), 8 * gen["spp"], gen["chunk"], 950000 + v, gen["irradiance"])
        np.savez_compressed(p, color=view["color"].astype(np.float32), alpha=view["alpha"].astype(np.float32))
        print(args.config, v, "ref", 8 * gen["spp"], f"{time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush(); sys.stderr.flush()
    import os

    os._exit(rc)
