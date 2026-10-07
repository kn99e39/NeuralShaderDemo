"""DIAGNOSTIC ONLY: path-traced incident radiance along the same K=32 probe directions (Windows, Mitsuba).

    windows/run.ps1 ../radiometric_relation_state_prototype/rrs_oracle_radiance.py --out <run dir> [--allow-dirty]

Predeclared in rrs_v1.json ("exact_direction_oracle"); run only if the real runtime
candidate fails the success test.  For every stable query of every state and each of
its 32 worklog-28 probe directions: the radiance arriving at the query along that
direction in the current LIT scene, estimated with Mitsuba's C++ path integrator
(max_depth -1, rr_depth 5, wavefront mode, as the references), 256 spp, two
independent seeds A and B.  Same origins (x + eps*n) and directions as the probes.
Never part of a method.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[0]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXP / "relational_residual_prototype"))
sys.path.insert(0, str(EXP / "8dna_deformation_replication"))
sys.path.insert(0, str(EXP / "neural_recompute_cost"))

import numpy as np  # noqa: E402

import rrp_common as C28  # noqa: E402
import rrs_common as C  # noqa: E402

SEEDS = {"A": 7_000_000, "B": 8_000_000}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--spp", type=int, default=None, help="smoke runs only")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    proto, p28 = C.protocol(), C28.protocol()
    spp = args.spp or proto["exact_direction_oracle"]["spp"]
    import ednalib as L
    import nrc_records as R

    git = R.git_state(L.ROOT)
    R.require_clean(git, args.allow_dirty)
    mi, dr = L.init_upstream()
    import teaset_parts as T

    cb = json.loads((L.EXPERIMENT / p28["base_protocol"]).read_text(encoding="utf-8"))
    wl28 = L.ROOT / proto["worklog28_reference"]["run"]
    dirs = C28.probe_dirs(p28["probes"]["K"])
    eps = p28["probes"]["origin_offset"]
    integ = mi.load_dict(dict(cb["lighting"]["reference_integrator"]))
    rec = {"git": git, "spp": spp, "seeds": SEEDS, "integrator": cb["lighting"]["reference_integrator"], "states": {}}
    chunk = 1 << 20
    for name in ("T0", "T1", "T1b", "T2", "T3"):
        tr = cb["states"][name]
        z = np.load(wl28 / "probes" / f"{name}.npz")
        f = np.load(L.RESULTS / cb["rna_features_dir"] / f"{name}.npz")
        x = f["position"][z["lanes"]].astype(np.float64)
        n = z["query_normal"].astype(np.float64)
        s_, t_ = C28.onb(n)
        d = C28.to_world(dirs, s_, n, t_).reshape(-1, 3).astype(np.float32)
        o = np.repeat(x + eps * n, len(dirs), 0).astype(np.float32)
        scene = mi.load_dict(T.scene_dict(64, tr, cb["lighting"]))
        t0 = time.perf_counter()
        for tag, base_seed in SEEDS.items():
            acc = np.zeros((len(o), 3), np.float64)
            for a in range(0, len(o), chunk):
                b = min(a + chunk, len(o))
                ray = mi.Ray3f(mi.Point3f(*o[a:b].T), mi.Vector3f(*d[a:b].T))
                part = None
                for sidx in range(spp):
                    sampler = mi.load_dict({"type": "independent"})
                    sampler.seed(base_seed + sidx * 4099 + a, b - a)
                    spec, _, _ = integ.sample(scene, sampler, ray, None, True)
                    part = spec if part is None else part + spec
                    dr.eval(part)
                acc[a:b] = np.stack([np.array(part.x), np.array(part.y), np.array(part.z)], -1) / spp
            arr = acc.reshape(len(x), len(dirs), 3).astype(np.float32)
            path = out / "oracle" / f"{name}_{tag}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, arr)
            if name == "T0" and tag == "B":  # T0_B (independent light samples of the query) uses T0's seed-B oracle
                np.save(out / "oracle" / "T0_B_A.npy", arr)
        ea, eb = C.encode(np.load(out / "oracle" / f"{name}_A.npy")), C.encode(np.load(out / "oracle" / f"{name}_B.npy"))
        rec["states"][name] = {"rays": int(len(o)), "seconds": time.perf_counter() - t0,
                               "probe_repeat_noise_encoded_mean_abs": float(np.abs(ea - eb).mean()),
                               "query_repeat_noise_encoded_mean_abs": float(np.abs(ea.mean(1) - eb.mean(1)).mean())}
        print(name, rec["states"][name], flush=True)
    # noise vs the T0 -> state change of the same quantity (identical queries for T1b/T2/T3)
    e0 = C.encode(np.load(out / "oracle" / "T0_A.npy"))
    p0 = np.load(wl28 / "probes" / "T0.npz")
    for name in ("T1b", "T2", "T3"):
        zs = np.load(wl28 / "probes" / f"{name}.npz")
        pix0, pixs = p0["pixels"][p0["stable"]], zs["pixels"][zs["stable"]]
        common, i0, i1 = np.intersect1d(pix0, pixs, return_indices=True)
        q0 = (i0[:, None] * 16 + np.arange(16)).reshape(-1)
        q1 = (i1[:, None] * 16 + np.arange(16)).reshape(-1)
        es = C.encode(np.load(out / "oracle" / f"{name}_A.npy"))
        rec["states"][name]["change_vs_T0_encoded_mean_abs"] = float(np.abs(es[q1] - e0[q0]).mean())
        rec["states"][name]["change_over_probe_noise"] = rec["states"][name]["change_vs_T0_encoded_mean_abs"] / rec["states"][name]["probe_repeat_noise_encoded_mean_abs"]
    L.write_json(out / "oracle" / "oracle.json", rec)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
