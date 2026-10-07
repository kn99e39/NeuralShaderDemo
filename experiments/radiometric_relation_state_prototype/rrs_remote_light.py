"""Current direct-light inputs at every remote probe hit (Windows, Mitsuba) -- worklog 29, stage 1.

    windows/run.ps1 ../radiometric_relation_state_prototype/rrs_remote_light.py --out <run dir> [--allow-dirty]

For each state:
  1. Reproduce the worklog-28 probe state: re-cast the K=32 probes of the stored
     stable queries with the unchanged rrp_probes.cast_probes and require every
     array to equal the stored worklog-28 probe file bit for bit (else STOP).
  2. Re-cast the same rays in the current LIT scene (same geometry + protocol
     emitter) and require each geometric hit to land on the same point.
  3. At every hit, the existing rna_bridge.area_light_samples (16 samples, strata
     at their centres -> deterministic) with the remote normal faced toward the
     query: light directions, weights, current segment visibility.
Writes remote_light/<state>.npz (hit probes only, row-major over (query, probe)).
No reference, decomposition or state label is read.
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

COMPARE = ("hit", "dist", "remote_normal_local", "remote_facing", "remote_part", "remote_canonical", "remote_light_vis", "remote_light_cos")


class CentreRNG:
    """Stand-in for numpy's Generator in rna_bridge.area_light_samples: every jitter
    draw is 0.5, i.e. the 4x4 stratum centres -> the same samples for the same geometry."""

    def random(self, n):
        return np.full(n, 0.5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    proto, p28 = C.protocol(), C28.protocol()
    import ednalib as L
    import nrc_records as R

    git = R.git_state(L.ROOT)
    R.require_clean(git, args.allow_dirty)
    mi, dr = L.init_upstream()
    import rna_bridge as B
    import teaset_parts as T
    import torch

    import rrp_probes

    cb = json.loads((L.EXPERIMENT / p28["base_protocol"]).read_text(encoding="utf-8"))
    wl28 = L.ROOT / proto["worklog28_reference"]["run"]
    dirs = C28.probe_dirs(p28["probes"]["K"])
    to_light = np.asarray(cb["lighting"]["to_light"], float)
    to_light /= np.linalg.norm(to_light)
    eps = p28["probes"]["origin_offset"]
    rec = {"git": git, "environment": L.environment_record(), "states": {}}
    for name in proto["evaluate_states"]:
        st = "T0" if name == "T0_B" else name
        tr = cb["states"][st]
        z = dict(np.load(wl28 / "probes" / f"{name}.npz"))
        f = np.load(L.RESULTS / cb["rna_features_dir"] / f"{name}.npz")
        x = f["position"][z["lanes"]].astype(np.float64)
        n = z["query_normal"].astype(np.float64)
        # 1. worklog-28 probe state reproduces
        geo = mi.load_dict(T.scene_dict(64, tr, None))
        again = rrp_probes.cast_probes(mi, dr, B, T, geo, T.part_shape_ids(geo, tr), tr, x, n, dirs, to_light, eps, p28["probes"]["chunk"])
        repro = {k: bool(np.array_equal(again[k], z[k])) for k in COMPARE}
        if not all(repro.values()):
            raise SystemExit(f"{name}: worklog-28 probe state does not reproduce: {repro}")
        # 2. same rays in the lit scene; hits only
        s_, t_ = C28.onb(n)
        d = C28.to_world(dirs, s_, n, t_)                    # (Q, K, 3)
        hit = z["hit"]
        qi, ki = np.nonzero(hit)
        dh = d[qi, ki]
        oh = x[qi] + eps * n[qi]
        lit = mi.load_dict(T.scene_dict(64, tr, cb["lighting"]))
        ray = mi.Ray3f(mi.Point3f(*oh.T.astype(np.float32)), mi.Vector3f(*dh.T.astype(np.float32)))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        si = lit.ray_intersect(ray)
        nr = B._vec(si.sh_frame.n)
        flip = np.sum(nr * dh, -1) > 0
        nr[flip] *= -1
        light = B.area_light_samples(lit, cb, si, nr, CentreRNG())
        torch.cuda.synchronize()
        build_s = time.perf_counter() - t0
        t_lit = np.array(si.t)
        same = np.abs(t_lit - z["dist"][hit]).max()
        nl = C28.to_local(nr, s_[qi], n[qi], t_[qi])
        nsame = np.abs(nl - z["remote_normal_local"][hit]).max()
        if same > 1e-4 or nsame > 1e-4 or not np.array(si.is_valid()).all():
            raise SystemExit(f"{name}: lit-scene hits differ from the probe hits (max |dt| {same}, |dn| {nsame})")
        timing = None
        if name in p28["timing_states"]:  # warm repeats of the light sampling alone (protocol timing state)
            times = []
            for _ in range(proto["timing"]["repeats"]):
                torch.cuda.synchronize()
                a = time.perf_counter()
                si2 = lit.ray_intersect(ray)
                B.area_light_samples(lit, cb, si2, nr, CentreRNG())
                torch.cuda.synchronize()
                times.append(time.perf_counter() - a)
            timing = {"hits": int(len(qi)), "shadow_rays": int(len(qi) * light["light_vis"].shape[1]),
                      "seconds_median": float(np.median(times)), "includes": "lit-scene re-cast + 16 emitter samples and shadow rays per hit + host arrays"}
        path = out / "remote_light" / f"{name}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, query_index=qi.astype(np.int32), probe_index=ki.astype(np.int8), camera_dir=(-dh).astype(np.float32),
                 normal=nr.astype(np.float32), canonical=z["remote_canonical"][hit].astype(np.float32),
                 light_dir=light["light_dir"].astype(np.float16), light_weight=light["light_weight"].astype(np.float32),
                 light_vis=light["light_vis"], shape=np.array(hit.shape))
        rec["states"][name] = {"queries": int(hit.shape[0]), "hits": int(len(qi)), "wl28_probe_state_reproduced": repro,
                               "lit_vs_geometry_max_abs_dt": float(same), "lit_vs_geometry_max_abs_dnormal": float(nsame),
                               "remote_lit_fraction": float(light["light_vis"].mean()), "build_s_incl_host": build_s,
                               "light_sampling_timing": timing}
        print(name, json.dumps({k: v for k, v in rec["states"][name].items() if k not in ("wl28_probe_state_reproduced",)}), flush=True)
    L.write_json(out / "remote_light" / "remote_light.json", rec)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
