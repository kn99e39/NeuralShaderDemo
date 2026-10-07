"""Synthetic contract for the remote light inputs (Windows, Mitsuba):
    windows/run.ps1 ../radiometric_relation_state_prototype/tests/test_rrs_light_win.py

Fixed query surface (plane), fixed remote sphere, the protocol area light, and a blocker
that moves out of the sphere's light path.  Same persistent query state, same K, same
probe hits on the sphere -- but the remote point's current light visibility (an input of
the runtime proxy) must change, and repeat evaluations must be identical (stratum centres).
Semantics only; says nothing about architecture viability.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
EXP = HERE.parents[0]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXP / "relational_residual_prototype"))
sys.path.insert(0, str(EXP / "8dna_deformation_replication"))

import numpy as np  # noqa: E402

import rrp_common as C28  # noqa: E402

FAIL = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAIL.append(msg)


def main() -> int:
    import ednalib as L

    mi, dr = L.init_upstream()
    import rna_bridge as B
    import teaset_parts as T

    from rrs_remote_light import CentreRNG

    cb = json.loads((L.EXPERIMENT / C28.protocol()["base_protocol"]).read_text(encoding="utf-8"))
    tl = np.asarray(cb["lighting"]["to_light"], float)
    tl /= np.linalg.norm(tl)
    sun = T.scene_dict(8, None, cb["lighting"])["sun"]
    ball_c = np.array([0.35, 0.18, 0.0])

    def scene(blocker_shift):
        c = ball_c + 0.8 * tl + np.array([blocker_shift, 0.0, 0.0])
        return mi.load_dict({"type": "scene", "sun": sun,
                             "plane": {"type": "rectangle", "to_world": mi.ScalarTransform4f.rotate([1, 0, 0], -90) @ mi.ScalarTransform4f.scale([3, 3, 1])},
                             "ball": {"type": "sphere", "center": ball_c.tolist(), "radius": 0.15},
                             "blocker": {"type": "rectangle", "to_world": mi.ScalarTransform4f.look_at(origin=c.tolist(), target=ball_c.tolist(), up=[0, 0, 1])
                                         @ mi.ScalarTransform4f.scale([0.6, 0.6, 1.0])}})
    x = np.array([[0.0, 0.0, 0.0]])
    n = np.array([[0.0, 1.0, 0.0]])
    dirs = C28.probe_dirs(32)
    s_, t_ = C28.onb(n)
    d = C28.to_world(dirs, s_, n, t_)[0]
    res = {}
    for tag, shift in (("blocked", 0.0), ("open", 5.0)):
        sc = scene(shift)
        ray = mi.Ray3f(mi.Point3f(*np.repeat(x + 2e-4 * n, 32, 0).T.astype(np.float32)), mi.Vector3f(*d.T.astype(np.float32)))
        si = sc.ray_intersect(ray)
        p = B._vec(si.p)
        on_ball = np.array(si.is_valid()) & (np.abs(np.linalg.norm(p - ball_c, axis=1) - 0.15) < 1e-4)
        nr = B._vec(si.sh_frame.n)
        nr[np.sum(nr * d, -1) > 0] *= -1
        a = B.area_light_samples(sc, cb, si, nr, CentreRNG())
        b = B.area_light_samples(sc, cb, si, nr, CentreRNG())
        res[tag] = {"on_ball": on_ball, "p": p, "vis": a["light_vis"], "w": a["light_weight"], "again_equal": all(np.array_equal(a[k], b[k]) for k in a)}
    ob = res["blocked"]["on_ball"]
    check(ob.sum() > 0, f"synthetic: {int(ob.sum())} of 32 probes hit the remote sphere")
    check(np.array_equal(ob, res["open"]["on_ball"]) and np.abs(res["blocked"]["p"][ob] - res["open"]["p"][ob]).max() < 1e-6,
          "synthetic: the same probes hit the same remote points in both configurations")
    vb, vo = res["blocked"]["vis"][ob].mean(), res["open"]["vis"][ob].mean()
    check(vo > vb, f"synthetic: remote light visibility changes with the blocker (lit fraction {vb:.3f} -> {vo:.3f})")
    check(res["blocked"]["again_equal"] and res["open"]["again_equal"], "synthetic: stratum-centre light samples are deterministic")
    print("FAIL" if FAIL else "PASS", len(FAIL), flush=True)
    return 1 if FAIL else 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
