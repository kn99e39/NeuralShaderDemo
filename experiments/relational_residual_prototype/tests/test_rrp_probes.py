"""Probe-stage tests (Windows, Mitsuba):
    windows/run.ps1 ../relational_residual_prototype/tests/test_rrp_probes.py

1. Synthetic fixture: a fixed plane (query surface) and a sphere occluder that moves.
   Proves semantics only: the current relation state changes with the occluder,
   the persistent query state does not, and remote hits pull back exactly onto the
   canonical occluder.  It says nothing about architecture viability.
2. Real teaset T0 / T3 on a subsample of stable ROI queries: query identity,
   determinism, canonical pull-back of milk-pot hits onto the T0 milk pot, and that
   stationary-part hits are their own canonical points.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "8dna_deformation_replication"))

import numpy as np  # noqa: E402

import rrp_common as C  # noqa: E402

FAIL = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAIL.append(msg)


def shape_id_at(mi, T, scene, o, d):
    si = scene.ray_intersect(mi.Ray3f(mi.Point3f(*o), mi.Vector3f(*d)))
    return int(np.array(T.shape_ids(si))[0])


def synthetic(mi, dr, B, T, probes_mod):
    r, c0 = 0.25, np.array([0.0, 0.45, 0.0])

    def scene_at(t):
        return mi.load_dict({"type": "scene",
                             "plane": {"type": "rectangle", "to_world": mi.ScalarTransform4f.rotate([1, 0, 0], -90) @ mi.ScalarTransform4f.scale([3, 3, 1])},
                             "ball": {"type": "sphere", "center": [float(v) for v in c0 + t], "radius": r}})
    rng = np.random.default_rng(0)
    x = np.stack([rng.uniform(-0.4, 0.4, 400), np.zeros(400), rng.uniform(-0.4, 0.4, 400)], -1)
    n = np.tile([0.0, 1.0, 0.0], (400, 1))
    dirs = C.probe_dirs(32)
    to_light = np.array([0.0, 1.0, 0.0])
    out = {}
    for tag, t in (("near", np.zeros(3)), ("moved", np.array([1.5, 0.0, 0.0]))):
        sc = scene_at(t)
        ids = {"plate": shape_id_at(mi, T, sc, (2.0, 1.0, 2.0), (0.0, -1.0, 0.0)),
               "teapot2": shape_id_at(mi, T, sc, tuple(c0 + t + [0, 2, 0]), (0.0, -1.0, 0.0))}
        out[tag] = probes_mod.cast_probes(mi, dr, B, T, sc, ids, {"teapot2": t.tolist()}, x, n, dirs, to_light, 2e-4, 1 << 20)
        out[tag]["frame"] = C.onb(n)
        out[tag]["t"] = t
    a, b = out["near"], out["moved"]
    check(all(np.array_equal(u, v) for u, v in zip(a["frame"], b["frame"])), "synthetic: persistent query frame identical in both configurations")
    check(a["hit"].mean() > 0.05, f"synthetic: near occluder is seen by probes ({a['hit'].mean():.3f} of probes hit)")
    check(not np.array_equal(a["hit"], b["hit"]), "synthetic: current relation state (hit pattern) changes when the occluder moves")
    for tag in ("near", "moved"):
        z = out[tag]
        sel = z["hit"] & (z["remote_part"] == C.protocol()["parts_order"].index("teapot2"))
        if sel.any():
            d = np.linalg.norm(z["remote_canonical"][sel] - c0, axis=-1)
            check(np.abs(d - r).max() < 1e-4, f"synthetic {tag}: occluder hits pull back onto the canonical sphere (max |d-r| {np.abs(d - r).max():.1e})")
    sc = scene_at(np.zeros(3))  # one scene instance: shape ids are per loaded scene
    again = probes_mod.cast_probes(mi, dr, B, T, sc, {"plate": shape_id_at(mi, T, sc, (2.0, 1.0, 2.0), (0, -1, 0)),
                                                      "teapot2": shape_id_at(mi, T, sc, tuple(c0 + [0, 2, 0]), (0, -1, 0))},
                                   {}, x, n, dirs, to_light, 2e-4, 1 << 20)
    check(all(np.array_equal(again[k], a[k]) for k in ("hit", "dist", "remote_normal_local", "remote_part")), "synthetic: probe state is deterministic")


def teaset(mi, dr, B, T, probes_mod, L):
    proto = C.protocol()
    cb = json.loads((L.EXPERIMENT / proto["base_protocol"]).read_text(encoding="utf-8"))
    names = proto["parts_order"]
    f0 = dict(np.load(L.RESULTS / cb["rna_features_dir"] / "T0.npz"))
    f3 = dict(np.load(L.RESULTS / cb["rna_features_dir"] / "T3.npz"))
    pix = np.flatnonzero(np.load(L.RESULTS / cb["gt_output"] / "rois_T3.npz")["mask_interaction"].reshape(-1))
    k = int(f0["samples"])
    stable = C.stable_pixels(f0, f3, pix, pix, k, names.index(C.QUERY_PART), require_identity=True)
    check(stable.mean() > 0.99, f"teaset: stable T0/T3 ROI pixels {stable.sum()}/{len(pix)}")
    lanes = C.roi_lanes(pix[stable], k)[::16]
    for key in ("canonical_position", "normal", "camera_dir", "part"):
        check(np.array_equal(f0[key][lanes], f3[key][lanes]), f"teaset: stable-query {key} identical T0 vs T3")
    x, n = f0["position"][lanes].astype(np.float64), f0["normal"][lanes].astype(np.float64)
    dirs = C.probe_dirs(proto["probes"]["K"])
    tl = np.asarray(cb["lighting"]["to_light"], float)
    tl /= np.linalg.norm(tl)
    res = {}
    for s in ("T0", "T3"):
        tr = cb["states"][s]
        sc = mi.load_dict(T.scene_dict(64, tr, None))
        res[s] = probes_mod.cast_probes(mi, dr, B, T, sc, T.part_shape_ids(sc, tr), tr, x, n, dirs, tl, proto["probes"]["origin_offset"], 1 << 20)
        res[s]["scene"] = sc
    check(not np.array_equal(res["T0"]["remote_part"], res["T3"]["remote_part"]), "teaset: probe state differs between T0 and T3")
    again = probes_mod.cast_probes(mi, dr, B, T, res["T3"]["scene"], T.part_shape_ids(res["T3"]["scene"], cb["states"]["T3"]),
                                   cb["states"]["T3"], x, n, dirs, tl, proto["probes"]["origin_offset"], 1 << 20)
    check(np.array_equal(again["remote_canonical"], res["T3"]["remote_canonical"]), "teaset: T3 probe state is deterministic")
    milk = names.index("teapot2")
    z3 = res["T3"]
    sel = z3["remote_part"] == milk
    check(sel.sum() > 0, f"teaset: T3 probes hitting the milk pot: {int(sel.sum())}")
    # pull-back: every T3 milk-pot hit, moved back by its translation, lies on the T0 milk pot
    qi, ki = np.nonzero(sel)
    s_, t_ = C.onb(n)
    d = C.to_world(dirs, s_, n, t_)[qi, ki]
    p = z3["remote_canonical"][sel].astype(np.float64)
    sc0 = res["T0"]["scene"]
    ids0 = T.part_shape_ids(sc0, None)
    ray = mi.Ray3f(mi.Point3f(*(p - 1e-3 * d).T.astype(np.float32)), mi.Vector3f(*d.T.astype(np.float32)))
    si = sc0.ray_intersect(ray)
    sid = np.array(T.shape_ids(si))
    tt = np.array(si.t)
    check(np.mean((sid == ids0["teapot2"]) & (np.abs(tt - 1e-3) < 2e-4)) > 0.99,
          f"teaset: canonical pull-back of T3 milk-pot hits lands on the T0 milk pot ({np.mean(sid == ids0['teapot2']):.4f})")
    for s in ("T0", "T3"):
        z = res[s]
        st = z["hit"] & (z["remote_part"] != milk)
        qi, ki = np.nonzero(st)
        world = x[qi] + proto["probes"]["origin_offset"] * n[qi] + C.to_world(dirs, s_, n, t_)[qi, ki] * z["dist"][st][:, None]
        check(np.abs(world - z["remote_canonical"][st]).max() < 1e-3, f"teaset {s}: stationary-part hits are their own canonical points")


def main() -> int:
    import ednalib as L

    mi, dr = L.init_upstream()
    import rna_bridge as B
    import teaset_parts as T

    import rrp_probes

    synthetic(mi, dr, B, T, rrp_probes)
    teaset(mi, dr, B, T, rrp_probes, L)
    print("FAIL" if FAIL else "PASS", len(FAIL), flush=True)
    return 1 if FAIL else 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
