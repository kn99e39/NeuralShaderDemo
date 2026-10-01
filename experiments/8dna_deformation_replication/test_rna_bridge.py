"""Focused tests of the RNA data bridge (run before generating the real datasets).

1. estimator   : the per-lane directional tracer with one fixed light direction
                 matches Mitsuba's prb with a directional emitter (difference
                 comparable to prb's own seed-to-seed difference, mean within 1%).
2. visibility  : pixels the binary visibility marks shadowed receive exactly zero
                 direct light (single-bounce tracer at the pixel centre).
3. schema      : a tiny H5 is written with the official channel names, shapes,
                 dtype and attributes (loaded by RNA's DataModule separately in WSL).
4. pullback    : in T3 features every milk-pot sample's canonical position lies on
                 the canonical milk-pot mesh; T0 canonical == current.
5. area light  : surface irradiance from the features' area-light samples
                 (sum of weight * visibility * n.w) matches Mitsuba's own emitter
                 sampling of the same rectangle: mean within 0.5%, per-pixel
                 difference at Monte Carlo level, same unlit fraction.
6. area mode   : the bridge's per-pixel square area light (training light of the
                 post-hoc area-trained sensitivity run), with every pixel at the
                 protocol direction, matches Mitsuba's C++ path on the evaluation
                 scene (mean within 1%); pixels its any-visibility marks shadowed
                 receive no direct light from the square.
Tests 1-2 use a directional light explicitly (the training-data semantics);
features for tests 4-5 go to a bridge_tests protocol copy at 128^2, never to
the evaluation features directory.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import ednalib as L
import rna_bridge as B
import teaset_parts as T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="protocol/teaset_cross_backbone_locked.json")
    args = ap.parse_args()
    mi, dr = L.init_upstream()
    proto = json.loads(open(L.EXPERIMENT / args.protocol, encoding="utf-8").read())
    out = L.RESULTS / "rna_teaset" / "bridge_tests"
    rec = {}
    res, spp = 128, 1024
    to_light = np.asarray(proto["lighting"]["to_light"], float)
    to_light /= np.linalg.norm(to_light)
    E = proto["lighting"]["irradiance"]
    directional = {"type": "directional", "to_light": proto["lighting"]["to_light"], "irradiance": E}

    # 1. estimator identity
    scene = mi.load_dict(T.scene_dict(res, {}, directional))
    sensor = scene.sensors()[0]
    mine = B.render_view(scene, sensor, res, np.repeat(to_light[None], res * res, 0), spp, 64, 1, E)["color"].reshape(res, res, 3)
    prb_a = np.array(mi.render(scene, spp=spp, seed=5), dtype=np.float32)  # scene reference integrator, wavefront
    prb_b = np.array(mi.render(scene, spp=spp, seed=6), dtype=np.float32)
    d_mine = float(np.abs(L.tonemap(mine) - L.tonemap(prb_a)).mean())
    d_prb = float(np.abs(L.tonemap(prb_b) - L.tonemap(prb_a)).mean())
    rec["estimator"] = {"display_mae_bridge_vs_prb": d_mine, "display_mae_prb_seed_repeat": d_prb,
                        "mean_bridge": float(mine.mean()), "mean_prb": float(0.5 * (prb_a + prb_b).mean()),
                        "pass": d_mine <= 1.15 * d_prb and abs(mine.mean() / (0.5 * (prb_a + prb_b)).mean() - 1) < 0.01}

    # 2. visibility semantics
    centre = B.camera_rays(sensor, res, np.array([[0.5, 0.5]]))
    si, feat = B.surface_features(scene, centre)
    vis = B.visibility(scene, si, feat["normal"], np.repeat(to_light[None], res * res, 0))
    sampler = mi.load_dict({"type": "independent"})
    sampler.seed(3, res * res)
    direct = B._vec(B.trace_directional(scene, sampler, centre, mi.Vector3f(*[float(v) for v in to_light]), E, max_depth=1)).sum(-1)
    hit = feat["hit"]
    rec["visibility"] = {"hit_pixels": int(hit.sum()), "visible": int((vis & hit).sum()),
                         "shadowed_with_nonzero_direct": int(((~vis) & hit & (direct > 0)).sum()),
                         "visible_with_nonzero_direct": int((vis & (direct > 0)).sum())}
    rec["visibility"]["pass"] = rec["visibility"]["shadowed_with_nonzero_direct"] == 0

    # 3. schema
    tiny = dict(proto)
    tiny["rna_dataset"] = dict(proto["rna_dataset"], resolution=64, views={"train": 2, "val": 2}, spp=16, chunk=16)
    tiny["rna_dataset_dir"] = "rna_teaset/bridge_tests"
    tiny_path = out / "tiny_protocol.json"
    L.write_json(tiny_path, tiny)
    import h5py

    for split in ("train", "val"):
        B.generate_h5(argparse.Namespace(protocol=str(tiny_path), state="T0", split=split))
    with h5py.File(out / "teaset_T0_train.h5") as f:
        shapes = {k: [list(f[k].shape), str(f[k].dtype)] for k in f}
        attrs = {k: np.array(f.attrs[k]).tolist() for k in f.attrs}
        alpha = np.array(f["alpha"], np.float32)
        ld = np.array(f["light_dir"], np.float32)
    expected = {"color": 3, "alpha": 1, "position": 3, "tangent": 3, "normal": 3, "camera_dir": 3, "uv": 2,
                "diffuse_direct": 1, "light_dir": 3}
    rec["schema"] = {"shapes": shapes, "attrs": attrs, "coverage": float((alpha > 0).mean()),
                     "light_dir_min_up": float(ld[..., 1].min()), "light_dir_norm_err": float(np.abs(np.linalg.norm(ld, axis=-1) - 1).max()),
                     "pass": all(shapes.get(k, [[0, 0, 0]])[0] == [2, 64 * 64, c] and shapes[k][1] == "float16" for k, c in expected.items())
                     and {"aabb_min", "aabb_max", "resolution"} <= set(attrs)}

    # 4. pullback
    small = dict(proto, res=res, rna_features_dir="rna_teaset/bridge_tests/features")
    small_path = out / "protocol_128.json"
    L.write_json(small_path, small)
    for s in ("T0", "T3"):
        B.features(argparse.Namespace(protocol=str(small_path), state=s, light_seed="A"))
    z0 = np.load(L.RESULTS / small["rna_features_dir"] / "T0.npz")
    z3 = np.load(L.RESULTS / small["rna_features_dir"] / "T3.npz")
    same0 = float(np.abs(z0["canonical_position"] - z0["position"])[z0["hit"]].max())
    mover = list(z3["part_names"]).index("teapot2")
    sel = z3["hit"] & (z3["part"] == mover)
    q = z3["canonical_position"][sel].astype(np.float64)
    d = -z3["camera_dir"][sel].astype(np.float64)
    solo = mi.load_dict({"type": "scene", "part": {"type": "obj", "face_normals": False,
                                                   "filename": str(L.UPSTREAM / "scenes" / "teaset" / T.PARTS["teapot2"][0])}})
    o = q - 0.01 * d
    s0 = solo.ray_intersect(mi.Ray3f(mi.Point3f(*o.T.astype(np.float32)), mi.Vector3f(*d.T.astype(np.float32))))
    ok = np.array(s0.is_valid()) & (np.abs(np.array(s0.t) - 0.01) < 1e-4)
    rec["pullback"] = {"T0_canonical_minus_current_max": same0, "T3_mover_samples": int(sel.sum()),
                       "T3_mover_rehit_fraction": float(ok.mean()),
                       "pass": same0 == 0.0 and ok.mean() > 0.999}
    # 5. area-light samples vs Mitsuba emitter sampling (pixel-centre hits)
    rec["area_light"] = {}
    for s in ("T0", "T3"):
        scene = mi.load_dict(T.scene_dict(96, proto["states"][s], proto["lighting"]))
        si, feat = B.surface_features(scene, B.camera_rays(scene.sensors()[0], 96, np.array([[0.5, 0.5]])))
        n, hit = feat["normal"], feat["hit"]
        ours = np.zeros(len(n))
        for r in range(8):
            ls = B.area_light_samples(scene, proto, si, n, np.random.default_rng(100 + r))
            ours += (ls["light_weight"] * ls["light_vis"] * np.maximum(0, np.einsum("lmk,lk->lm", ls["light_dir"], n))).mean(1) / 8
        ref = np.zeros(len(n))
        sampler = mi.load_dict({"type": "independent"})
        sampler.seed(7, len(n))
        for _ in range(256):
            ds, w = scene.sample_emitter_direction(si, sampler.next_2d(), True, si.is_valid())
            ref += np.array(w.x) * np.maximum(0, np.sum(n * B._vec(ds.d), -1)) / 256
        a, b = ours[hit], ref[hit]
        rec["area_light"][s] = {"mean_ours": float(a.mean()), "mean_mitsuba": float(b.mean()),
                                "rel_pixel_mae": float(np.abs(a - b).mean() / b.mean()),
                                "unlit_ours": float((a == 0).mean()), "unlit_mitsuba": float((b == 0).mean())}
    rec["area_light"]["pass"] = all(abs(v["mean_ours"] / v["mean_mitsuba"] - 1) < 0.005 and v["rel_pixel_mae"] < 0.03
                                    and abs(v["unlit_ours"] - v["unlit_mitsuba"]) < 0.005
                                    for v in rec["area_light"].values() if isinstance(v, dict))
    # 6. area mode of the tracer (sensitivity run's training light)
    lt = proto["lighting"]
    area = {"type": "area", "distance": lt["distance"], "half_angle_deg": lt["half_angle_deg"]}
    ref_scene = mi.load_dict(T.scene_dict(res, {}, lt))
    ra = np.array(mi.render(ref_scene, spp=spp, seed=5), dtype=np.float32)
    rb = np.array(mi.render(ref_scene, spp=spp, seed=6), dtype=np.float32)
    geo = mi.load_dict(T.scene_dict(res, {}, None))
    tl = np.repeat(to_light[None], res * res, 0)
    mine = B.render_view(geo, ref_scene.sensors()[0], res, tl, spp, 64, 1, E, area=area)["color"].reshape(res, res, 3)
    centre = B.camera_rays(ref_scene.sensors()[0], res, np.array([[0.5, 0.5]]))
    si, fc = B.surface_features(geo, centre)
    sq = B.area_squares(tl, E, area)
    vis = B.visibility_any(geo, si, fc["normal"], sq)
    direct = np.zeros(res * res)
    for k in range(64):
        sampler = mi.load_dict({"type": "independent"})
        sampler.seed(100 + k, res * res)
        direct += B._vec(B.trace_directional(geo, sampler, centre, mi.Vector3f(*[float(v) for v in to_light]), E,
                                             max_depth=1, squares=sq)).sum(-1)
    hit = fc["hit"]
    ref_mean = float(0.5 * (ra + rb).mean())
    rec["area_mode"] = {"mean_bridge": float(mine.mean()), "mean_path": ref_mean,
                        "display_mae_bridge_vs_path": float(np.abs(L.tonemap(mine) - L.tonemap(ra)).mean()),
                        "display_mae_path_seed_repeat": float(np.abs(L.tonemap(rb) - L.tonemap(ra)).mean()),
                        "shadowed_with_nonzero_direct": int(((~vis) & hit & (direct > 0)).sum()),
                        "visible_pixels": int((vis & hit).sum())}
    rec["area_mode"]["pass"] = (abs(mine.mean() / ref_mean - 1) < 0.01 and rec["area_mode"]["shadowed_with_nonzero_direct"] == 0)
    rec["pass"] = all(v["pass"] for v in rec.values() if isinstance(v, dict))
    L.write_json(out / "bridge_tests.json", rec)
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk not in ("shapes", "attrs")}) for k, v in rec.items()}, indent=1, default=str))
    return 0 if rec["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
