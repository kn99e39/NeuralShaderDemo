"""Focused validation of the teaset correspondence adapter (run before any moved-state neural render).

1. identity   : at the canonical state the adapter ('attached' and 'fixed')
                and the unmodified upstream call render the same seeds; their
                difference must be far below the neural seed-to-seed noise.
2. rigid map  : in a translated state, every primary hit on a part, pulled
                back by that part's translation, lies on that part's canonical
                mesh (ray re-hit of the part alone at the expected distance).
3. dispatch   : per-lane part ids from the adapter match the hit mesh for
                every asset lane (no uncovered lanes).
4. identity of inputs: checkpoint and asset hashes equal the recorded
                release hashes.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import ednalib as L
import teaset_parts as T

RELEASE_HASHES = {
    # recorded in asset_audit.json (worklog 19) from the released scenes.zip
    "scenes/teaset/teaplate.obj": "dfcda160a54e7ab5b0a94acf0c5b4d2c13bee66af4a96e1bc760127f34d46a6b",
    "scenes/teaset/teapot2.obj": "c15e2e9b9b43b6c00277b3a94edec6862a77a256268f019134fbbefdc0e9dada",
    "scenes/teaset/teapot3.obj": "40e04b5d7d61cc5fa6bc2cc49e337e512f3e31092d6f5ec4cc23623809e06129",
    "scenes/teaset/teapot4.obj": "bea0bc855fb93e8879e8d9892c32bbae1a80e6c223c1a68c014889daf1c40d81",
    "scenes.zip": "e60653896a978fb7a386c09c85f477cf98054082f41aa6745ea8d2538e0d1d19",
    "weights.zip": "ef134a50bfade0431c97a71fd224dd833a56faeb79bf2ae14312f3a0be0bb13c",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--spp", type=int, default=32)
    ap.add_argument("--probe", default='{"teapot2": [0.12, 0.0, 0.0], "teapot3": [0.0, 0.0, 0.1]}')
    ap.add_argument("--out", default="correspondence_tests")
    args = ap.parse_args()
    out = L.RESULTS / args.out
    mi, dr = L.init_upstream()
    probe = json.loads(args.probe)
    rec: dict = {"environment": L.environment_record(), "tests": {}}

    # 4. input identity
    hashes = {k: L.sha256(L.UPSTREAM / k) for k in RELEASE_HASHES}
    rec["tests"]["input_identity"] = {
        "release_hashes_match": all(hashes[k] == v for k, v in RELEASE_HASHES.items()),
        "hashes": hashes,
        "teaset_ckpt_sha256": L.sha256(L.checkpoint_path("teaset")),
    }

    # 1. canonical identity
    scene0 = mi.load_dict(T.scene_dict(args.res))
    bbox0 = scene0.shapes()[0].bbox()
    integ = L.load_neural_integrator("teaset")
    base = integ.asset_models[0]
    imgs = {}
    for mode in ("upstream", "attached", "fixed"):
        T.configure(integ, base, scene0, mode, {}, bbox0)
        imgs[mode], _ = L.render_chunked(scene0, integ, args.spp, 4, 0)
    T.configure(integ, base, scene0, "upstream")
    seed_b, _ = L.render_chunked(scene0, integ, args.spp, 4, 50_000)
    noise = L.metrics(seed_b, imgs["upstream"])
    ident = {m: L.metrics(imgs[m], imgs["upstream"]) for m in ("attached", "fixed")}
    ident_max = {m: float(np.abs(imgs[m] - imgs["upstream"]).max()) for m in ("attached", "fixed")}
    rec["tests"]["canonical_identity"] = {
        "adapter_vs_upstream_same_seed": ident, "adapter_vs_upstream_max_abs_linear": ident_max,
        "upstream_seed_repeat": noise,
        "criterion": "adapter-vs-upstream display MAE < 1% of the seed-to-seed display MAE",
        "pass": all(ident[m]["display_mae"] < 0.01 * noise["display_mae"] for m in ident),
    }

    # 2. rigid map, 3. dispatch
    scene1 = mi.load_dict(T.scene_dict(args.res, probe))
    ids1 = T.part_shape_ids(scene1, probe)
    sensor = scene1.sensors()[0]
    n = args.res
    j, i = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    film = np.stack([(i.ravel() + 0.5) / n, (j.ravel() + 0.5) / n], 0).astype(np.float32)
    ray, _ = sensor.sample_ray(0.0, 0.5, mi.Point2f(*film), mi.Point2f(0.5, 0.5))
    si = scene1.ray_intersect(ray)
    sid = np.array(T.shape_ids(si))
    p = np.stack([np.array(si.p.x), np.array(si.p.y), np.array(si.p.z)], -1)
    d = np.stack([np.array(ray.d.x), np.array(ray.d.y), np.array(ray.d.z)], -1)
    rigid = {}
    for part in T.PARTS:
        sel = sid == ids1[part]
        t = np.asarray(probe.get(part, (0, 0, 0)), float)
        q = p[sel] - t                     # analytic pullback of the hit point
        o = q - 0.01 * d[sel]
        r0 = mi.Ray3f(mi.Point3f(*o.T.astype(np.float32)), mi.Vector3f(*d[sel].T.astype(np.float32)))
        # re-hit against the canonical part alone: in the full canonical scene a
        # surface uncovered by a moved part is legitimately occluded again
        solo = mi.load_dict({"type": "scene", "part": {"type": "obj", "face_normals": False,
                             "filename": str(L.UPSTREAM / "scenes" / "teaset" / T.PARTS[part][0])}})
        s0 = solo.ray_intersect(r0)
        hit0 = np.stack([np.array(s0.p.x), np.array(s0.p.y), np.array(s0.p.z)], -1)
        err = np.linalg.norm(hit0 - q, axis=1)
        same = np.array(s0.is_valid()) & (np.abs(np.array(s0.t) - 0.01) < 1e-4)
        rigid[part] = {"pixels": int(sel.sum()), "translation": t.tolist(),
                       "canonical_rehit_same_part_fraction": float(same.mean()) if sel.any() else None,
                       "pullback_position_error_p99": float(np.percentile(err[same], 99)) if same.any() else None}
    asset = np.array(si.instance.eq_(mi.ShapePtr(scene1.shapes()[0])))
    covered = np.isin(sid[asset], list(ids1.values()))
    rec["tests"]["rigid_map"] = rigid
    rec["tests"]["dispatch_coverage"] = {"asset_lanes": int(asset.sum()), "covered_fraction": float(covered.mean())}
    rec["tests"]["rigid_map_pass"] = all(
        v["pixels"] == 0 or (v["canonical_rehit_same_part_fraction"] > 0.99 and v["pullback_position_error_p99"] < 1e-4)
        for v in rigid.values()) and bool(covered.all())

    # the adapter also runs end to end on the probe state (finite output)
    T.configure(integ, base, scene1, "attached", probe, bbox0)
    img1, _ = L.render_chunked(scene1, integ, 4, 4, 0)
    rec["tests"]["probe_state_render_finite"] = bool(np.isfinite(img1).all())
    rec["pass"] = bool(rec["tests"]["input_identity"]["release_hashes_match"] and rec["tests"]["canonical_identity"]["pass"]
                       and rec["tests"]["rigid_map_pass"] and rec["tests"]["probe_state_render_finite"])
    L.write_json(out / "correspondence_tests.json", rec)
    print(json.dumps({k: (v.get("pass") if isinstance(v, dict) and "pass" in v else v) for k, v in rec["tests"].items()
                      if k != "input_identity"}, default=str)[:2000])
    print("PASS" if rec["pass"] else "FAIL")
    return 0 if rec["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
