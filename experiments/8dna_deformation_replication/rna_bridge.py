"""RNA data bridge for the teaset common-light regime (Mitsuba -> RNA H5 / features).

RNA's official generator renders its H5 with a custom Cycles build.  This
bridge populates the same H5 schema from the teaset Mitsuba scene the 8DNA
experiment uses (released Ni_palik conductors, worklog-21 geometry), so that
RNA and 8DNA are compared against one physical renderer.  RNA source, network
and training code are unchanged; only the data producer differs.

Channel semantics follow the official pipeline (rna/datasets.py,
training_dataset/*, scripts/setup_blender_scene.py):
  color        pixel-mean radiance under one directional light of irradiance E
               (full path tracing, black world; misses count as 0)
  alpha        fraction of pixel samples that hit the asset
  position     world hit position       } mean over the pixel's hit samples
  normal       shading normal, flipped  } (Cycles flips N and Ng on
               to the camera side       }  back-facing hits)
  camera_dir   unit vector hit -> camera (Geometry "Incoming")
  tangent, uv  normalized dp/du and mesh uv (read by the loader, unused by
               the surface triplane model)
  diffuse_direct  1 if N.L > 0 and the light is unoccluded from the pixel-
               centre hit, else 0 (official: white-diffuse, 0-bounce, 1-spp
               pass thresholded > 0)
  light_dir    world direction towards the light, per pixel (training:
               independent per pixel; validation: one per image), uniform on
               the upper hemisphere with height = U * 0.999
Frame: Mitsuba's world, +y up.  RNA's generator is +z up; its network sees
only world-space vectors and AABB-normalized positions, so the up axis only
enters the camera and light hemisphere sampling, which use +y here.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import ednalib as L
import teaset_parts as T


def _vec(v) -> np.ndarray:
    return np.stack([np.array(v.x), np.array(v.y), np.array(v.z)], -1)


def area_squares(to_light: np.ndarray, irradiance: float, area: dict) -> dict:
    """Per-lane square emitters built like teaset_parts.scene_dict's area light.

    For lane direction d: centre distance*d, facing the origin, half edge
    distance*tan(half_angle), in-plane axes from Mitsuba's look_at with the same
    up-vector rule, radiance scaled so the irradiance at the origin is E. For the
    protocol direction this is exactly the evaluation emitter.
    """
    d = to_light / np.linalg.norm(to_light, axis=-1, keepdims=True)
    dist = float(area["distance"])
    half = dist * np.tan(np.deg2rad(float(area["half_angle_deg"])))
    up = np.where(np.abs(d[:, 1:2]) > 0.9, np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, 1.0, 0.0]]))
    fwd = -d
    left = np.cross(up, fwd)
    left /= np.linalg.norm(left, axis=-1, keepdims=True)
    new_up = np.cross(fwd, left)
    return {"centre": d * dist, "e1": left * half, "e2": new_up * half, "normal": fwd,
            "radiance": irradiance * dist ** 2 / (2 * half) ** 2, "area": (2 * half) ** 2}


def trace_directional(scene, sampler, ray, to_light, irradiance: float, rr_depth: int = 5, max_depth: int = 1024,
                      squares: dict | None = None):
    """Path tracer with one light per lane: a delta directional light, or a square area light.

    Same estimator as Mitsuba's prb for a directional emitter and a black
    world: next-event estimation towards the light at every smooth vertex,
    BSDF-sampled continuation, Russian roulette from rr_depth.  With `squares`
    (from area_squares; the scene must not contain the emitters) next-event
    estimation samples a uniform point y on the lane's square instead: weight
    L_r * A * cos_light / |y - x|^2 and a segment shadow test, as Mitsuba's
    emitter sampling of the same rectangle.  The delta path is unchanged.
    """
    mi, dr = L.init_upstream()
    ctx = mi.BSDFContext()
    if squares is not None:
        v3 = lambda a: mi.Vector3f(*np.asarray(a, np.float32).T)
        centre, e1, e2, n_l = (v3(squares[k]) for k in ("centre", "e1", "e2", "normal"))
        power = float(squares["radiance"] * squares["area"])
    ray = mi.Ray3f(ray)
    Lr = mi.Spectrum(0.0)
    beta = mi.Spectrum(1.0)
    eta = mi.Float(1.0)
    depth = mi.UInt32(0)
    active = mi.Bool(True)
    loop = mi.Loop(name="directional path tracer", state=lambda: (sampler, ray, Lr, beta, eta, depth, active))
    loop.set_max_iterations(max_depth)
    while loop(active):
        si = scene.ray_intersect(ray, active)
        active &= si.is_valid()
        bsdf = si.bsdf(ray)
        emit = active & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
        if squares is None:
            f = bsdf.eval(ctx, si, si.to_local(to_light), emit)
            unoccluded = ~scene.ray_test(si.spawn_ray(to_light), emit)
            Lr[emit & unoccluded] += beta * f * irradiance
        else:
            u = sampler.next_2d(emit)
            y = mi.Point3f(centre + (2 * u.x - 1) * e1 + (2 * u.y - 1) * e2)
            dv = y - si.p
            dist2 = dr.squared_norm(dv)
            wl = dv * dr.rsqrt(dist2)
            weight = power * dr.maximum(dr.dot(n_l, -wl), 0.0) / dist2
            f = bsdf.eval(ctx, si, si.to_local(wl), emit)
            unoccluded = ~scene.ray_test(si.spawn_ray_to(y), emit)
            Lr[emit & unoccluded] += beta * f * weight
        bs, w = bsdf.sample(ctx, si, sampler.next_1d(active), sampler.next_2d(active), active)
        beta[active] *= w
        eta[active] *= bs.eta
        ray = si.spawn_ray(si.to_world(bs.wo))
        depth[active] += 1
        q = dr.minimum(dr.max(beta) * dr.sqr(eta), 0.99)
        rr = depth >= rr_depth
        active &= (sampler.next_1d(active) < q) | ~rr
        beta[rr] = beta * dr.rcp(q)
        active &= dr.any(dr.neq(beta, 0.0)) & (depth < max_depth)
    return Lr


def sensor_dict(origin, target, fov: float, res: int) -> dict:
    mi, _ = L.init_upstream()
    return {"type": "perspective", "fov": fov,
            "to_world": mi.ScalarTransform4f.look_at(origin=list(map(float, origin)), target=list(map(float, target)), up=[0, 1, 0]),
            "film": {"type": "hdrfilm", "width": res, "height": res, "filter": {"type": "box"}}}


def camera_rays(sensor, res: int, offsets: np.ndarray):
    """Rays for every pixel x every sub-pixel offset (offsets: (K, 2) in [0,1)); lane = pixel * K + k."""
    mi, _ = L.init_upstream()
    k = len(offsets)
    j, i = np.meshgrid(np.arange(res), np.arange(res), indexing="ij")
    px = np.repeat(np.stack([i.ravel(), j.ravel()], -1), k, 0).astype(np.float64)
    film = (px + np.tile(offsets, (res * res, 1))) / res
    ray, _ = sensor.sample_ray(0.0, 0.5, mi.Point2f(*film.T.astype(np.float32)), mi.Point2f(0.5, 0.5))
    return ray


def surface_features(scene, ray):
    """First-hit features with official-pipeline semantics (per lane)."""
    mi, dr = L.init_upstream()
    si = scene.ray_intersect(ray)
    hit = np.array(si.is_valid())
    back = np.array(dr.dot(si.n, ray.d) > 0)
    n = _vec(si.sh_frame.n)
    n[back] *= -1
    t = _vec(dr.normalize(si.dp_du))
    return si, {"hit": hit, "position": _vec(si.p), "normal": n, "camera_dir": -_vec(ray.d),
                "tangent": np.nan_to_num(t), "uv": np.stack([np.array(si.uv.x), np.array(si.uv.y)], -1),
                "shape_id": np.array(T.shape_ids(si))}


def visibility(scene, si, normal: np.ndarray, to_light: np.ndarray) -> np.ndarray:
    """Official binary visibility: N.L > 0 and an unoccluded shadow ray."""
    mi, dr = L.init_upstream()
    tl = mi.Vector3f(*to_light.T.astype(np.float32))
    blocked = np.array(scene.ray_test(si.spawn_ray(tl), si.is_valid()))
    return (np.sum(normal * to_light, -1) > 0) & ~blocked & np.array(si.is_valid())


def visibility_any(scene, si, normal: np.ndarray, squares: dict, k: int = 4) -> np.ndarray:
    """Official binary visibility under an area light: 1 if any part of the lane's
    square is unoccluded and above the surface (k x k stratum centres), as
    RNA's Cycles diffuse_direct pass thresholded > 0 with a soft light."""
    mi, dr = L.init_upstream()
    x = _vec(si.p)
    valid = np.array(si.is_valid())
    out = np.zeros(len(x), bool)
    g = (np.arange(k) + 0.5) / k * 2 - 1
    for a in g:
        for b in g:
            y = squares["centre"] + a * squares["e1"] + b * squares["e2"]
            w = y - x
            w /= np.maximum(np.linalg.norm(w, axis=-1, keepdims=True), 1e-12)
            blocked = np.array(scene.ray_test(si.spawn_ray_to(mi.Point3f(*y.T.astype(np.float32))), si.is_valid()))
            out |= (np.sum(normal * w, -1) > 0) & ~blocked & valid
    return out


def hemisphere_dirs(rng, n: int, height_scale: float) -> np.ndarray:
    h = rng.random(n) * height_scale
    r = np.sqrt(np.maximum(0.0, 1.0 - h * h))
    phi = 2 * np.pi * rng.random(n)
    return np.stack([r * np.cos(phi), h, r * np.sin(phi)], -1)


def render_view(scene, sensor, res: int, to_light: np.ndarray, spp: int, chunk: int, seed: int, irradiance: float,
                record: bool = False, area: dict | None = None):
    """color, alpha and hit-averaged AOVs for one view; to_light: (res*res, 3).

    area=None: one delta directional light per pixel (RNA's default, light_radius
    0).  area={"distance", "half_angle_deg"}: a square area light per pixel
    centred on that direction (the official generator's light_radius > 0 case,
    with the evaluation emitter's shape); diffuse_direct is then "any of it
    visible".

    record=False (wavefront) is the validated default; record=True is used only
    where probe_bridge_record.py has shown it matches the CPU backend.
    """
    mi, dr = L.init_upstream()
    dr.set_flag(dr.JitFlag.LoopRecord, record)
    try:
        npx = res * res
        color = np.zeros((npx, 3))
        hits = np.zeros(npx)
        acc = {k: np.zeros((npx, d)) for k, d in (("position", 3), ("normal", 3), ("camera_dir", 3), ("tangent", 3), ("uv", 2))}
        rng = np.random.default_rng(seed)
        for c in range(spp // chunk):
            offsets = rng.random((chunk, 2))
            ray = camera_rays(sensor, res, offsets)
            _, feat = surface_features(scene, ray)
            sampler = mi.load_dict({"type": "independent"})
            sampler.seed((seed * 4096 + c) & 0xFFFFFFFF, npx * chunk)
            tl = np.repeat(to_light, chunk, 0)
            sq = None if area is None else area_squares(tl, irradiance, area)
            Lr = _vec(trace_directional(scene, sampler, ray, mi.Vector3f(*tl.T.astype(np.float32)), irradiance,
                                        squares=sq))
            color += Lr.reshape(npx, chunk, 3).sum(1)
            h = feat["hit"].reshape(npx, chunk)
            hits += h.sum(1)
            for k in acc:
                acc[k] += (feat[k].reshape(npx, chunk, -1) * h[..., None]).sum(1)
        out = {"color": color / spp, "alpha": (hits / spp)[:, None]}
        denom = np.maximum(hits, 1)[:, None]
        for k in acc:
            out[k] = acc[k] / denom
        for k in ("normal", "camera_dir", "tangent"):
            out[k] /= np.maximum(np.linalg.norm(out[k], axis=-1, keepdims=True), 1e-12)
            out[k][hits == 0] = 0
        centre = camera_rays(sensor, res, np.array([[0.5, 0.5]]))
        si, fc = surface_features(scene, centre)
        vis = (visibility(scene, si, fc["normal"], to_light) if area is None
               else visibility_any(scene, si, fc["normal"], area_squares(to_light, irradiance, area)))
        out["diffuse_direct"] = vis.astype(np.float32)[:, None]
        out["light_dir"] = to_light
        return out
    finally:
        dr.set_flag(dr.JitFlag.LoopRecord, False)


def generate_h5(args) -> None:
    import h5py

    mi, dr = L.init_upstream()
    proto = json.loads(open(L.EXPERIMENT / args.protocol, encoding="utf-8").read())
    gen = proto["rna_dataset"]
    # optional training light shape (absent: delta directional light per pixel)
    area = gen["light"] if gen.get("light", {}).get("type") == "area" else None
    tr = proto["states"][args.state]
    scene = mi.load_dict(T.scene_dict(64, tr, None))  # geometry/materials only; no emitter is used
    bbox = scene.shapes()[0].bbox()
    res, n_views = gen["resolution"], gen["views"][args.split]
    out = L.RESULTS / proto["rna_dataset_dir"] / f"teaset_{args.state}_{args.split}.h5"
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(gen["seeds"][args.split] + 7919 * list(proto["states"]).index(args.state))
    chans = {"color": 3, "alpha": 1, "position": 3, "tangent": 3, "normal": 3, "camera_dir": 3, "uv": 2,
             "diffuse_direct": 1, "light_dir": 3}
    cams = []
    with h5py.File(out, "w") as f:
        ds = {k: f.create_dataset(k, (n_views, res * res, c), dtype=np.float16) for k, c in chans.items()}
        for v in range(n_views):
            h = rng.random() * 0.9999
            r = np.sqrt(max(0.0, 1 - h * h))
            phi = 2 * np.pi * rng.random()
            origin = np.array(gen["camera_lookat"]) + gen["camera_radius"] * np.array([r * np.cos(phi), h, r * np.sin(phi)])
            sensor = mi.load_dict(sensor_dict(origin, gen["camera_lookat"], gen["fov"], res))
            if args.split == "train":
                to_light = hemisphere_dirs(rng, res * res, 0.999)
            else:
                to_light = np.repeat(hemisphere_dirs(rng, 1, 0.999), res * res, 0)
            view = render_view(scene, sensor, res, to_light, gen["spp"], gen["chunk"], int(rng.integers(1 << 20)), gen["irradiance"],
                               area=area)
            for k in chans:
                ds[k][v] = np.nan_to_num(view[k]).astype(np.float16)
            cams.append({"origin": origin.tolist(), "target": gen["camera_lookat"], "fov": gen["fov"]})
            print(f"view {v + 1}/{n_views}", flush=True)
        f.attrs["base_radius"] = 1.0 / res / 2.0
        f.attrs["resolution"] = [res, res]
        f.attrs["aabb_min"] = np.array(bbox.min, dtype=np.float32)
        f.attrs["aabb_max"] = np.array(bbox.max, dtype=np.float32)
        f.attrs["bridge"] = "experiments/8dna_deformation_replication/rna_bridge.py (Mitsuba, teaset common-light)"
        f.attrs["state"] = args.state
        # Provenance so a dataset from a superseded regime cannot be mistaken for a
        # current one: the lighting and the generation settings it was rendered under.
        f.attrs["lighting"] = json.dumps(proto["lighting"], sort_keys=True)
        f.attrs["generation"] = json.dumps(gen, sort_keys=True)
        f.attrs["project_commit"] = L.git_head(L.ROOT) or ""
    L.write_json(out.with_suffix(".cameras.json"), {"state": args.state, "split": args.split, "cameras": cams,
                                                    "aabb": [list(bbox.min), list(bbox.max)], "sha256": L.sha256(out)})
    print("wrote", L.rel(out))


def area_light_samples(scene, proto: dict, si, normal: np.ndarray, rng) -> dict:
    """Stratified samples of the protocol's area light for every lane of `si`.

    RNA's own light model (rna/lights.py RectangularLight): a point y uniform on
    the emitter rectangle, direction w = (y - x)/|y - x| from the current hit x
    and weight E = L_r * cos_light / (|y - x|^2 * pdf_area), the perpendicular
    irradiance that sample stands for; the network's directional response is
    scaled by E / training_light_intensity.  Visibility is the binary test of
    the segment x -> y (and n.w > 0), the per-direction semantics of the
    training data's diffuse_direct.  The rectangle is read from the loaded
    scene, so it is exactly the emitter the reference was rendered with.
    Returns (lanes, M, .) arrays.
    """
    mi, dr = L.init_upstream()
    ls = proto["rna_inference"]["light_sampling"]
    nx, ny = ls["strata"]
    m = ls["samples_per_subpixel"]
    if nx * ny != m:
        raise ValueError("strata must cover samples_per_subpixel")
    emitters = [s for s in scene.shapes() if s.is_emitter()]
    if len(emitters) != 1 or proto["lighting"]["type"] != "area":
        raise RuntimeError("expected exactly one area emitter")
    rect = emitters[0]
    radiance = T.scene_dict(8, None, proto["lighting"])["sun"]["emitter"]["radiance"]["value"]
    lanes = len(normal)
    x = _vec(si.p)
    valid = si.is_valid()
    dirs = np.zeros((lanes, m, 3), np.float32)
    weight = np.zeros((lanes, m), np.float32)
    vis = np.zeros((lanes, m), bool)
    for j in range(m):
        a, b = j % nx, j // nx
        u = ((a + rng.random(lanes)) / nx).astype(np.float32)
        v = ((b + rng.random(lanes)) / ny).astype(np.float32)
        ps = rect.sample_position(0.0, mi.Point2f(u, v))
        y = _vec(ps.p)
        d = y - x
        dist = np.linalg.norm(d, axis=-1)
        w = d / np.maximum(dist, 1e-12)[:, None]
        cos_l = np.maximum(0.0, -np.sum(_vec(ps.n) * w, -1))
        e = radiance * cos_l / (np.maximum(dist, 1e-12) ** 2 * np.array(ps.pdf))
        blocked = np.array(scene.ray_test(si.spawn_ray_to(ps.p), valid))
        dirs[:, j] = w
        weight[:, j] = e
        vis[:, j] = (np.sum(normal * w, -1) > 0) & ~blocked & np.array(valid)
    return {"light_dir": dirs, "light_weight": weight, "light_vis": vis}


def features(args) -> None:
    """Inference buffers at the evaluation camera for one state.

    K stratified sub-pixel samples, each with M stratified area-light samples
    (protocol rna_inference.light_sampling); light arrays are stored for hit
    samples only, in hit order.  --light-seed B draws independent light samples
    (same camera samples) for RNA's own seed-to-seed noise.
    """
    mi, dr = L.init_upstream()
    proto = json.loads(open(L.EXPERIMENT / args.protocol, encoding="utf-8").read())
    tr = proto["states"][args.state]
    res, k = proto["res"], proto["rna_inference"]["subpixel_grid"]
    ls = proto["rna_inference"]["light_sampling"]
    scene = mi.load_dict(T.scene_dict(res, tr, proto["lighting"]))
    ids = T.part_shape_ids(scene, tr)
    g = (np.arange(k) + 0.5) / k
    offsets = np.stack(np.meshgrid(g, g, indexing="xy"), -1).reshape(-1, 2)
    ray = camera_rays(scene.sensors()[0], res, offsets)
    si, feat = surface_features(scene, ray)
    seed = ls["seeds"][args.light_seed] + 7919 * list(proto["states"]).index(args.state)
    light = area_light_samples(scene, proto, si, feat["normal"], np.random.default_rng(seed))
    hit = feat["hit"]
    light = {key: val[hit] for key, val in light.items()}
    canonical = feat["position"].copy()
    part = np.full(len(canonical), -1, np.int8)
    for pi, (name, sid) in enumerate(ids.items()):
        sel = feat["shape_id"] == sid
        part[sel] = pi
        canonical[sel] -= np.asarray(tr.get(name, (0, 0, 0)), float)
    if np.any(hit & (part < 0)):
        raise RuntimeError("asset sample on an undeclared part")
    asset = [s for s in scene.shapes() if not s.is_emitter()]
    if len(asset) != 1:
        raise RuntimeError(f"expected one asset shape, found {len(asset)}")
    cb = asset[0].bbox()
    suffix = "" if args.light_seed == "A" else f"_{args.light_seed}"
    out = L.RESULTS / proto["rna_features_dir"] / f"{args.state}{suffix}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, res=res, samples=k * k, hit=hit, part=part, part_names=np.array(list(ids)),
                        position=feat["position"].astype(np.float32), canonical_position=canonical.astype(np.float32),
                        normal=feat["normal"].astype(np.float32), camera_dir=feat["camera_dir"].astype(np.float32),
                        light_dir=light["light_dir"].astype(np.float16), light_weight=light["light_weight"],
                        light_vis=light["light_vis"], light_seed=args.light_seed,
                        lighting=json.dumps(proto["lighting"], sort_keys=True),
                        light_sampling=json.dumps(ls, sort_keys=True), project_commit=L.git_head(L.ROOT) or "",
                        current_aabb_min=np.array(cb.min, np.float32), current_aabb_max=np.array(cb.max, np.float32))
    e = light["light_weight"].mean(1)
    print("wrote", L.rel(out), "hit fraction", float(hit.mean()), "lit fraction", float(light["light_vis"].mean()),
          "mean weight", float(e.mean()), "irradiance", proto["lighting"]["irradiance"])


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("h5")
    g.add_argument("--protocol", required=True)
    g.add_argument("--state", required=True)
    g.add_argument("--split", choices=("train", "val"), required=True)
    f = sub.add_parser("features")
    f.add_argument("--protocol", required=True)
    f.add_argument("--state", required=True)
    f.add_argument("--light-seed", choices=("A", "B"), default="A")
    args = ap.parse_args()
    generate_h5(args) if args.cmd == "h5" else features(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
