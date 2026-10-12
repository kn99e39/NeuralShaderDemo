"""Monte-Carlo radiance targets at surface queries, from the validated worklog-24 estimator.

A query is (part, canonical position x, oriented geometric normal n, side s, outgoing direction w)
with dot(w, s*n) > 0.  It is turned into a ray that starts D0 outside the surface along w and
points back at the query point; transport_decomposition.decompose() (a re-implementation of
Mitsuba 3.5.1's prb/path estimator, worklog 24, validated against Mitsuba C++ path) then returns
the radiance arriving along the ray split by path class.  The first hit must be the query point
on the query part (checked; rejects are reported).

  direct    = class v = 1 (light reaching x directly, current visibility)
  indirect  = classes with v >= 2 (the transport scope the units own)
"""

from __future__ import annotations

import math

import sor_common as S

D0 = 2e-3            # ray origin offset along w
GRAZING_MIN = 0.05   # min dot(w, side*n): queries more grazing than ~87 deg are not drawn
HIT_TOL = 2e-4       # |hit - x| tolerance for an accepted query


def scene_for(state: str, res: int = 64):
    mi, dr = S.init()
    import teaset_parts as T

    tr = S.states()[state]
    scene = mi.load_dict(T.scene_dict(res, tr))
    ids = T.part_shape_ids(scene, tr)
    return scene, ids, tr


def part_areas():
    return {p: float(S.part_geometry(p)[3].sum()) for p in S.PARTS}


def sample_queries(n: int, rng, translations: dict, parts=None, unit_area_weights=None):
    """Area-uniform surface points (optionally restricted to `parts`), random side, uniform
    outgoing direction in that side's hemisphere (rejecting the GRAZING_MIN band)."""
    import numpy as np

    parts = parts or S.PARTS
    areas = np.array([part_areas()[p] for p in parts])
    counts = rng.multinomial(n, areas / areas.sum())
    out = {k: [] for k in ("part", "x_local", "n", "side", "w")}
    for p, c in zip(parts, counts):
        if c == 0:
            continue
        x, nrm, _ = S.sample_surface(p, c, rng)
        side = np.where(rng.random(c) < 0.5, 1.0, -1.0)
        # uniform hemisphere about side*n, cos >= GRAZING_MIN
        u1, u2 = rng.random(c), rng.random(c)
        z = GRAZING_MIN + (1 - GRAZING_MIN) * u1
        r = np.sqrt(np.clip(1 - z * z, 0, None))
        phi = 2 * math.pi * u2
        loc = np.stack([r * np.cos(phi), r * np.sin(phi), z], -1)
        nn = nrm * side[:, None]
        a = np.where(np.abs(nn[:, :1]) > 0.9, [[0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0]])
        t = np.cross(nn, a); t /= np.linalg.norm(t, axis=1, keepdims=True)
        b = np.cross(nn, t)
        w = loc[:, :1] * t + loc[:, 1:2] * b + loc[:, 2:3] * nn
        out["part"].append(np.full(c, S.PARTS.index(p))); out["x_local"].append(x); out["n"].append(nrm)
        out["side"].append(side); out["w"].append(w)
    q = {k: np.concatenate(v) for k, v in out.items()}
    q["x_world"] = q["x_local"] + np.stack([np.asarray(translations.get(S.PARTS[i], (0, 0, 0)), float)
                                            for i in q["part"]])
    return q


def directions(x, n, part, gen, translations: dict):
    """Torch: random side and uniform outgoing direction in that side's hemisphere (cos >= GRAZING_MIN);
    returns a query dict of torch tensors (canonical x_local, world x_world)."""
    import torch

    N = len(x)
    side = torch.where(torch.rand(N, device=x.device, generator=gen) < 0.5, 1.0, -1.0)
    z = GRAZING_MIN + (1 - GRAZING_MIN) * torch.rand(N, device=x.device, generator=gen)
    phi = 2 * math.pi * torch.rand(N, device=x.device, generator=gen)
    r = (1 - z * z).clamp(min=0).sqrt()
    nn = n * side[:, None]
    t, b = S.frames(nn)
    w = (r * torch.cos(phi))[:, None] * t + (r * torch.sin(phi))[:, None] * b + z[:, None] * nn
    tr = torch.tensor([list(translations.get(p, (0.0, 0.0, 0.0))) for p in S.PARTS], dtype=torch.float32, device=x.device)
    return {"part": part, "x_local": x, "n": n, "side": side, "w": w, "x_world": x + tr[part]}


def trace_t(scene, ids: dict, q: dict, seed: int, spp: int = 1):
    """Torch-in/torch-out variant of trace() for one chunk (queries already on the GPU)."""
    mi, dr = S.init()
    import numpy as np
    import torch
    import teaset_parts as T
    from transport_decomposition import CLASSES, decompose

    n = len(q["part"])
    o = (q["x_world"] + D0 * q["w"]).contiguous()
    d = (-q["w"]).contiguous()
    ray = mi.Ray3f(mi.Point3f(o), mi.Vector3f(d))
    si = scene.ray_intersect(ray)
    hit = si.p.torch()
    sid = T.shape_ids(si).torch().long()
    sid_of = torch.tensor([ids[p] for p in S.PARTS], device=o.device)
    ok = si.is_valid().torch().bool() & (sid == sid_of[q["part"]]) & ((hit - q["x_world"]).norm(dim=-1) < HIT_TOL)
    acc_d = torch.zeros(n, 3, device=o.device); acc_i = torch.zeros(n, 3, device=o.device)
    for k in range(spp):
        sampler = mi.load_dict({"type": "independent"})
        sampler.seed((seed * 31 + k) & 0xFFFFFFFF, n)
        per, total, _ = decompose(scene, mi.Ray3f(ray), sampler, ids)
        acc_d += torch.from_numpy(per["direct"]).to(o.device)
        acc_i += torch.from_numpy(sum(per[c] for c in CLASSES[2:])).to(o.device)
    return acc_d / spp, acc_i / spp, ok


def trace(scene, ids: dict, q: dict, seed: int, chunk: int = 2_000_000, spp: int = 1):
    """MC estimate per query: (direct (N,3), indirect (N,3), accepted (N,) bool)."""
    mi, dr = S.init()
    import numpy as np
    import teaset_parts as T
    from transport_decomposition import decompose

    N = len(q["part"])
    direct = np.zeros((N, 3)); indirect = np.zeros((N, 3)); ok = np.zeros(N, bool)
    cls_ind = [2, 3, 4, 5, 6, 7, 8]
    sid_of = np.array([ids[p] for p in S.PARTS])
    for s0 in range(0, N, chunk):
        sl = slice(s0, min(N, s0 + chunk))
        n = sl.stop - sl.start
        o = q["x_world"][sl] + D0 * q["w"][sl]
        d = -q["w"][sl]
        acc_d = np.zeros((n, 3)); acc_i = np.zeros((n, 3))
        for k in range(spp):
            ray = mi.Ray3f(mi.Point3f(*o.T.astype(np.float32)), mi.Vector3f(*d.T.astype(np.float32)))
            if k == 0:
                si = scene.ray_intersect(ray)
                hit = np.stack([np.array(si.p.x), np.array(si.p.y), np.array(si.p.z)], -1)
                sid = np.array(T.shape_ids(si))
                ok[sl] = (np.array(si.is_valid()) & (sid == sid_of[q["part"][sl]])
                          & (np.linalg.norm(hit - q["x_world"][sl], axis=1) < HIT_TOL))
            sampler = mi.load_dict({"type": "independent"})
            sampler.seed(((seed * 1_000_003 + s0) * 31 + k) & 0xFFFFFFFF, n)
            per, total, _ = decompose(scene, ray, sampler, ids)
            acc_d += per["direct"]
            acc_i += sum(per[c] for c in [list(per)[i] for i in cls_ind])
        direct[sl] = acc_d / spp
        indirect[sl] = acc_i / spp
    return direct, indirect, ok


def camera_queries(scene, ids: dict, translations: dict, res: int, sub: int, seed: int):
    """Camera first hits on a sub x sub jittered lattice per pixel: part, canonical position,
    geometric normal, outgoing direction; pixel index; valid (asset hit)."""
    mi, dr = S.init()
    import numpy as np
    import teaset_parts as T

    sensor = scene.sensors()[0]
    rng = np.random.default_rng(seed)
    pix = np.arange(res * res)
    xy = np.stack([pix % res, pix // res], -1).astype(np.float64)
    off = (np.stack(np.meshgrid(np.arange(sub), np.arange(sub), indexing="ij"), -1).reshape(-1, 2) + rng.random((sub * sub, 2))) / sub
    film = (np.repeat(xy, sub * sub, 0) + np.tile(off, (res * res, 1))) / res
    ray, _ = sensor.sample_ray(0.0, 0.5, mi.Point2f(*film.T.astype(np.float32)), mi.Point2f(0.5, 0.5))
    si = scene.ray_intersect(ray)
    valid = np.array(si.is_valid())
    sid = np.array(T.shape_ids(si))
    part = -np.ones(len(sid), np.int64)
    for i, p in enumerate(S.PARTS):
        part[sid == ids[p]] = i
    valid &= part >= 0
    p_w = np.stack([np.array(si.p.x), np.array(si.p.y), np.array(si.p.z)], -1)
    n_g = np.stack([np.array(si.n.x), np.array(si.n.y), np.array(si.n.z)], -1)
    d = np.stack([np.array(ray.d.x), np.array(ray.d.y), np.array(ray.d.z)], -1)
    tr = np.stack([np.asarray(translations.get(S.PARTS[i], (0, 0, 0)), float) if i >= 0 else np.zeros(3) for i in part])
    return {"part": part, "x_local": p_w - tr, "n": n_g, "w": -d, "pix": np.repeat(pix, sub * sub), "valid": valid}
