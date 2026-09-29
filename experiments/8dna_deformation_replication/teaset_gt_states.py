"""GT-only evaluation of teaset part configurations (no neural rendering).

For each state of a protocol file this renders two independent path-traced
references (seed sets A and B), primary-hit buffers through pixel centres, and
cosine-weighted mutual-visibility probes, then evaluates the physical-signal
gate: the GT change in the interaction ROI must exceed the declared multiple
of the A-vs-B repeat noise in that ROI for at least one non-control state.

ROIs are geometric, defined on the T0 image (protocol["rois"]), and carried
to every state by surface correspondence: a state pixel belongs to an ROI if
its primary hit, pulled back by its part's translation, lies within
`match_eps` of a T0 ROI hit on the same part, and it is paired with that T0
pixel for every change metric.  "Fixed" states move only `mover`; in them
stationary surfaces map to the same pixel.
  interaction : T0 pixels on `target` in every fixed state within `radius`
                of `mover` in the strongest approach state;
  tray        : T0 pixels on the tray in every fixed state;
  far         : T0 pixels on `far_part` in every fixed state, farther than
                `radius` from `mover` in every fixed state;
  mover       : T0 pixels on `mover`.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import ednalib as L
import teaset_parts as T


def primary_buffers(scene, res: int):
    """Pixel-centre primary hit: part name per pixel, world position and shading normal."""
    mi, dr = L.init_upstream()
    sensor = scene.sensors()[0]
    j, i = np.meshgrid(np.arange(res), np.arange(res), indexing="ij")
    film = np.stack([(i.ravel() + 0.5) / res, (j.ravel() + 0.5) / res], 0).astype(np.float32)
    ray, _ = sensor.sample_ray(0.0, 0.5, mi.Point2f(*film), mi.Point2f(0.5, 0.5))
    si = scene.ray_intersect(ray)
    sid = np.array(T.shape_ids(si)).reshape(res, res)
    p = np.stack([np.array(si.p.x), np.array(si.p.y), np.array(si.p.z)], -1).reshape(res, res, 3)
    n = np.stack([np.array(si.sh_frame.n.x), np.array(si.sh_frame.n.y), np.array(si.sh_frame.n.z)], -1).reshape(res, res, 3)
    d = np.stack([np.array(ray.d.x), np.array(ray.d.y), np.array(ray.d.z)], -1).reshape(res, res, 3)
    n = np.where((n * d).sum(-1, keepdims=True) > 0, -n, n)  # face the camera (two-sided BSDFs)
    return sid, p, n


def part_map(sid: np.ndarray, ids: dict[str, int]) -> np.ndarray:
    names = np.full(sid.shape, "", dtype=object)
    for part, v in ids.items():
        names[sid == v] = part
    return names


def surface_samples(part: str, translations: dict, n: int = 200_000) -> np.ndarray:
    v, f = T.part_mesh(part, translations)
    tri = v[f]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    rng = np.random.default_rng(0)
    k = rng.choice(len(f), n, p=area / area.sum())
    u, w = rng.random(n), rng.random(n)
    flip = u + w > 1
    u[flip], w[flip] = 1 - u[flip], 1 - w[flip]
    t = tri[k]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + w[:, None] * (t[:, 2] - t[:, 0])


def distance_to_part(points: np.ndarray, part: str, translations: dict) -> np.ndarray:
    from scipy.spatial import cKDTree

    tree = cKDTree(surface_samples(part, translations))
    return tree.query(points.reshape(-1, 3))[0].reshape(points.shape[:-1])


def pullback(partmap: np.ndarray, p: np.ndarray, translations: dict) -> np.ndarray:
    """Canonical position of every primary hit: subtract the hit part's translation."""
    q = p.copy()
    for part, t in translations.items():
        q[partmap == part] -= np.asarray(t, dtype=np.float64)
    return q


def correspond(buf_s, tr_s: dict, buf0, mask0: np.ndarray, eps: float):
    """Carry a T0 pixel set to state s.  Each state pixel is paired with the T0 pixel
    whose hit on the same part is nearest to its pulled-back hit; it joins the set if
    that T0 pixel is in `mask0` and lies within eps.  Returns (mask in state s, flat
    T0 pixel index of each masked pixel in raster order).  Unmoved surfaces map to
    themselves at distance 0, so the canonical state reproduces `mask0` exactly."""
    from scipy.spatial import cKDTree

    pm_s, p_s, _ = buf_s
    pm0, p0, _ = buf0
    q_s = pullback(pm_s, p_s, tr_s)
    mask = np.zeros(mask0.shape, bool)
    idx = np.full(mask0.size, -1, np.int64)
    for part in T.PARTS:
        all0 = pm0 == part
        sel_s = pm_s == part
        if not (all0 & mask0).any() or not sel_s.any():
            continue
        flat0 = np.flatnonzero(all0.reshape(-1))
        d, k = cKDTree(p0[all0]).query(q_s[sel_s])
        nearest = flat0[k]
        ok = (d < eps) & mask0.reshape(-1)[nearest]
        rows = np.flatnonzero(sel_s.reshape(-1))[ok]
        mask.reshape(-1)[rows] = True
        idx[rows] = nearest[ok]
    return mask, idx[mask.reshape(-1)]


def mutual_visibility(scene, ids, points, normals, part: str, rays: int = 256) -> float:
    """Cosine-weighted fraction of hemisphere rays from `points` whose first hit is `part`."""
    mi, dr = L.init_upstream()
    rng = np.random.default_rng(1)
    m = len(points)
    u = rng.random((m, rays, 2))
    r, phi = np.sqrt(u[..., 0]), 2 * np.pi * u[..., 1]
    local = np.stack([r * np.cos(phi), r * np.sin(phi), np.sqrt(1 - u[..., 0])], -1)
    nrm = normals[:, None, :]
    a = np.where(np.abs(nrm[..., 0:1]) > 0.9, [[[0.0, 1.0, 0.0]]], [[[1.0, 0.0, 0.0]]])
    s = np.cross(nrm, a)
    s /= np.linalg.norm(s, axis=-1, keepdims=True)
    t = np.cross(nrm, s)
    d = local[..., 0:1] * s + local[..., 1:2] * t + local[..., 2:3] * nrm
    o = points[:, None, :] + 1e-4 * nrm
    o, d = np.broadcast_to(o, d.shape).reshape(-1, 3), d.reshape(-1, 3)
    ray = mi.Ray3f(mi.Point3f(*o.T.astype(np.float32)), mi.Vector3f(*d.T.astype(np.float32)))
    sid = np.array(T.shape_ids(scene.ray_intersect(ray)))
    return float(np.mean(sid == ids[part]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    proto = json.loads(open(args.protocol, encoding="utf-8").read())
    out = L.RESULTS / args.out
    mi, dr = L.init_upstream()
    res, gt = proto["res"], proto["gt"]
    roi = proto["rois"]
    states = proto["states"]

    bufs, gts, ids_by_state = {}, {}, {}
    for name, tr in states.items():
        scene = mi.load_dict(T.scene_dict(res, tr))
        ids = T.part_shape_ids(scene, tr)
        ids_by_state[name] = ids
        sid, p, n = primary_buffers(scene, res)
        bufs[name] = (part_map(sid, ids), p, n)
        a, ta = L.render_reference(scene, gt["spp"], gt["chunk"], gt["seed_A"])
        b, _ = L.render_reference(scene, gt["spp"], gt["chunk"], gt["seed_B"])
        gts[name] = (a, b)
        L.save_exr(out / name / "gt_A.exr", a)
        L.save_exr(out / name / "gt_B.exr", b)
        L.save_png(out / name / "gt.png", L.to_u8(L.tonemap(0.5 * (a + b))))
        print(name, "GT seconds", round(ta, 1), ids)

    # --- ROIs: T0 pixel sets carried to every state by surface correspondence ---
    mover, target, far_part = roi["mover"], roi["target"], roi["far_part"]
    fixed = [s for s in states if not any(np.any(np.asarray(states[s].get(p, (0, 0, 0))) != 0)
                                          for p in T.PARTS if p != mover)]
    on = lambda part: np.all([bufs[s][0] == part for s in fixed], axis=0)
    strong = states[roi["strongest_state"]]
    d_strong = distance_to_part(bufs[roi["strongest_state"]][1], mover, strong)
    d_min = np.min([distance_to_part(bufs[s][1], mover, states[s]) for s in fixed], axis=0)
    base = {
        "interaction": on(target) & (d_strong < roi["radius"]),
        "tray": on("plate"),
        "far": on(far_part) & (d_min > roi["radius"]),
        "mover": bufs["T0"][0] == mover,
    }
    rois = {s: {k: correspond(bufs[s], states[s], bufs["T0"], m, roi["match_eps"]) for k, m in base.items()}
            for s in states}
    for s in states:
        np.savez_compressed(out / f"rois_{s}.npz", **{f"mask_{k}": v[0] for k, v in rois[s].items()},
                            **{f"idx0_{k}": v[1] for k, v in rois[s].items()})
    overlay = L.to_u8(L.tonemap(0.5 * sum(gts["T0"])))
    colours = {"interaction": (255, 60, 60), "tray": (60, 160, 255), "far": (80, 255, 80), "mover": (255, 200, 0)}
    for k, c in colours.items():
        overlay[base[k]] = (0.5 * overlay[base[k]] + 0.5 * np.array(c)).astype(np.uint8)
    L.save_png(out / "rois_on_T0.png", overlay)

    # --- measurements (all changes are paired through the correspondence) ---
    ref0_a = gts["T0"][0].reshape(-1, 3)
    ref0 = (0.5 * (gts["T0"][0] + gts["T0"][1])).reshape(-1, 3)
    record = {"protocol": proto, "part_shape_ids": ids_by_state, "fixed_pixel_states": fixed,
              "roi_pixels_T0": {k: int(m.sum()) for k, m in base.items()}, "states": {}}
    for name, tr in states.items():
        a, b = gts[name]
        scene = mi.load_dict(T.scene_dict(res, tr))
        ids = T.part_shape_ids(scene, tr)
        m_int = rois[name]["interaction"][0]
        pts, nrm = bufs[name][1][m_int], bufs[name][2][m_int]
        rng = np.random.default_rng(2)
        pick = rng.choice(len(pts), min(2000, len(pts)), replace=False) if len(pts) else []
        entry = {"translations": tr, "role": proto["roles"][name],
                 "primary_part_changed_fraction_image_space": float(np.mean(bufs[name][0] != bufs["T0"][0])),
                 "mutual_visibility_interaction_to_mover": mutual_visibility(scene, ids, pts[pick], nrm[pick], mover) if len(pick) else None,
                 "regions": {}}
        for k, (m, idx0) in rois[name].items():
            sel = m.reshape(-1)
            av, bv, g = a.reshape(-1, 3)[sel], b.reshape(-1, 3)[sel], (0.5 * (a + b)).reshape(-1, 3)[sel]
            chg = np.abs(L.tonemap(av) - L.tonemap(ref0_a[idx0])).mean(-1)
            noise = np.abs(L.tonemap(av) - L.tonemap(bv)).mean(-1)
            entry["regions"][k] = {
                "pixels": int(sel.sum()), "matched_fraction_of_T0": float(sel.sum() / max(base[k].sum(), 1)),
                "gt_change_display_mae_A": float(chg.mean()) if len(chg) else None,
                "gt_change_display_p95_A": float(np.percentile(chg, 95)) if len(chg) else None,
                "gt_repeat_noise_display_mae": float(noise.mean()) if len(noise) else None,
                "gt_change_vs_T0_metrics": L.pixel_metrics(g, ref0[idx0]),
            }
        diff_img = np.abs(L.tonemap(a) - L.tonemap(gts["T0"][0])).mean(-1)
        entry["regions"]["full_image_space"] = {"gt_change_display_mae_A": float(diff_img.mean())}
        r = entry["regions"]["interaction"]
        entry["interaction_signal_to_noise"] = r["gt_change_display_mae_A"] / max(r["gt_repeat_noise_display_mae"], 1e-12)
        record["states"][name] = entry
        L.save_png(out / name / "gt_change_vs_T0_image_space.png", L.error_map(diff_img, proto["error_scale"]))

    null = [s for s in states if proto["roles"][s] == "low-interaction control"]
    ref_chg = record["states"][roi["strongest_state"]]["regions"]["interaction"]["gt_change_display_mae_A"]
    record["null_control"] = {
        s: {"interaction_change_ratio_to_strongest": record["states"][s]["regions"]["interaction"]["gt_change_display_mae_A"] / ref_chg,
            "qualifies": record["states"][s]["regions"]["interaction"]["gt_change_display_mae_A"] <= proto["null_max_ratio"] * ref_chg}
        for s in null}
    k = proto["gate"]["noise_multiple"]
    tested = [s for s in states if proto["roles"][s] not in ("canonical", "low-interaction control")]
    passing = [s for s in tested if record["states"][s]["interaction_signal_to_noise"] > k]
    record["physical_signal_gate"] = {"noise_multiple": k, "tested_states": tested, "passing_states": passing,
                                      "verdict": "PASS" if passing else "PHYSICAL EFFECT TOO WEAK"}
    L.write_json(out / "gt_states.json", record)
    print(json.dumps({s: round(record["states"][s]["interaction_signal_to_noise"], 2) for s in states}),
          record["physical_signal_gate"]["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
