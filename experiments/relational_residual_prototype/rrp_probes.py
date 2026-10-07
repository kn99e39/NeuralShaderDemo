"""Current-geometry probe state for the ROI queries of every teaset state (Windows, Mitsuba).

    windows/run.ps1 ../relational_residual_prototype/rrp_probes.py --out <run dir> [--allow-dirty]

For each state of the common-light protocol and each ROI query (sub-pixel
sample of the historical RNA feature buffers, `rna_teaset/features/<state>.npz`):

  * local frame: the buffer's shading normal n (already flipped to the camera
    side) and the deterministic tangents of rrp_common.onb;
  * K fixed probe directions (rrp_common.probe_dirs) cast from x + eps*n into the
    state's CURRENT geometry-only scene (teaset_parts.scene_dict(..., lighting=None):
    the four parts, no emitter);
  * per probe: hit, distance, remote normal (faced to the query) in the query
    frame, the remote part and its canonical pull-back (hit - t_part, exactly
    the correspondence of teaset_parts / rna_bridge), and the remote point's
    direct visibility toward the protocol light centre via the existing
    rna_bridge.visibility.

Also writes, per state, the ROI pixels' reference (mean of the locked seed-A/B
renders) and the historical frozen RNA pixels, and the stable-query flags.
Nothing here reads a reference image into a probe descriptor.

Timing (warm, GPU-synchronised): probe ray casting and remote shadow rays for all
Q x K probes, kept separate from everything else.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "8dna_deformation_replication"))
sys.path.insert(0, str(HERE.parents[0] / "neural_recompute_cost"))

import numpy as np  # noqa: E402

import rrp_common as C  # noqa: E402


def cast_probes(mi, dr, B, T, scene, part_ids: dict, translations: dict, x: np.ndarray, n: np.ndarray,
                dirs_local: np.ndarray, to_light: np.ndarray, eps: float, chunk: int) -> dict:
    """Probe the current scene from Q query points (x, n) along K local directions."""
    s, t = C.onb(n)
    q, k = len(x), len(dirs_local)
    d_world = C.to_world(dirs_local, s, n, t).reshape(-1, 3)
    o = np.repeat(x + eps * n, k, 0)
    names = list(C.protocol()["parts_order"])
    sid_to_part = {sid: names.index(p) for p, sid in part_ids.items()}
    out = {"hit": np.zeros(q * k, bool), "dist": np.zeros(q * k, np.float32), "remote_normal_local": np.zeros((q * k, 3), np.float32),
           "remote_facing": np.zeros(q * k, np.float32), "remote_part": np.full(q * k, -1, np.int8),
           "remote_canonical": np.zeros((q * k, 3), np.float32), "remote_light_vis": np.zeros(q * k, bool),
           "remote_light_cos": np.zeros(q * k, np.float32)}
    for a in range(0, q * k, chunk):
        b = min(a + chunk, q * k)
        ray = mi.Ray3f(mi.Point3f(*o[a:b].T.astype(np.float32)), mi.Vector3f(*d_world[a:b].T.astype(np.float32)))
        si = scene.ray_intersect(ray)
        hit = np.array(si.is_valid())
        p = B._vec(si.p)
        nr = B._vec(si.sh_frame.n)
        dd = d_world[a:b]
        flip = np.sum(nr * dd, -1) > 0          # face the remote normal toward the query (as surface_features)
        nr[flip] *= -1
        sid = np.array(T.shape_ids(si))
        part = np.array([sid_to_part.get(int(v), -1) for v in sid], np.int8)
        if np.any(hit & (part < 0)):
            raise RuntimeError("probe hit an undeclared shape")
        tl = np.broadcast_to(to_light, (b - a, 3)).copy()
        vis = B.visibility(scene, si, nr, tl)
        canon = p.copy()
        for pi, name in enumerate(names):
            sel = part == pi
            canon[sel] -= np.asarray(translations.get(name, (0.0, 0.0, 0.0)), float)
        qi = np.arange(a, b) // k
        out["hit"][a:b] = hit
        out["dist"][a:b] = np.where(hit, np.array(si.t), 0.0)
        out["remote_normal_local"][a:b] = np.where(hit[:, None], C.to_local(nr, s[qi], n[qi], t[qi]), 0.0)
        out["remote_facing"][a:b] = np.where(hit, -np.sum(dd * nr, -1), 0.0)
        out["remote_part"][a:b] = np.where(hit, part, -1)
        out["remote_canonical"][a:b] = np.where(hit[:, None], canon, 0.0)
        out["remote_light_vis"][a:b] = hit & vis
        out["remote_light_cos"][a:b] = np.where(hit, np.maximum(0.0, np.sum(nr * to_light, -1)), 0.0)
    return {key: v.reshape(q, k, *v.shape[1:]) for key, v in out.items()}


def time_probes(mi, dr, B, T, scene, x, n, dirs_local, to_light, eps, chunk, repeats: int) -> dict:
    """GPU time of the probe ray casts + remote shadow rays alone (no host feature assembly)."""
    import torch

    s, t = C.onb(n)
    k = len(dirs_local)
    d_world = C.to_world(dirs_local, s, n, t).reshape(-1, 3).astype(np.float32)
    o = np.repeat(x + eps * n, k, 0).astype(np.float32)
    rays = [(mi.Point3f(*o[a:a + chunk].T), mi.Vector3f(*d_world[a:a + chunk].T)) for a in range(0, len(o), chunk)]
    tl = mi.Vector3f(*[float(v) for v in to_light])

    def once():
        cnt = 0
        for po, dv in rays:
            si = scene.ray_intersect(mi.Ray3f(po, dv))
            blocked = scene.ray_test(si.spawn_ray(tl), si.is_valid())
            dr.eval(si.t, blocked)
            cnt += dr.width(si.t)
        dr.sync_thread()
        torch.cuda.synchronize()
        return cnt

    once()  # warm-up (kernel compilation)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        nrays = once()
        times.append(time.perf_counter() - t0)
    return {"probe_rays": int(nrays), "shadow_rays_upper_bound": int(nrays), "seconds": times,
            "seconds_median": float(np.median(times)), "chunk": chunk}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    proto = C.protocol()
    C.check_feature_names()
    import ednalib as L
    import nrc_records as R

    git = R.git_state(L.ROOT)
    R.require_clean(git, args.allow_dirty)
    mi, dr = L.init_upstream()
    import rna_bridge as B
    import teaset_parts as T

    cb = json.loads((L.EXPERIMENT / proto["base_protocol"]).read_text(encoding="utf-8"))
    states = cb["states"]
    names = [str(v) for v in np.load(L.RESULTS / cb["rna_features_dir"] / "T0.npz")["part_names"]]
    if names != proto["parts_order"]:
        raise SystemExit(f"feature-buffer part order {names} != protocol {proto['parts_order']}")
    qpart = names.index(C.QUERY_PART)
    pr = proto["probes"]
    dirs = C.probe_dirs(pr["K"])
    to_light = np.asarray(cb["lighting"]["to_light"], float)
    to_light /= np.linalg.norm(to_light)
    f0 = dict(np.load(L.RESULTS / cb["rna_features_dir"] / "T0.npz"))
    rois = {s: np.load(L.RESULTS / cb["gt_output"] / f"rois_{s}.npz") for s in states}
    pix0 = np.flatnonzero(rois["T0"]["mask_interaction"].reshape(-1))
    k = int(f0["samples"])
    rec = {"environment": L.environment_record(), "git": git, "protocol": proto, "states": {}}
    fdir = L.RESULTS / cb["frozen_output"]
    rdir = L.RESULTS / cb["rna_output"]
    for name in proto["evaluate_states"]:
        buf = name  # T0_B: the T0 geometry with independent light samples (RNA's own noise floor)
        st = "T0" if name == "T0_B" else name
        tr = states[st]
        f = f0 if name == "T0" else dict(np.load(L.RESULTS / cb["rna_features_dir"] / f"{buf}.npz"))
        roi = rois[st]
        pix = np.flatnonzero(roi["mask_interaction"].reshape(-1))
        idx0 = roi["idx0_interaction"]
        identity = np.array_equal(pix, idx0)
        if identity:
            # stationary tea pot (T0, T1b, T2, T3): every sample of a stable pixel has the T0
            # sample's part, canonical position, normal and view exactly
            stable = C.stable_pixels(f0, f, pix, idx0, k, qpart, require_identity=True)
        else:
            # T1 moves the whole asset: pixels map to other T0 pixels (worklog-22 pairing), so
            # only the part condition applies
            stable = C.stable_pixels(f, f, pix, pix, k, qpart, require_identity=False)
        lanes = C.roi_lanes(pix, k)
        sl = lanes[np.repeat(stable, k)]
        hidx = C.hit_order_index(f["hit"], sl)
        x, n, v = f["position"][sl].astype(np.float64), f["normal"][sl].astype(np.float64), f["camera_dir"][sl].astype(np.float64)
        s_, t_ = C.onb(n)
        ld = f["light_dir"][hidx].astype(np.float64)
        lw = f["light_weight"][hidx].astype(np.float64)
        lv = f["light_vis"][hidx].astype(np.float64)
        lmean = ld.mean(1)
        lmean /= np.linalg.norm(lmean, axis=-1, keepdims=True)
        scene = mi.load_dict(T.scene_dict(64, tr, None))  # geometry and materials only: no emitter in the probe scene
        pids = T.part_shape_ids(scene, tr)
        t0 = time.perf_counter()
        probes = cast_probes(mi, dr, B, T, scene, pids, tr, x, n, dirs, to_light, pr["origin_offset"], pr["chunk"])
        build_s = time.perf_counter() - t0
        timing = time_probes(mi, dr, B, T, scene, x, n, dirs, to_light, pr["origin_offset"], pr["chunk"], pr["timing_repeats"]) \
            if name in proto["timing_states"] else None
        # reference and historical frozen pixels (never part of a descriptor)
        gt = 0.5 * (L.load_exr(fdir / st / "gt_A.exr") + L.load_exr(fdir / st / "gt_B.exr"))
        frozen_name = "T0_canonical_B" if name == "T0_B" else f"{st}_canonical"
        frozen = np.load(rdir / f"{frozen_name}.npy").astype(np.float32)
        payload = {
            "pixels": pix, "stable": stable, "lanes": sl, "samples_per_pixel": k,
            "query_canonical": f["canonical_position"][sl], "query_normal": n.astype(np.float32),
            "query_camera_dir": v.astype(np.float32),  # world; RNA's own input for the frozen base prediction
            "query_view_local": C.to_local(v, s_, n, t_).astype(np.float32),
            "query_light_local": C.to_local(lmean, s_, n, t_).astype(np.float32),
            "direct_vis_fraction": lv.mean(1).astype(np.float32),
            "direct_irradiance": ((lv * lw).mean(1) / cb["rna_inference"]["training_light_intensity"]).astype(np.float32),
            "light_dir": ld.astype(np.float16), "light_weight": lw.astype(np.float32), "light_vis": lv.astype(bool),
            "probe_dirs_local": dirs.astype(np.float32), **probes,
            "gt_pixels": gt.reshape(-1, 3)[pix].astype(np.float32), "frozen_pixels": frozen.reshape(-1, 3)[pix],
        }
        path = out / "probes" / f"{name}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **payload)
        hp = probes["hit"]
        mover = names.index(proto["moved_part_for_reporting_only"])
        rec["states"][name] = {
            "file": L.rel(path), "roi_pixels": int(len(pix)), "stable_pixels": int(stable.sum()),
            "queries": int(len(sl)), "probes": int(hp.size), "probe_hit_fraction": float(hp.mean()),
            "probe_hits_by_part": {nm: int((probes["remote_part"] == i).sum()) for i, nm in enumerate(names)},
            "probe_hits_on_moved_part_fraction_reporting_only": float((probes["remote_part"] == mover).mean()),
            "build_wall_s_incl_host_assembly": build_s, "probe_gpu_timing": timing,
            "features_buffer": L.rel(L.RESULTS / cb["rna_features_dir"] / f"{buf}.npz"),
            "frozen_render": L.rel(rdir / f"{frozen_name}.npy")}
        print(name, json.dumps({kk: vv for kk, vv in rec["states"][name].items() if kk in ("stable_pixels", "queries", "probe_hit_fraction",
                                                                                          "probe_hits_by_part")}), flush=True)
    L.write_json(out / "probes" / "probes.json", rec)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
