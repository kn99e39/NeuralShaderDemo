"""Frozen-RNA inputs for the relational-residual prototype (WSL, RNA venv).

    cd external/relightable-neural-assets
    .venv/bin/python ../../experiments/relational_residual_prototype/wsl/rrp_features.py --run <run dir>

For every state written by rrp_probes.py:

  * L_base per query sample: the unchanged rna_infer.py canonical-mode loop
    (load_from_checkpoint strict=False, blur sigma 1, branch by per-light-sample
    visibility, clamp >= 0, times weight / training light intensity, mean over the
    16 light samples).  The per-pixel mean of these samples must reproduce the
    historical frozen render at every stable ROI pixel.
  * the query's persistent feature: RNA's own triplane lookup at the canonical
    point, captured by a forward hook on model.triplane_grid (no RNA code changed);
  * each remote probe hit's persistent feature: the same triplane at the hit's
    canonical pull-back (persistent, surface-attached; no per-state embedding).

Writes dataset/<state>.npz with the assembled local inputs (rrp_common.LOCAL_FEATURES)
and probe descriptors (rrp_common.PROBE_FEATURES), raw (unnormalised).
The frozen RNA's parameters are digested before and after; they must not change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import rrp_common as C  # noqa: E402


def digest_module(module) -> str:
    import torch

    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().to("cpu", torch.float32).contiguous().numpy().tobytes())
    return h.hexdigest()


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = Path(args.run)
    proto = C.protocol()
    cb = json.loads((C.EXP8 / proto["base_protocol"]).read_text(encoding="utf-8"))
    sys.path.insert(0, os.getcwd())  # RNA root
    import h5py
    import torch as th

    from rna import interfaces
    from utils import ops

    dev = th.device("cuda")
    ck = C.ROOT / proto["frozen_rna"]["checkpoint"]
    if sha256(ck) != proto["frozen_rna"]["sha256"]:
        raise SystemExit("frozen RNA checkpoint hash differs from the protocol")
    module = interfaces.NeuralSurfaceTriplaneModule.load_from_checkpoint(str(ck), strict=False, map_location=dev)
    module.model.initial_sigma = 1
    module.model.iterations_to_sigma_1 = 1
    module.to(dev).eval().freeze()
    digest_before = digest_module(module)
    captured = {}
    def hook(m, i, o):
        captured["feat"] = o
        captured["sigma"] = i[1]  # the blur sigma RNA's forward passes to its triplane

    module.model.triplane_grid.register_forward_hook(hook)
    with h5py.File(C.RES8 / cb["rna_dataset_dir"] / "teaset_T0_train.h5", "r") as f:
        lo = th.tensor(np.array(f.attrs["aabb_min"]), dtype=th.float32, device=dev).unsqueeze(0)
        hi = th.tensor(np.array(f.attrs["aabb_max"]), dtype=th.float32, device=dev).unsqueeze(0)
    tt = lambda a: th.tensor(np.ascontiguousarray(a), dtype=th.float32, device=dev)
    E = float(cb["rna_inference"]["training_light_intensity"])
    batch = proto["features"]["batch"]

    def triplane(pos):
        """Frozen triplane feature (n, 8) at canonical positions (n, 3); directions do not enter it."""
        outs = []
        up = th.tensor([0.0, 1.0, 0.0], device=dev)
        with th.no_grad():
            for s in range(0, len(pos), batch):
                p = ops.normalize_positions(tt(pos[s:s + batch]).unsqueeze(0), lo, hi)
                d = up.expand(p.shape[1], 3).unsqueeze(0)
                module(p, d, d, d)
                outs.append(captured["feat"].squeeze(0).transpose(0, 1).float().cpu())
        return th.cat(outs).numpy() if outs else np.zeros((0, 8), np.float32)

    def base(z):
        """rna_infer.py canonical-mode radiance per query sample (n, 3)."""
        pos, cam, nrm = z["query_canonical"], z["query_camera_dir"], z["query_normal"]
        ld, lw, lv = z["light_dir"].astype(np.float32), z["light_weight"], z["light_vis"]
        m = ld.shape[1]
        step = max(1, (1 << 20) // m)
        rad = np.zeros((len(pos), 3), np.float32)
        with th.no_grad():
            for s in range(0, len(pos), step):
                e = min(s + step, len(pos))
                rep = lambda a: tt(a[s:e]).unsqueeze(1).expand(-1, m, -1).reshape(-1, a.shape[-1])
                p = ops.normalize_positions(rep(pos).unsqueeze(0), lo, hi)
                pred = module(p, rep(cam).unsqueeze(0), tt(ld[s:e]).reshape(1, -1, 3), rep(nrm).unsqueeze(0))
                vis = tt(lv[s:e].astype(np.float32)).reshape(1, -1, 1)
                pred = th.where(vis > 0.0, pred[..., :3], pred[..., 3:]).clamp(min=0.0)
                w = tt(lw[s:e]).reshape(1, -1, 1) / E
                rad[s:e] = (pred * w).reshape(e - s, m, 3).mean(1).cpu().numpy()
        return rad

    rec = {"checkpoint": str(ck), "checkpoint_sha256": proto["frozen_rna"]["sha256"], "rna_digest_before": digest_before,
           "states": {}, "torch": th.__version__, "gpu": th.cuda.get_device_name(0)}
    for name in proto["evaluate_states"]:
        z = dict(np.load(run / "probes" / f"{name}.npz"))
        k = int(z["samples_per_pixel"])
        lb = base(z)
        stable_pix = np.flatnonzero(z["stable"])
        base_px = lb.reshape(-1, k, 3).mean(1)
        frozen_px = z["frozen_pixels"][stable_pix]
        diff = float(np.abs(base_px - frozen_px).max())
        qf = triplane(z["query_canonical"])
        hit = z["hit"]
        rf = np.zeros(hit.shape + (8,), np.float32)
        th.cuda.synchronize()
        t0 = time.perf_counter()
        rf[hit] = triplane(z["remote_canonical"][hit])
        th.cuda.synchronize()
        t_remote = time.perf_counter() - t0
        # the same lookup with the positions already on the GPU and the features kept there
        # (what a resident state update would pay); median of repeats after one warm-up
        gpos = ops.normalize_positions(tt(z["remote_canonical"][hit]).unsqueeze(0), lo, hi)
        up = th.tensor([0.0, 1.0, 0.0], device=dev).expand(gpos.shape[1], 3).unsqueeze(0)
        gt_times, tp_times = [], []
        with th.no_grad():
            module(gpos[:, :64], up[:, :64], up[:, :64], up[:, :64])  # captures the sigma RNA passes to its triplane
            sigma = captured["sigma"]
            ref = None
            for rep in range(proto["features"]["gpu_timing_repeats"] + 1):
                th.cuda.synchronize()
                t0 = time.perf_counter()
                for s in range(0, gpos.shape[1], batch):
                    module(gpos[:, s:s + batch], up[:, s:s + batch], up[:, s:s + batch], up[:, s:s + batch])
                th.cuda.synchronize()
                t1 = time.perf_counter()
                outs = [module.model.triplane_grid(gpos[:, s:s + batch], sigma) for s in range(0, gpos.shape[1], batch)]
                th.cuda.synchronize()
                t2 = time.perf_counter()
                if rep:
                    gt_times.append(t1 - t0)
                    tp_times.append(t2 - t1)
                else:
                    ref = th.cat([o.squeeze(0) for o in outs], -1).transpose(0, 1).float().cpu().numpy()
        t_remote_gpu = float(np.median(gt_times))
        t_triplane_only = float(np.median(tp_times))
        triplane_only_matches = bool(np.array_equal(ref, rf[hit]))
        local = np.concatenate([qf, z["query_view_local"], z["query_light_local"], z["direct_vis_fraction"][:, None],
                                z["direct_irradiance"][:, None], np.log1p(lb)], -1).astype(np.float32)
        hf = hit.astype(np.float32)[..., None]
        dist = z["dist"][..., None]
        q, kk = hit.shape
        probes = np.concatenate([hf, np.exp(-dist / 0.1) * hf, np.exp(-dist / 1.0) * hf, z["remote_normal_local"],
                                 z["remote_facing"][..., None], rf, z["remote_light_vis"].astype(np.float32)[..., None],
                                 z["remote_light_cos"][..., None], np.broadcast_to(z["probe_dirs_local"], (q, kk, 3))],
                                -1).astype(np.float32)
        assert local.shape[-1] == len(C.LOCAL_FEATURES) and probes.shape[-1] == len(C.PROBE_FEATURES)
        out = run / "dataset" / f"{name}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, local=local, probes=probes, base_samples=lb, base_pixels=base_px, frozen_pixels=frozen_px,
                 gt_pixels=z["gt_pixels"][stable_pix], roi_pixels=z["pixels"], stable_pixels=stable_pix,
                 query_to_pixel=np.repeat(np.arange(len(stable_pix)), k), samples_per_pixel=k)
        rec["states"][name] = {"queries": int(q), "stable_pixels": int(len(stable_pix)),
                               "base_vs_historical_frozen_max_abs": diff,
                               "base_vs_historical_frozen_max_rel": float((np.abs(base_px - frozen_px) / np.maximum(np.abs(frozen_px), 1e-3)).max()),
                               "remote_hits": int(hit.sum()), "remote_feature_lookup_s": t_remote,
                               "remote_feature_lookup_gpu_resident_s": t_remote_gpu,
                               "remote_triplane_only_gpu_s": t_triplane_only,
                               "remote_triplane_only_equals_hooked_features": triplane_only_matches,
                               "remote_feature_lookup_note": "gpu_resident runs RNA's full forward (triplane + MLP) to reach the hooked output; triplane_only calls RNA's triplane module directly with the sigma its forward uses, and must give identical features"}
        print(name, rec["states"][name], flush=True)
    rec["rna_digest_after"] = digest_module(module)
    rec["rna_unchanged"] = rec["rna_digest_after"] == digest_before
    (run / "dataset" / "features.json").write_text(json.dumps(rec, indent=1))
    bad = [n for n, s in rec["states"].items() if s["base_vs_historical_frozen_max_rel"] > proto["features"]["base_reproduction_rel_tol"]]
    if bad or not rec["rna_unchanged"]:
        print("FAIL: base does not reproduce the historical frozen render" if bad else "FAIL: RNA changed", bad, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
