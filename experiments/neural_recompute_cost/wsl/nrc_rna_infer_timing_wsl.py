"""Steady-state RNA inference timing on one feature buffer (WSL, RNA venv).

    cd external/relightable-neural-assets
    .venv/bin/python ../../experiments/neural_recompute_cost/wsl/nrc_rna_infer_timing_wsl.py \
        --checkpoint <ckpt> --features <T3.npz> --reference <rna_infer output .npy> --out <json> [--repeats 5]

Times, separately: model load (load_from_checkpoint as rna_infer does),
feature-file read, host->device upload of the per-sample inputs, and the
network evaluation of every sub-pixel x light-sample query (the loop of
rna_infer.main, mode current, light model area-sampled, reproduced here so
the network can be timed without the file I/O around it).  The loop's image
must equal rna_infer.py's own output for the same checkpoint and features
(max abs difference reported; the timing is rejected if it is not ~0).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--training-light-intensity", type=float, default=5.0)
    ap.add_argument("--batch", type=int, default=1 << 20)
    args = ap.parse_args()
    sys.path.insert(0, os.getcwd())
    import torch as th

    from rna import interfaces
    from utils import ops

    dev = th.device("cuda")
    th.zeros(1, device=dev)
    th.cuda.synchronize()
    t = time.perf_counter()
    module = interfaces.NeuralSurfaceTriplaneModule.load_from_checkpoint(args.checkpoint, strict=False, map_location=dev)
    module.model.initial_sigma = 1
    module.model.iterations_to_sigma_1 = 1
    module.to(dev).eval().freeze()
    th.cuda.synchronize()
    t_load = time.perf_counter() - t

    t = time.perf_counter()
    z = np.load(args.features)
    res, k = int(z["res"]), int(z["samples"])
    hit = z["hit"].astype(bool)
    idx = np.flatnonzero(hit)
    pos, lo, hi = z["position"], z["current_aabb_min"], z["current_aabb_max"]
    light_dir = z["light_dir"].astype(np.float32)
    light_w = z["light_weight"].astype(np.float32)
    light_vis = z["light_vis"]
    pos_h, cam_h, nrm_h = pos[idx], z["camera_dir"][idx], z["normal"][idx]
    t_read = time.perf_counter() - t
    m = light_dir.shape[1]

    t = time.perf_counter()
    tt = lambda a: th.tensor(np.ascontiguousarray(a), dtype=th.float32, device=dev)
    g_pos, g_cam, g_nrm = tt(pos_h), tt(cam_h), tt(nrm_h)
    g_ld, g_w, g_vis = tt(light_dir), tt(light_w), tt(light_vis.astype(np.float32))
    g_lo, g_hi = tt(lo).unsqueeze(0), tt(hi).unsqueeze(0)
    th.cuda.synchronize()
    t_upload = time.perf_counter() - t
    step = max(1, args.batch // m)
    n = len(idx)

    def network():
        rad = th.zeros((n, 3), device=dev)
        with th.no_grad():
            for s in range(0, n, step):
                e = min(s + step, n)
                rep = lambda a: a[s:e].unsqueeze(1).expand(-1, m, -1).reshape(-1, a.shape[-1])
                p = ops.normalize_positions(rep(g_pos).unsqueeze(0), g_lo, g_hi)
                pred = module(p, rep(g_cam).unsqueeze(0), g_ld[s:e].reshape(1, -1, 3), rep(g_nrm).unsqueeze(0))
                vis = g_vis[s:e].reshape(1, -1, 1)
                pred = th.where(vis > 0.0, pred[..., :3], pred[..., 3:]).clamp(min=0.0)
                w = g_w[s:e].reshape(1, -1, 1) / args.training_light_intensity
                rad[s:e] = (pred * w).reshape(e - s, m, 3).mean(1)
        return rad

    network()  # warm-up
    th.cuda.synchronize()
    times = []
    for _ in range(args.repeats):
        th.cuda.synchronize()
        t = time.perf_counter()
        rad = network()
        th.cuda.synchronize()
        times.append(time.perf_counter() - t)
    full = np.zeros((len(hit), 3), np.float32)
    full[idx] = rad.cpu().numpy()
    img = full.reshape(res * res, k, 3).mean(1).reshape(res, res, 3)
    ref = np.load(args.reference)
    diff = float(np.abs(img - ref).max())
    out = {"checkpoint": args.checkpoint, "features": args.features, "res": res, "subpixel_samples": k,
           "light_samples": m, "hit_samples": n, "network_queries": n * m, "batch": args.batch,
           "model_load_s": t_load, "feature_read_s": t_read, "upload_s": t_upload,
           "network_s": times, "network_s_median": float(np.median(times)),
           "max_abs_diff_vs_rna_infer": diff, "equivalent": diff <= 1e-5 * max(1.0, float(np.abs(ref).max())),
           "gpu": th.cuda.get_device_name(0), "torch": th.__version__,
           "max_memory_allocated_mib": th.cuda.max_memory_allocated() / 2 ** 20}
    open(args.out, "w").write(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("network_s_median", "model_load_s", "feature_read_s", "max_abs_diff_vs_rna_infer")}))
    return 0 if out["equivalent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
