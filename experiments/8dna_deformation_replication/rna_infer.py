"""RNA inference on rna_bridge feature buffers (runs in RNA's WSL environment).

    cd external/relightable-neural-assets
    .venv/bin/python ../../experiments/8dna_deformation_replication/rna_infer.py \
        --checkpoint <ckpt> --train-h5 <teaset_T0_train.h5> --features <state.npz> \
        --mode canonical|current --out <prefix>

Mirrors rna/renderers.py render_frame_neural with a RectangularLight:
module(position_aabb01, camera_dir, light_dir, normal) -> 6 channels, the
lit (visibility > 0) or shadowed branch, clamp >= 0, times the light sample's
irradiance / training_light_intensity.  Each sub-pixel sample carries M
area-light samples from rna_bridge.features (direction, weight, binary
segment visibility); the sample's radiance is their mean.  The module is
loaded as the official renderer does (load_from_checkpoint, strict=False,
blur sigma 1).  Pixel value = mean over the buffer's sub-pixel samples,
misses count 0.

mode canonical : position = pulled-back canonical hit, normalized by the
                 training H5 AABB (primary correspondence contract)
mode current   : position = current-world hit, normalized by the current
                 scene AABB (official renderer behaviour; diagnostic)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train-h5", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--mode", choices=("canonical", "current"), required=True)
    ap.add_argument("--training-light-intensity", type=float, default=5.0)
    ap.add_argument("--module", default="NeuralSurfaceTriplaneModule")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=1 << 20, help="network queries per batch")
    args = ap.parse_args()
    sys.path.insert(0, os.getcwd())
    import h5py
    import torch as th

    from rna import interfaces
    from utils import ops

    dev = th.device("cuda")
    module = getattr(interfaces, args.module).load_from_checkpoint(args.checkpoint, strict=False, map_location=dev)
    module.model.initial_sigma = 1
    module.model.iterations_to_sigma_1 = 1
    module.to(dev).eval().freeze()

    z = np.load(args.features)
    res, k = int(z["res"]), int(z["samples"])
    with h5py.File(args.train_h5, "r") as f:
        train_min, train_max = np.array(f.attrs["aabb_min"]), np.array(f.attrs["aabb_max"])
    if args.mode == "canonical":
        pos, lo, hi = z["canonical_position"], train_min, train_max
    else:
        pos, lo, hi = z["position"], z["current_aabb_min"], z["current_aabb_max"]
    hit = z["hit"].astype(bool)
    idx = np.flatnonzero(hit)
    if "light_dir" not in z.files:
        raise SystemExit("features lack area-light samples (superseded delta-light contract)")
    light_dir = z["light_dir"].astype(np.float32)   # (hits, M, 3), in hit order
    light_w = z["light_weight"].astype(np.float32)  # (hits, M)
    light_vis = z["light_vis"]                      # (hits, M)
    m = light_dir.shape[1]
    if light_dir.shape[0] != len(idx):
        raise SystemExit("light samples do not match the hit samples")
    pos_h, cam_h, nrm_h = pos[idx], z["camera_dir"][idx], z["normal"][idx]
    rad = np.zeros((len(hit), 3), np.float32)
    step = max(1, args.batch // m)
    t = lambda a: th.tensor(np.ascontiguousarray(a), dtype=th.float32, device=dev)
    with th.no_grad():
        for s in range(0, len(idx), step):
            e = min(s + step, len(idx))
            rep = lambda a: t(a[s:e]).unsqueeze(1).expand(-1, m, -1).reshape(-1, a.shape[-1])
            p = ops.normalize_positions(rep(pos_h).unsqueeze(0), t(lo).unsqueeze(0), t(hi).unsqueeze(0))
            pred = module(p, rep(cam_h).unsqueeze(0), t(light_dir[s:e]).reshape(1, -1, 3), rep(nrm_h).unsqueeze(0))
            vis = t(light_vis[s:e].astype(np.float32)).reshape(1, -1, 1)
            pred = th.where(vis > 0.0, pred[..., :3], pred[..., 3:]).clamp(min=0.0)
            w = t(light_w[s:e]).reshape(1, -1, 1) / args.training_light_intensity
            rad[idx[s:e]] = (pred * w).reshape(e - s, m, 3).mean(1).cpu().numpy()
    img = rad.reshape(res * res, k, 3).mean(1).reshape(res, res, 3)
    np.save(args.out + ".npy", img)
    json.dump({"checkpoint": args.checkpoint, "features": args.features, "mode": args.mode, "train_h5_aabb": [train_min.tolist(), train_max.tolist()],
               "aabb_used": [np.asarray(lo).tolist(), np.asarray(hi).tolist()],
               "training_light_intensity": args.training_light_intensity, "light_samples_per_sample": int(m),
               "light_seed": str(z["light_seed"]), "hit_samples": int(len(idx))}, open(args.out + ".json", "w"), indent=1)
    print("wrote", args.out + ".npy", "mean", float(img.mean()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
