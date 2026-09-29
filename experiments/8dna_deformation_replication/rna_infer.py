"""RNA inference on rna_bridge feature buffers (runs in RNA's WSL environment).

    cd external/relightable-neural-assets
    .venv/bin/python ../../experiments/8dna_deformation_replication/rna_infer.py \
        --checkpoint <ckpt> --train-h5 <teaset_T0_train.h5> --features <state.npz> \
        --mode canonical|current --out <prefix>

Mirrors rna/renderers.py render_frame_neural for a directional light:
module(position_aabb01, camera_dir, light_dir, normal) -> 6 channels, the
lit (visibility > 0) or shadowed branch, clamp >= 0, times
irradiance / training_light_intensity.  The module is loaded as the official
renderer does (load_from_checkpoint, strict=False, blur sigma 1).  Pixel value
= mean over the buffer's sub-pixel samples, misses count 0.

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
    ap.add_argument("--batch", type=int, default=1 << 20)
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
    to_light = np.broadcast_to(z["to_light"], (len(idx), 3)).astype(np.float32)
    scale = float(z["irradiance"]) / args.training_light_intensity
    rad = np.zeros((len(hit), 3), np.float32)
    with th.no_grad():
        for s in range(0, len(idx), args.batch):
            sel = idx[s:s + args.batch]
            t = lambda a: th.tensor(np.ascontiguousarray(a), dtype=th.float32, device=dev).unsqueeze(0)
            p = ops.normalize_positions(t(pos[sel]), t(lo), t(hi))
            pred = module(p, t(z["camera_dir"][sel]), t(to_light[:len(sel)]), t(z["normal"][sel]))
            vis = t(z["visibility"][sel].astype(np.float32)[:, None])
            pred = th.where(vis > 0.0, pred[..., :3], pred[..., 3:]).clamp(min=0.0)
            rad[sel] = (pred * scale).squeeze(0).cpu().numpy()
    img = rad.reshape(res * res, k, 3).mean(1).reshape(res, res, 3)
    np.save(args.out + ".npy", img)
    json.dump({"checkpoint": args.checkpoint, "features": args.features, "mode": args.mode, "train_h5_aabb": [train_min.tolist(), train_max.tolist()],
               "aabb_used": [np.asarray(lo).tolist(), np.asarray(hi).tolist()], "irradiance_scale": scale,
               "hit_samples": int(len(idx))}, open(args.out + ".json", "w"), indent=1)
    print("wrote", args.out + ".npy", "mean", float(img.mean()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
