"""Render many RNA checkpoints at one state with the unchanged rna_infer.py (WSL, RNA venv).

    cd external/relightable-neural-assets
    .venv/bin/python ../../experiments/neural_recompute_cost/wsl/nrc_rna_eval_wsl.py --jobs <jobs.json> --out <timing.json>

jobs.json: {"common": [rna_infer args shared by every job], "jobs": [{"checkpoint": ..., "out": ...}, ...]}.
Calls rna_infer.main() once per checkpoint in one process (it re-reads the
checkpoint and the feature file each call, exactly as a standalone call does),
so per-call wall time is the existing evaluation path's cost after the first
call's CUDA/import warm-up.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def checkpoint_meta(path: str) -> dict:
    """epoch / global_step fields and a digest of the weights (sorted keys, raw bytes).

    RNA checkpoints pickle RNA's own classes, so they are read here, in RNA's venv.
    """
    import hashlib

    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False)
    h = hashlib.sha256()
    for k in sorted(ck["state_dict"]):
        v = ck["state_dict"][k].contiguous()
        h.update(k.encode())
        h.update(str(v.dtype).encode())
        h.update(v.numpy().tobytes() if v.dtype != torch.bfloat16 else v.float().numpy().tobytes())
    return {"ckpt_epoch_field": int(ck.get("epoch", -1)), "global_step": int(ck.get("global_step", -1)),
            "weights_digest": h.hexdigest()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    spec = json.loads(Path(args.jobs).read_text())
    sys.path.insert(0, str(HERE.parents[1] / "8dna_deformation_replication"))
    import torch

    import rna_infer

    timing = []
    for i, job in enumerate(spec["jobs"]):
        sys.argv = ["rna_infer.py", "--checkpoint", job["checkpoint"], "--out", job["out"], *spec["common"]]
        torch.cuda.synchronize()
        t = time.perf_counter()
        rc = rna_infer.main()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        timing.append({"checkpoint": job["checkpoint"], "out": job["out"], "seconds": dt, "rc": rc, "call_index": i,
                       **checkpoint_meta(job["checkpoint"])})
        print(f"{i + 1}/{len(spec['jobs'])} {Path(job['checkpoint']).name} {dt:.1f}s", flush=True)
        if rc:
            break
    Path(args.out).write_text(json.dumps({"calls": timing}, indent=1))
    return 0 if all(t["rc"] == 0 for t in timing) and len(timing) == len(spec["jobs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
