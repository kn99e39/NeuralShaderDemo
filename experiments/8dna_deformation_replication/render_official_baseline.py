"""Reproduce the released 8DNA demo path without modifying upstream source."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "external" / "8dna26"
OUT_ROOT = ROOT / "results" / "8dna_deformation_replication" / "official_baseline"
MANIFEST = ROOT / "experiments" / "8dna_deformation_replication" / "baseline_manifest.json"
OFFICIAL_COMMIT = "4a2157ca24e506c5ac0831f27d656ecc50a64f07"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_manifest(payload: dict[str, object]) -> None:
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "official"), default="smoke")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = {
        "asset": "seal",
        "scene_function": "scene2",
        "resolution": 64 if args.mode == "smoke" else 256,
        "neural_spp": 1 if args.mode == "smoke" else 256,
        "chunk_spp": 1 if args.mode == "smoke" else 4,
        "reference_spp": 1 if args.mode == "smoke" else 256,
        "seed": 0,
        "max_depth": 10,
        "rr_depth": 5,
    }
    checkpoint = UPSTREAM / "checkpoints" / "seal.ckpt"
    payload: dict[str, object] = {
        "status": "running",
        "gate_closing_run": args.mode == "official",
        "official_repository": "https://github.com/lwwu2/8dna26",
        "official_commit": OFFICIAL_COMMIT,
        "config": config,
        "checkpoint": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha256(checkpoint) if checkpoint.is_file() else None,
        "python": sys.version,
        "platform": platform.platform(),
    }
    save_manifest(payload)

    original_cwd = Path.cwd()
    try:
        os.chdir(UPSTREAM)
        sys.path.insert(0, str(UPSTREAM))

        import numpy as np
        import torch
        import torch.nn.functional as torch_f
        import mitsuba as mi

        mi.set_variant("cuda_ad_rgb")
        import models.integrator  # noqa: F401 - registers official integrators
        from scenes.seal import scene2

        payload["environment"] = {
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "mitsuba": mi.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
        }

        scene = mi.load_dict(scene2(config["resolution"]))
        integrator = mi.load_dict(
            {
                "type": "neuralvolpath",
                "max_depth": config["max_depth"],
                "rr_depth": config["rr_depth"],
                "device": 0,
                "model_type": "8dna",
                "asset_instance0": str(checkpoint),
            }
        )

        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        image = None
        chunks = config["neural_spp"] // config["chunk_spp"]
        for chunk in range(chunks):
            sample = mi.render(
                scene,
                integrator=integrator,
                spp=config["chunk_spp"],
                seed=config["seed"] + chunk,
            )
            image = sample if image is None else image + sample
        image /= chunks
        torch.cuda.synchronize()
        neural_seconds = time.perf_counter() - start

        start = time.perf_counter()
        reference = mi.render(
            scene, spp=config["reference_spp"], seed=config["seed"]
        )
        torch.cuda.synchronize()
        reference_seconds = time.perf_counter() - start

        neural = image.torch()
        gt = reference.torch()
        mse = torch_f.mse_loss(neural, gt).item()
        psnr = float("inf") if mse == 0 else -10.0 * math.log10(mse)

        neural_exr = OUT_ROOT / f"{args.mode}_seal_8dna.exr"
        reference_exr = OUT_ROOT / f"{args.mode}_seal_pt.exr"
        comparison_png = OUT_ROOT / f"{args.mode}_seal_comparison.png"
        mi.Bitmap(image).write(str(neural_exr))
        mi.Bitmap(reference).write(str(reference_exr))

        display = torch.cat([gt, neural], dim=1).clamp_min(0).pow(1 / 2.2).clamp(0, 1)
        rgb8 = (display.cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        from PIL import Image

        Image.fromarray(rgb8).save(comparison_png)

        payload.update(
            {
                "status": "pass" if args.mode == "official" else "smoke_pass",
                "metrics": {"linear_mse": mse, "linear_psnr_db": psnr},
                "runtime_seconds": {
                    "neural": neural_seconds,
                    "reference": reference_seconds,
                },
                "max_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
                "outputs": {
                    "neural_exr": str(neural_exr.relative_to(ROOT)).replace("\\", "/"),
                    "reference_exr": str(reference_exr.relative_to(ROOT)).replace("\\", "/"),
                    "comparison_png": str(comparison_png.relative_to(ROOT)).replace("\\", "/"),
                },
            }
        )
        save_manifest(payload)
        print(json.dumps(payload, indent=2))
        return 0
    except Exception as exc:
        payload.update(
            {
                "status": "failed",
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        save_manifest(payload)
        print(json.dumps(payload, indent=2), file=sys.stderr)
        return 1
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    raise SystemExit(main())
