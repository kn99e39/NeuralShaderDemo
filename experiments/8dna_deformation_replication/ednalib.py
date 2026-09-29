"""Shared helpers for the RTX 5080 (native Windows) 8DNA replication runs.

Upstream source under external/8dna26 is imported, never modified.  Import
this module first; `init_upstream()` performs the demo notebook's setup
(cuda_ad_rgb variant, LoopRecord off) and registers the official
`neuralpath` / `neuralvolpath` integrators.

One runtime deviation from the notebook: VCallRecord stays on.  With it off,
DrJit 0.4.6 dispatches virtual calls in wavefront mode, and on this SM 12.0
GPU that dispatch never returns once a render exceeds 8192 lanes (64x64x4 and
128x128x4 time out; 4096 and 8192 lanes finish).  At 4096 and 8192 lanes the
two modes agree to 1.1e-5 in linear radiance at the same seed, i.e. the flag
changes kernel construction, not the estimator.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "external" / "8dna26"
EXPERIMENT = ROOT / "experiments" / "8dna_deformation_replication"
RESULTS = ROOT / "results" / "8dna_replication"
OFFICIAL_REPOSITORY = "https://github.com/lwwu2/8dna26"
OFFICIAL_COMMIT = "4a2157ca24e506c5ac0831f27d656ecc50a64f07"

_mi = None
_dr = None


def init_upstream():
    """Import Mitsuba/DrJit and the official integrators as the demo notebook does."""
    global _mi, _dr
    if _mi is not None:
        return _mi, _dr
    os.chdir(UPSTREAM)
    sys.path.insert(0, str(UPSTREAM))
    import mitsuba as mi

    mi.set_variant("cuda_ad_rgb")
    import drjit as dr

    import models.integrator  # noqa: F401  registers neuralpath / neuralvolpath

    dr.set_flag(dr.JitFlag.LoopRecord, False)
    dr.set_flag(dr.JitFlag.VCallRecord, True)  # notebook: False; see module docstring
    _mi, _dr = mi, dr
    return mi, dr


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")


def git_head(path: Path) -> str | None:
    proc = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    return proc.stdout.strip() or None


def git_dirty(path: Path) -> list[str]:
    proc = subprocess.run(["git", "-C", str(path), "status", "--porcelain"], capture_output=True, text=True)
    return [line for line in proc.stdout.splitlines() if line.strip()]


def _cmd(args: list[str]) -> str | None:
    try:
        return subprocess.run(args, capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        return None


def environment_record() -> dict:
    """Runtime identity for the manifest; call after init_upstream()."""
    import drjit
    import mitsuba
    import torch

    return {
        "os": platform.platform(),
        "cuda_home": os.environ.get("CUDA_HOME"),
        "msvc": os.environ.get("VCToolsVersion"),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "torch_arch_list": torch.cuda.get_arch_list(),
        "mitsuba": mitsuba.__version__,
        "drjit": drjit.__version__,
        "numpy": np.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "nvidia_smi": _cmd(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "nvcc": (_cmd([str(Path(os.environ.get("CUDA_HOME", "")) / "bin" / "nvcc"), "--version"]) or "").splitlines()[-1:],
        "project_commit": git_head(ROOT),
        "project_dirty": git_dirty(ROOT),
        "upstream_commit": git_head(UPSTREAM),
        "jit_flags": {"LoopRecord_neural": False, "VCallRecord": True},
    }


def checkpoint_path(asset: str) -> Path:
    return UPSTREAM / "checkpoints" / f"{asset}.ckpt"


def load_neural_integrator(asset: str, integrator_type: str = "neuralpath", max_depth: int = 10, rr_depth: int = 5):
    """The integrator dict of demo/demo.ipynb, for the instance id 'instance0'."""
    mi, _ = init_upstream()
    return mi.load_dict(
        {
            "type": integrator_type,
            "max_depth": max_depth,
            "rr_depth": rr_depth,
            "device": 0,
            "model_type": "8dna",
            "asset_instance0": str(checkpoint_path(asset)),
        }
    )


def render_chunked(scene, integrator, spp: int, chunk: int, seed: int) -> tuple[np.ndarray, float]:
    """Average of spp//chunk renders with seeds seed, seed+1, ... (demo notebook protocol)."""
    mi, dr = init_upstream()
    import torch

    torch.cuda.synchronize()
    start = time.perf_counter()
    img = None
    n = spp // chunk
    for s in range(n):
        part = mi.render(scene, integrator=integrator, spp=chunk, seed=seed + s)
        img = part if img is None else img + part
        if (s + 1) % max(1, n // 8) == 0:
            print(f"  render {type(integrator).__name__} {s + 1}/{n} chunks, {time.perf_counter() - start:.0f}s", flush=True)
    img /= spp // chunk
    out = np.array(img, dtype=np.float32)
    torch.cuda.synchronize()
    return out, time.perf_counter() - start


def render_reference(scene, spp: int, chunk: int, seed: int) -> tuple[np.ndarray, float]:
    """Path-traced reference with the scene's own integrator (prb / prbvolpath).

    Loop recording is re-enabled for the reference only; it changes how the
    kernel is compiled, not the estimator.
    """
    mi, dr = init_upstream()
    dr.set_flag(dr.JitFlag.LoopRecord, True)
    try:
        return render_chunked(scene, scene.integrator(), spp, chunk, seed)
    finally:
        dr.set_flag(dr.JitFlag.LoopRecord, False)


# --- image output ------------------------------------------------------------

def tonemap(img: np.ndarray) -> np.ndarray:
    """Display transform of the demo notebook: clamp(x^(1/2.2), 0, 1)."""
    return np.clip(np.clip(img, 0, None) ** (1 / 2.2), 0, 1)


def to_u8(img_ldr: np.ndarray) -> np.ndarray:
    return (np.clip(img_ldr, 0, 1) * 255 + 0.5).astype(np.uint8)


def save_exr(path: Path, img: np.ndarray) -> None:
    mi, _ = init_upstream()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mi.Bitmap(np.ascontiguousarray(img, dtype=np.float32)).write(str(path))


def load_exr(path: Path) -> np.ndarray:
    mi, _ = init_upstream()
    return np.array(mi.Bitmap(str(path)), dtype=np.float32)[..., :3]


def save_png(path: Path, img_u8: np.ndarray) -> None:
    from PIL import Image

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_u8).save(path)


def label(img_u8: np.ndarray, text: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    im = Image.fromarray(img_u8)
    draw = ImageDraw.Draw(im)
    draw.rectangle([0, 0, 8 + 7 * len(text), 16], fill=(0, 0, 0))
    draw.text((4, 2), text, fill=(255, 255, 255))
    return np.array(im)


def save_gif(path: Path, frames: list[np.ndarray], ms: int = 700) -> None:
    from PIL import Image

    ims = [Image.fromarray(f) for f in frames]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=ms, loop=0)


def error_map(err: np.ndarray, vmax: float) -> np.ndarray:
    """Per-pixel scalar error -> inferno RGB with a fixed, locked scale."""
    import matplotlib

    cmap = matplotlib.colormaps["inferno"]
    return to_u8(cmap(np.clip(err / vmax, 0, 1))[..., :3])


# --- metrics -----------------------------------------------------------------

def metrics(est: np.ndarray, ref: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Linear-HDR and display-space metrics, optionally restricted to a mask."""
    from skimage.metrics import structural_similarity

    e_ldr, r_ldr = tonemap(est), tonemap(ref)
    sel = np.ones(est.shape[:2], bool) if mask is None else mask.astype(bool)
    n = int(sel.sum())
    if n == 0:
        return {"pixels": 0}
    d_lin = (est - ref)[sel]
    d_ldr = (e_ldr - r_ldr)[sel]
    mse_ldr = float(np.mean(d_ldr ** 2))
    out = {
        "pixels": n,
        "linear_mae": float(np.mean(np.abs(d_lin))),
        "linear_rmse": float(np.sqrt(np.mean(d_lin ** 2))),
        "linear_relmse": float(np.mean(d_lin ** 2 / (ref[sel] ** 2 + 1e-2))),
        "display_mae": float(np.mean(np.abs(d_ldr))),
        "display_psnr_db": float("inf") if mse_ldr == 0 else float(-10 * np.log10(mse_ldr)),
    }
    if mask is None:
        out["display_ssim"] = float(structural_similarity(e_ldr, r_ldr, channel_axis=2, data_range=1.0))
    return out


def pixel_metrics(est: np.ndarray, ref: np.ndarray) -> dict:
    """metrics() for paired pixel arrays of shape (n, 3) (no SSIM)."""
    if len(est) == 0:
        return {"pixels": 0}
    d_lin = est - ref
    d_ldr = tonemap(est) - tonemap(ref)
    mse_ldr = float(np.mean(d_ldr ** 2))
    return {
        "pixels": int(len(est)),
        "linear_mae": float(np.mean(np.abs(d_lin))),
        "linear_rmse": float(np.sqrt(np.mean(d_lin ** 2))),
        "linear_relmse": float(np.mean(d_lin ** 2 / (ref ** 2 + 1e-2))),
        "display_mae": float(np.mean(np.abs(d_ldr))),
        "display_psnr_db": float("inf") if mse_ldr == 0 else float(-10 * np.log10(mse_ldr)),
    }


def write_json(path: Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))
