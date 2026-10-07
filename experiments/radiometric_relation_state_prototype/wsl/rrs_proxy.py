"""Runtime radiometric proxy L_hat_{j->i} with the frozen RNA (WSL, RNA venv) -- worklog 29, stage 2.

    cd external/relightable-neural-assets
    .venv/bin/python ../../experiments/radiometric_relation_state_prototype/wsl/rrs_proxy.py --run <run dir>

`rna_render` is rna_infer.py's canonical-mode loop (load_from_checkpoint strict=False,
blur sigma 1, branch by per-light-sample visibility, clamp >= 0, x weight / training
intensity, mean over the light samples), applied to arbitrary surface points.  It is
used twice:
  * self-check: on the worklog-28 query samples it must reproduce worklog 28's frozen
    base_samples exactly -- i.e. the proxy is RNA's own renderer, nothing new;
  * proxy: on every current probe hit j, with camera = the query i (camera_dir j -> i),
    the hit's current normal faced to i and stage 1's current light samples at j.
Writes proxy/<state>.npy, (Q, K, 3) linear RGB, 0 on misses.  RNA is digested before
and after; it must not change.
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

import rrs_common as C  # noqa: E402
import rrp_common as C28  # noqa: E402


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def digest(module) -> str:
    import torch

    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().to("cpu", torch.float32).contiguous().numpy().tobytes())
    return h.hexdigest()


def make_renderer(module, lo, hi, E, dev):
    """rna_infer.py's canonical-mode loop over arbitrary surface points -> (n, 3) radiance."""
    import torch as th
    from utils import ops

    tt = lambda a: th.tensor(np.ascontiguousarray(a), dtype=th.float32, device=dev)

    def rna_render(pos, cam, nrm, ld, lw, lv):
        m = ld.shape[1]
        step = max(1, (1 << 20) // m)
        rad = np.zeros((len(pos), 3), np.float32)
        with th.no_grad():
            for s in range(0, len(pos), step):
                e = min(s + step, len(pos))
                rep = lambda a: tt(a[s:e]).unsqueeze(1).expand(-1, m, -1).reshape(-1, a.shape[-1])
                p = ops.normalize_positions(rep(pos).unsqueeze(0), lo, hi)
                pred = module(p, rep(cam).unsqueeze(0), tt(ld[s:e].astype(np.float32)).reshape(1, -1, 3), rep(nrm).unsqueeze(0))
                vis = tt(lv[s:e].astype(np.float32)).reshape(1, -1, 1)
                pred = th.where(vis > 0.0, pred[..., :3], pred[..., 3:]).clamp(min=0.0)
                rad[s:e] = (pred * (tt(lw[s:e]).reshape(1, -1, 1) / E)).reshape(e - s, m, 3).mean(1).cpu().numpy()
        return rad

    return rna_render


def load_frozen(proto28, cb, dev):
    """The worklog-22 frozen T0 RNA exactly as rna_infer.py loads it, plus the training-H5 AABB."""
    import h5py
    import torch as th

    from rna import interfaces

    ck = C.ROOT / proto28["frozen_rna"]["checkpoint"]
    if sha256(ck) != proto28["frozen_rna"]["sha256"]:
        raise SystemExit("frozen RNA checkpoint hash differs")
    module = interfaces.NeuralSurfaceTriplaneModule.load_from_checkpoint(str(ck), strict=False, map_location=dev)
    module.model.initial_sigma = 1
    module.model.iterations_to_sigma_1 = 1
    module.to(dev).eval().freeze()
    with h5py.File(C28.RES8 / cb["rna_dataset_dir"] / "teaset_T0_train.h5", "r") as f:
        lo = th.tensor(np.array(f.attrs["aabb_min"]), dtype=th.float32, device=dev).unsqueeze(0)
        hi = th.tensor(np.array(f.attrs["aabb_max"]), dtype=th.float32, device=dev).unsqueeze(0)
    return module, lo, hi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = Path(args.run)
    proto, p28 = C.protocol(), C28.protocol()
    cb = json.loads((C28.EXP8 / p28["base_protocol"]).read_text(encoding="utf-8"))
    wl28 = C.ROOT / proto["worklog28_reference"]["run"]
    sys.path.insert(0, os.getcwd())
    import torch as th
    from utils import ops

    dev = th.device("cuda")
    module, lo, hi = load_frozen(p28, cb, dev)
    d0 = digest(module)
    E = float(cb["rna_inference"]["training_light_intensity"])
    tt = lambda a: th.tensor(np.ascontiguousarray(a), dtype=th.float32, device=dev)

    rna_render = make_renderer(module, lo, hi, E, dev)

    def gpu_time(pos, cam, nrm, ld, lw, lv):
        """Median time of the rna_render network evaluation with every input already on the GPU."""
        m = ld.shape[1]
        g = [tt(pos), tt(cam), tt(nrm)]
        gld, glw, glv = tt(ld.astype(np.float32)), tt(lw), tt(lv.astype(np.float32))
        n, step, times = len(pos), max(1, (1 << 20) // m), []
        with th.no_grad():
            for rep_i in range(proto["timing"]["repeats"] + 1):
                th.cuda.synchronize()
                a = time.perf_counter()
                for s in range(0, n, step):
                    e = min(s + step, n)
                    rp = lambda x: x[s:e].unsqueeze(1).expand(-1, m, -1).reshape(1, -1, 3)
                    pred = module(ops.normalize_positions(rp(g[0]), lo, hi), rp(g[1]), gld[s:e].reshape(1, -1, 3), rp(g[2]))
                    pred = th.where(glv[s:e].reshape(1, -1, 1) > 0, pred[..., :3], pred[..., 3:]).clamp(min=0.0)
                    (pred * glw[s:e].reshape(1, -1, 1) / E).reshape(e - s, m, 3).mean(1)
                th.cuda.synchronize()
                if rep_i:
                    times.append(time.perf_counter() - a)
        return {"points": n, "rna_queries": n * m, "seconds_median": float(np.median(times))}

    rec = {"checkpoint_sha256": p28["frozen_rna"]["sha256"], "rna_digest_before": d0, "states": {},
           "torch": th.__version__, "gpu": th.cuda.get_device_name(0)}
    for name in proto["evaluate_states"]:
        # self-check: the same function on the worklog-28 query samples == worklog-28 frozen base
        z = np.load(wl28 / "probes" / f"{name}.npz")
        ds = np.load(wl28 / "dataset" / f"{name}.npz")
        base = rna_render(z["query_canonical"], z["query_camera_dir"], z["query_normal"], z["light_dir"], z["light_weight"], z["light_vis"])
        self_check = float(np.abs(base - ds["base_samples"]).max())
        rl = np.load(run / "remote_light" / f"{name}.npz")
        th.cuda.synchronize()
        t0 = time.perf_counter()
        prox = rna_render(rl["canonical"], rl["camera_dir"], rl["normal"], rl["light_dir"], rl["light_weight"], rl["light_vis"])
        th.cuda.synchronize()
        t_host = time.perf_counter() - t0
        q, k = (int(v) for v in rl["shape"])
        full = np.zeros((q, k, 3), np.float32)
        full[rl["query_index"], rl["probe_index"]] = prox
        out = run / "proxy" / f"{name}.npy"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, full)
        timing = None
        if name in p28["timing_states"]:  # GPU-resident inputs: the network evaluation alone (protocol timing state)
            timing = {"proxy": gpu_time(rl["canonical"], rl["camera_dir"], rl["normal"], rl["light_dir"], rl["light_weight"], rl["light_vis"]),
                      "frozen_rna_roi_render": gpu_time(z["query_canonical"], z["query_camera_dir"], z["query_normal"], z["light_dir"],
                                                        z["light_weight"], z["light_vis"])}
        rec["states"][name] = {"hits": int(len(prox)), "self_check_max_abs_vs_wl28_base": self_check,
                               "proxy_mean": prox.mean(0).tolist(), "proxy_max": float(prox.max()), "host_s": t_host, "gpu_timing": timing}
        print(name, rec["states"][name], flush=True)
    rec["rna_digest_after"] = digest(module)
    rec["rna_unchanged"] = rec["rna_digest_after"] == d0
    (run / "proxy" / "proxy.json").write_text(json.dumps(rec, indent=1))
    bad = [n for n, s in rec["states"].items() if s["self_check_max_abs_vs_wl28_base"] > 1e-6]
    if bad or not rec["rna_unchanged"]:
        print("FAIL", bad, rec["rna_unchanged"], flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
