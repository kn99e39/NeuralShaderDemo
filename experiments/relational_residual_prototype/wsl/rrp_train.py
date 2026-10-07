"""Train the relational candidate and the matched local-only control; predict every state (WSL).

    cd external/relightable-neural-assets
    .venv/bin/python ../../experiments/relational_residual_prototype/wsl/rrp_train.py --run <run dir> --project-commit <sha>

Only the residual branch trains; the frozen RNA is never loaded here (its outputs
were precomputed by rrp_features.py, and the checkpoint's hash is re-checked).
Training data: protocol "split.train_states" only, minus the spatial validation
blocks; selection: lowest validation loss on those blocks of the training states.
Held-out states are opened only after training, for prediction.

Loss (both branches, protocol "training"): per ROI pixel, residual prediction =
mean of its 16 sample predictions; L = mean_c ((dL_pred - dL_GT) / (GT + 0.1))^2,
dL_GT = GT - frozen RNA (historical pixels).  A fixed per-pixel scale, no ROI or
state weighting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import rrp_common as C  # noqa: E402
from rrp_model import LocalResidual, RelationalResidual, n_params  # noqa: E402


class SplitError(RuntimeError):
    pass


def load_training_state(run: Path, name: str, proto: dict) -> dict:
    """The only way training data is read: refuses anything but the protocol's training states."""
    if name not in proto["split"]["train_states"] or name in proto["split"]["holdout_states"]:
        raise SplitError(f"{name} is not a training state")
    return dict(np.load(run / "dataset" / f"{name}.npz"))


def load_eval_state(run: Path, name: str) -> dict:
    return dict(np.load(run / "dataset" / f"{name}.npz"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--project-commit", required=True)
    ap.add_argument("--smoke-steps", type=int, default=None, help="smoke runs only: an untrained model, never evidence")
    args = ap.parse_args()
    run = Path(args.run)
    proto = C.protocol()
    sp, trc, mc = proto["split"], proto["training"], proto["model"]
    if args.smoke_steps:
        trc = dict(trc, steps=args.smoke_steps, val_every=min(trc["val_every"], args.smoke_steps), seeds=trc["seeds"][:1])
    if set(sp["train_states"]) & set(sp["holdout_states"]):
        raise SplitError("train and hold-out states overlap")
    import torch as th

    dev = th.device("cuda")
    res = 512

    # ---- training data (training states only) ---------------------------------
    tr = {s: load_training_state(run, s, proto) for s in sp["train_states"]}
    pools = {"train": [], "val": []}
    for s, d in tr.items():
        pix = d["roi_pixels"][d["stable_pixels"]]
        val = C.validation_pixels(pix, res, sp["val_block"], sp["val_modulus"])
        for part, sel in (("train", ~val), ("val", val)):
            pools[part].append({"state": s, "pix_idx": np.flatnonzero(sel), "d": d})
    k = int(next(iter(tr.values()))["samples_per_pixel"])

    def gather(entries):
        loc, prb, tgt, gt = [], [], [], []
        for e in entries:
            d, idx = e["d"], e["pix_idx"]
            q = (idx[:, None] * k + np.arange(k)[None]).reshape(-1)
            loc.append(d["local"][q])
            prb.append(d["probes"][q])
            tgt.append(d["gt_pixels"][idx] - d["frozen_pixels"][idx])
            gt.append(d["gt_pixels"][idx])
        return np.concatenate(loc), np.concatenate(prb), np.concatenate(tgt), np.concatenate(gt)

    Ltr, Ptr, Ytr, Gtr = gather(pools["train"])
    Lva, Pva, Yva, Gva = gather(pools["val"])
    # normalisation statistics from the training split of the training states only (then frozen)
    stats = {"local_mean": Ltr.mean(0), "local_std": np.maximum(Ltr.std(0), 1e-3),
             "probe_mean": Ptr.reshape(-1, Ptr.shape[-1]).mean(0), "probe_std": np.maximum(Ptr.reshape(-1, Ptr.shape[-1]).std(0), 1e-3)}
    tt = lambda a: th.tensor(np.ascontiguousarray(a), dtype=th.float32, device=dev)
    S = {key: tt(v) for key, v in stats.items()}
    norm_l = lambda x: (x - S["local_mean"]) / S["local_std"]
    norm_p = lambda x: (x - S["probe_mean"]) / S["probe_std"]
    gtr = {"L": norm_l(tt(Ltr)).reshape(-1, k, Ltr.shape[-1]), "P": norm_p(tt(Ptr)).reshape(-1, k, *Ptr.shape[1:]),
           "Y": tt(Ytr), "G": tt(Gtr)}
    gva = {"L": norm_l(tt(Lva)).reshape(-1, k, Lva.shape[-1]), "P": norm_p(tt(Pva)).reshape(-1, k, *Pva.shape[1:]),
           "Y": tt(Yva), "G": tt(Gva)}
    counts = {"train_pixels": int(len(Ytr)), "val_pixels": int(len(Yva)), "samples_per_pixel": k,
              "train_queries": int(len(Ytr) * k), "probes_per_query": int(Ptr.shape[1]),
              "per_state": {e["state"]: int(len(e["pix_idx"])) for e in pools["train"]},
              "per_state_val": {e["state"]: int(len(e["pix_idx"])) for e in pools["val"]}}

    def predict_pixels(model, L, P, chunk=512):
        outs = []
        for s in range(0, len(L), chunk):
            l = L[s:s + chunk]
            q = l.shape[0]
            lf = l.reshape(q * k, -1)
            if model.kind == "relational":
                p = P[s:s + chunk]
                y = model(lf, p.reshape(q * k, *p.shape[2:]))
            else:
                y = model(lf)
            outs.append(y.reshape(q, k, 3).mean(1))
        return th.cat(outs)

    def loss_fn(pred, Y, G):
        return (((pred - Y) / (G + trc["loss_scale_offset"])) ** 2).mean()

    results = {"counts": counts, "project_commit": args.project_commit, "smoke_steps": args.smoke_steps, "runs": {}, "normalisation": {k_: v.tolist() for k_, v in stats.items()}}
    ckdir = run / "models"
    ckdir.mkdir(parents=True, exist_ok=True)
    for branch in ("local_only", "relational"):
        for seed in trc["seeds"]:
            th.manual_seed(seed)
            np.random.seed(seed)
            model = (RelationalResidual(len(C.PROBE_FEATURES), len(C.LOCAL_FEATURES), mc) if branch == "relational"
                     else LocalResidual(len(C.LOCAL_FEATURES), mc)).to(dev)
            opt = th.optim.Adam(model.parameters(), lr=trc["lr"])
            rng = np.random.default_rng(seed)
            n = gtr["Y"].shape[0]
            best, best_state, curve = None, None, []
            th.cuda.reset_peak_memory_stats()
            th.cuda.synchronize()
            t0 = time.perf_counter()
            for step in range(1, trc["steps"] + 1):
                idx = th.tensor(rng.integers(0, n, trc["batch_pixels"]), device=dev)
                model.train()
                pred = predict_pixels(model, gtr["L"][idx], gtr["P"][idx], chunk=trc["batch_pixels"])
                loss = loss_fn(pred, gtr["Y"][idx], gtr["G"][idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if step % trc["val_every"] == 0 or step == trc["steps"]:
                    model.eval()
                    with th.no_grad():
                        vl = float(loss_fn(predict_pixels(model, gva["L"], gva["P"]), gva["Y"], gva["G"]))
                    curve.append({"step": step, "train_loss": float(loss), "val_loss": vl})
                    if best is None or vl < best:
                        best, best_step = vl, step
                        best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
            th.cuda.synchronize()
            train_s = time.perf_counter() - t0
            model.load_state_dict(best_state)
            model.eval()
            tag = f"{branch}_s{seed}"
            th.save({"state_dict": best_state, "branch": branch, "seed": seed, "model_cfg": mc, "normalisation": stats,
                     "best_step": best_step}, ckdir / f"{tag}.pt")
            # ---- predictions for every evaluated state (hold-outs opened only now) ----
            preds = {}
            with th.no_grad():
                for name in proto["evaluate_states"]:
                    d = load_eval_state(run, name)
                    L = norm_l(tt(d["local"])).reshape(-1, k, d["local"].shape[-1])
                    P = norm_p(tt(d["probes"])).reshape(-1, k, *d["probes"].shape[1:])
                    preds[name] = predict_pixels(model, L, P).cpu().numpy()
                    if branch == "relational" and seed == trc["seeds"][0] and name in ("T0", "T3"):
                        g = model.state(P.reshape(-1, *P.shape[2:])).cpu().numpy().astype(np.float32)
                        np.save(run / "models" / f"dynamic_state_{name}.npy", g)
            np.savez(run / "models" / f"pred_{tag}.npz", **preds)
            results["runs"][tag] = {"branch": branch, "seed": seed, "params": n_params(model), "train_s": train_s,
                                    "peak_gpu_mib": th.cuda.max_memory_allocated() / 2 ** 20, "best_step": best_step,
                                    "best_val_loss": best, "final_train_loss": curve[-1]["train_loss"], "curve": curve}
            print(tag, "params", n_params(model), f"train {train_s:.1f}s best step {best_step} val {best:.5f}", flush=True)

    # ---- update / inference latency on the held-out T3 queries (seed-0 models) ---
    d = load_eval_state(run, "T3")
    L = norm_l(tt(d["local"]))
    P = norm_p(tt(d["probes"]))
    timing = {}
    for branch in ("local_only", "relational"):
        ck = th.load(ckdir / f"{branch}_s{trc['seeds'][0]}.pt", map_location=dev, weights_only=False)
        model = (RelationalResidual(len(C.PROBE_FEATURES), len(C.LOCAL_FEATURES), mc) if branch == "relational"
                 else LocalResidual(len(C.LOCAL_FEATURES), mc)).to(dev)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        with th.no_grad():
            def state():
                return model.state(P) if branch == "relational" else model.state(L)

            def full():
                g = state()
                return model.psi(g, L)

            for f in (state, full):
                f()
            th.cuda.synchronize()
            ts, tf = [], []
            for _ in range(proto["timing"]["repeats"]):
                t0 = time.perf_counter(); state(); th.cuda.synchronize(); ts.append(time.perf_counter() - t0)
                t0 = time.perf_counter(); full(); th.cuda.synchronize(); tf.append(time.perf_counter() - t0)
        timing[branch] = {"queries": int(L.shape[0]), "state_s_median": float(np.median(ts)),
                          "state_plus_decoder_s_median": float(np.median(tf)),
                          "decoder_s_median": float(np.median(tf) - np.median(ts)), "state_dim": mc["state_dim"],
                          "state_bytes_float32": int(L.shape[0] * mc["state_dim"] * 4)}
    results["timing_T3"] = timing
    h = hashlib.sha256((C.ROOT / proto["frozen_rna"]["checkpoint"]).read_bytes()).hexdigest()
    results["frozen_rna_checkpoint_sha256_after"] = h
    results["frozen_rna_unchanged"] = h == proto["frozen_rna"]["sha256"]
    (run / "models" / "training.json").write_text(json.dumps(results, indent=1))
    return 0 if results["frozen_rna_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
