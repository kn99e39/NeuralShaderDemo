"""Train the matched worklog-29 branches and predict every state (WSL, RNA venv) -- stage 3.

    cd external/relightable-neural-assets
    .venv/bin/python ../../experiments/radiometric_relation_state_prototype/wsl/rrs_train.py --run <run dir> \
        --project-commit <sha> [--variants geom20 zero shuffled real] [--oracle-seed A]

Inputs: the worklog-28 datasets (local inputs, 20-channel probe descriptors, references,
frozen pixels), read through worklog 28's own split-enforcing loader, plus this run's
proxy/<state>.npy (or oracle/<state>_<seed>.npy for the diagnostic).  The training loop
is worklog 28's (rrp_train.py) with only the probe tensor's channel count changed:
same model class and widths, loss, optimiser, steps, batch, validation blocks,
selection and seeds.

variant geom20 (seed 0 only) is the reproduction check: it must reproduce worklog-28
relational_s0's validation curve.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "relational_residual_prototype"))
sys.path.insert(0, str(HERE.parents[1] / "relational_residual_prototype" / "wsl"))

import rrs_common as C  # noqa: E402
import rrp_common as C28  # noqa: E402
from rrp_model import RelationalResidual, n_params  # noqa: E402
from rrp_train import SplitError, load_eval_state, load_training_state  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--project-commit", required=True)
    ap.add_argument("--variants", nargs="+", default=["geom20", "zero", "shuffled", "real"])
    ap.add_argument("--oracle-seed", default="A")
    ap.add_argument("--smoke-steps", type=int, default=None, help="smoke runs only: an untrained model, never evidence")
    args = ap.parse_args()
    run = Path(args.run)
    proto, p28 = C.protocol(), C28.protocol()
    if proto["split"]["train_states"] != p28["split"]["train_states"] or proto["split"]["holdout_states"] != p28["split"]["holdout_states"]:
        raise SplitError("worklog-29 split differs from worklog 28")
    sp, trc, mc = p28["split"], p28["training"], p28["model"]
    for key in ("lr", "steps", "batch_pixels", "val_every", "seeds", "loss_scale_offset"):
        if trc[key] != proto["training"][key]:
            raise SystemExit(f"training contract differs from worklog 28 in {key}")
    if args.smoke_steps:
        trc = dict(trc, steps=args.smoke_steps, val_every=args.smoke_steps)
    wl28 = C.ROOT / proto["worklog28_reference"]["run"]
    import torch as th

    dev = th.device("cuda")
    k = 16

    def radiometric(name):
        if args.variants == ["oracle"]:
            return C.encode(np.load(run / "oracle" / f"{name}_{args.oracle_seed}.npy"))
        return C.encode(np.load(run / "proxy" / f"{name}.npy"))

    def probes_for(d, name, variant):
        p20 = d["probes"]
        if variant == "geom20":
            return p20
        hit = p20[..., C28.PROBE_FEATURES.index("hit")] > 0
        return C.assemble(p20, radiometric(name), hit, "real" if variant == "oracle" else variant)

    out = {"project_commit": args.project_commit, "variants": args.variants, "smoke_steps": args.smoke_steps, "runs": {}}
    for variant in args.variants:
        tr = {s: load_training_state(wl28, s, p28) for s in sp["train_states"]}
        pools = {"train": [], "val": []}
        for s, d in tr.items():
            pix = d["roi_pixels"][d["stable_pixels"]]
            val = C28.validation_pixels(pix, 512, sp["val_block"], sp["val_modulus"])
            P = probes_for(d, s, variant)
            for part, sel in (("train", ~val), ("val", val)):
                idx = np.flatnonzero(sel)
                q = (idx[:, None] * k + np.arange(k)[None]).reshape(-1)
                pools[part].append((d["local"][q], P[q], d["gt_pixels"][idx] - d["frozen_pixels"][idx], d["gt_pixels"][idx]))
        cat = lambda part, j: np.concatenate([p[j] for p in pools[part]])
        Ltr, Ptr, Ytr, Gtr = (cat("train", j) for j in range(4))
        Lva, Pva, Yva, Gva = (cat("val", j) for j in range(4))
        stats = {"local_mean": Ltr.mean(0), "local_std": np.maximum(Ltr.std(0), 1e-3),
                 "probe_mean": Ptr.reshape(-1, Ptr.shape[-1]).mean(0), "probe_std": np.maximum(Ptr.reshape(-1, Ptr.shape[-1]).std(0), 1e-3)}
        tt = lambda a: th.tensor(np.ascontiguousarray(a), dtype=th.float32, device=dev)
        S = {key: tt(v) for key, v in stats.items()}
        nl = lambda x: (x - S["local_mean"]) / S["local_std"]
        npb = lambda x: (x - S["probe_mean"]) / S["probe_std"]
        gtr = {"L": nl(tt(Ltr)).reshape(-1, k, Ltr.shape[-1]), "P": npb(tt(Ptr)).reshape(-1, k, *Ptr.shape[1:]), "Y": tt(Ytr), "G": tt(Gtr)}
        gva = {"L": nl(tt(Lva)).reshape(-1, k, Lva.shape[-1]), "P": npb(tt(Pva)).reshape(-1, k, *Pva.shape[1:]), "Y": tt(Yva), "G": tt(Gva)}
        del Ptr, Pva

        def predict_pixels(model, L, P, chunk=512):
            outs = []
            for s in range(0, len(L), chunk):
                l, p = L[s:s + chunk], P[s:s + chunk]
                q = l.shape[0]
                outs.append(model(l.reshape(q * k, -1), p.reshape(q * k, *p.shape[2:])).reshape(q, k, 3).mean(1))
            return th.cat(outs)

        loss_fn = lambda pred, Y, G: (((pred - Y) / (G + trc["loss_scale_offset"])) ** 2).mean()
        seeds = trc["seeds"][:1] if variant == "geom20" else trc["seeds"]
        for seed in seeds:
            th.manual_seed(seed)
            np.random.seed(seed)
            model = RelationalResidual(gtr["P"].shape[-1], gtr["L"].shape[-1], mc).to(dev)
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
                    curve.append({"step": step, "train_loss": float(loss.detach()), "val_loss": vl})
                    if best is None or vl < best:
                        best, best_step = vl, step
                        best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
            th.cuda.synchronize()
            train_s = time.perf_counter() - t0
            model.load_state_dict(best_state)
            model.eval()
            tag = f"{variant}_s{seed}"
            (run / "models").mkdir(parents=True, exist_ok=True)
            th.save({"state_dict": best_state, "variant": variant, "seed": seed, "normalisation": stats, "best_step": best_step},
                    run / "models" / f"{tag}.pt")
            preds = {}
            with th.no_grad():
                for name in p28["evaluate_states"]:
                    d = load_eval_state(wl28, name)
                    P = probes_for(d, name, variant)
                    preds[name] = predict_pixels(model, nl(tt(d["local"])).reshape(-1, k, d["local"].shape[-1]),
                                                 npb(tt(P)).reshape(-1, k, *P.shape[1:])).cpu().numpy()
            np.savez(run / "models" / f"pred_{tag}.npz", **preds)
            out["runs"][tag] = {"variant": variant, "seed": seed, "params": n_params(model), "probe_channels": int(gtr["P"].shape[-1]),
                                "train_s": train_s, "peak_gpu_mib": th.cuda.max_memory_allocated() / 2 ** 20, "best_step": best_step,
                                "best_val_loss": best, "curve": curve}
            print(tag, "params", n_params(model), f"train {train_s:.1f}s best step {best_step} val {best:.5f}", flush=True)
        del gtr, gva
        th.cuda.empty_cache()

    # reproduction of worklog 28 by the geom20 run
    if "geom20_s0" in out["runs"]:
        h = json.loads((wl28 / "models" / "training.json").read_text())["runs"]["relational_s0"]
        a = [c["val_loss"] for c in out["runs"]["geom20_s0"]["curve"]]
        b = [c["val_loss"] for c in h["curve"]]
        out["wl28_reproduction"] = {"best_step_ours": out["runs"]["geom20_s0"]["best_step"], "best_step_wl28": h["best_step"],
                                    "best_val_ours": out["runs"]["geom20_s0"]["best_val_loss"], "best_val_wl28": h["best_val_loss"],
                                    "max_abs_val_curve_diff": float(np.max(np.abs(np.array(a) - np.array(b)))),
                                    "params_ours": out["runs"]["geom20_s0"]["params"], "params_wl28": h["params"]}
        print("wl28 reproduction", out["wl28_reproduction"], flush=True)

    # latency of the real branch on T3 (seed 0)
    if "real_s0" in out["runs"]:
        ck = th.load(run / "models" / "real_s0.pt", map_location=dev, weights_only=False)
        st = ck["normalisation"]
        model = RelationalResidual(len(C.PROBE_FEATURES), len(C28.LOCAL_FEATURES), mc).to(dev)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        d = load_eval_state(wl28, "T3")
        Lg = (tt(d["local"]) - tt(st["local_mean"])) / tt(st["local_std"])
        Pg = (tt(probes_for(d, "T3", "real")) - tt(st["probe_mean"])) / tt(st["probe_std"])
        with th.no_grad():
            g = model.state(Pg)
            model.psi(g, Lg)
            ts, td = [], []
            for _ in range(proto["timing"]["repeats"]):
                th.cuda.synchronize(); a = time.perf_counter(); g = model.state(Pg); th.cuda.synchronize(); ts.append(time.perf_counter() - a)
                a = time.perf_counter(); model.psi(g, Lg); th.cuda.synchronize(); td.append(time.perf_counter() - a)
        out["timing_T3_real"] = {"queries": int(Lg.shape[0]), "state_s_median": float(np.median(ts)),
                                 "decoder_s_median": float(np.median(td)), "state_dim": mc["state_dim"],
                                 "state_bytes_float32": int(Lg.shape[0] * mc["state_dim"] * 4)}
    name = "oracle_training.json" if args.variants == ["oracle"] else "training.json"
    (run / "models" / name).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
