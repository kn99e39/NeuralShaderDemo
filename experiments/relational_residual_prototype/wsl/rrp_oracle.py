"""Oracle diagnostic for the relational-residual prototype (WSL, RNA venv) -- NOT a runtime method.

    cd external/relightable-neural-assets
    .venv/bin/python ../../experiments/relational_residual_prototype/wsl/rrp_oracle.py --run <run dir> --project-commit <sha>

Predeclared in protocol rrp_v1.json ("oracle_diagnostic"), run because the relational
candidate failed T3.  The probe state g is replaced by a reference-derived incident-
transport state: the worklog-24 path-class radiances of each ROI pixel
(transport_decomposition/common_light/roi_<state>_A.npz, 9 classes x RGB, log1p;
T0_B uses seed B), encoded by an MLP of the local-only encoder's shape and fed to
the same decoder with the same local inputs.  Training contract identical to
rrp_train.py: training states only, same loss, optimiser, steps, batch, selection
and seeds.  It asks whether the shared operator can learn and carry a T3 correction
across configurations when the needed transport information is present in its state.
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
sys.path.insert(0, str(HERE))

import rrp_common as C  # noqa: E402
from rrp_model import Decoder, mlp, n_params  # noqa: E402
from rrp_train import SplitError, load_eval_state, load_training_state  # noqa: E402

import torch as th  # noqa: E402
import torch.nn as nn  # noqa: E402


class OracleResidual(nn.Module):
    kind = "oracle"

    def __init__(self, oracle_dim: int, local_dim: int, cfg: dict):
        super().__init__()
        self.lam = mlp([oracle_dim, *cfg["local_encoder"], cfg["state_dim"]], last_act=False)
        self.psi = Decoder(cfg["state_dim"], local_dim, cfg["decoder"])

    def forward(self, local, oracle):
        return self.psi(self.lam(oracle), local)


def oracle_features(name: str, d: dict) -> np.ndarray:
    """(stable pixels, 27) log1p path-class radiance, in the dataset's stable-pixel order."""
    st, seed = ("T0", "B") if name == "T0_B" else (name, "A")
    z = np.load(C.RES8 / "transport_decomposition" / "common_light" / f"roi_{st}_{seed}.npz")
    pix = d["roi_pixels"]
    if not np.array_equal(z["pix"], pix):
        raise RuntimeError(f"decomposition pixels differ from the ROI pixels for {name}")
    cls = z["classes"].reshape(len(pix), -1)[d["stable_pixels"]]
    return np.log1p(np.maximum(cls, 0.0)).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--project-commit", required=True)
    args = ap.parse_args()
    run = Path(args.run)
    proto = C.protocol()
    sp, trc, mc = proto["split"], proto["training"], proto["model"]
    if set(sp["train_states"]) & set(sp["holdout_states"]):
        raise SplitError("train and hold-out states overlap")
    dev = th.device("cuda")
    tr = {s: load_training_state(run, s, proto) for s in sp["train_states"]}
    k = int(next(iter(tr.values()))["samples_per_pixel"])
    parts = {"train": [], "val": []}
    for s, d in tr.items():
        pix = d["roi_pixels"][d["stable_pixels"]]
        val = C.validation_pixels(pix, 512, sp["val_block"], sp["val_modulus"])
        o = oracle_features(s, d)
        for part, sel in (("train", ~val), ("val", val)):
            idx = np.flatnonzero(sel)
            q = (idx[:, None] * k + np.arange(k)[None]).reshape(-1)
            parts[part].append((d["local"][q], o[idx], d["gt_pixels"][idx] - d["frozen_pixels"][idx], d["gt_pixels"][idx]))
    cat = lambda part, j: np.concatenate([p[j] for p in parts[part]])
    Ltr, Otr, Ytr, Gtr = (cat("train", j) for j in range(4))
    Lva, Ova, Yva, Gva = (cat("val", j) for j in range(4))
    stats = {"local_mean": Ltr.mean(0), "local_std": np.maximum(Ltr.std(0), 1e-3),
             "oracle_mean": Otr.mean(0), "oracle_std": np.maximum(Otr.std(0), 1e-3)}
    tt = lambda a: th.tensor(np.ascontiguousarray(a), dtype=th.float32, device=dev)
    S = {key: tt(v) for key, v in stats.items()}
    nl = lambda x: (x - S["local_mean"]) / S["local_std"]
    no = lambda x: (x - S["oracle_mean"]) / S["oracle_std"]
    gtr = (nl(tt(Ltr)).reshape(-1, k, Ltr.shape[-1]), no(tt(Otr)), tt(Ytr), tt(Gtr))
    gva = (nl(tt(Lva)).reshape(-1, k, Lva.shape[-1]), no(tt(Ova)), tt(Yva), tt(Gva))

    def predict(model, L, O, chunk=512):
        outs = []
        for s in range(0, len(L), chunk):
            l, o = L[s:s + chunk], O[s:s + chunk]
            q = l.shape[0]
            y = model(l.reshape(q * k, -1), o[:, None].expand(q, k, o.shape[-1]).reshape(q * k, -1))
            outs.append(y.reshape(q, k, 3).mean(1))
        return th.cat(outs)

    loss_fn = lambda p, Y, G: (((p - Y) / (G + trc["loss_scale_offset"])) ** 2).mean()
    res = {"project_commit": args.project_commit, "diagnostic_only": True, "runs": {}}
    for seed in trc["seeds"]:
        th.manual_seed(seed)
        model = OracleResidual(Otr.shape[-1], Ltr.shape[-1], mc).to(dev)
        opt = th.optim.Adam(model.parameters(), lr=trc["lr"])
        rng = np.random.default_rng(seed)
        best, curve = None, []
        t0 = time.perf_counter()
        for step in range(1, trc["steps"] + 1):
            idx = th.tensor(rng.integers(0, gtr[2].shape[0], trc["batch_pixels"]), device=dev)
            model.train()
            loss = loss_fn(predict(model, gtr[0][idx], gtr[1][idx], trc["batch_pixels"]), gtr[2][idx], gtr[3][idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % trc["val_every"] == 0 or step == trc["steps"]:
                model.eval()
                with th.no_grad():
                    vl = float(loss_fn(predict(model, gva[0], gva[1]), gva[2], gva[3]))
                curve.append({"step": step, "train_loss": float(loss), "val_loss": vl})
                if best is None or vl < best:
                    best, best_step, best_state = vl, step, {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()
        preds = {}
        with th.no_grad():
            for name in proto["evaluate_states"]:
                d = load_eval_state(run, name)
                preds[name] = predict(model, nl(tt(d["local"])).reshape(-1, k, d["local"].shape[-1]), no(tt(oracle_features(name, d)))).cpu().numpy()
        tag = f"oracle_s{seed}"
        np.savez(run / "models" / f"pred_{tag}.npz", **preds)
        res["runs"][tag] = {"branch": "oracle", "seed": seed, "params": n_params(model), "train_s": time.perf_counter() - t0,
                            "best_step": best_step, "best_val_loss": best, "final_train_loss": curve[-1]["train_loss"], "curve": curve}
        print(tag, "params", n_params(model), f"best step {best_step} val {best:.5f}", flush=True)
    (run / "models" / "oracle_training.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
