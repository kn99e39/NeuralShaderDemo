"""Render frozen RNA for every locked state and query mode (drives rna_infer.py in WSL).

RNA's package and its custom bpy wheel live in the WSL venv, so inference runs
there; this script only selects the checkpoint, dispatches one WSL call per
(state, mode) and records what was run.  The feature buffers and the training
H5 come from rna_bridge.py, so the RNA and 8DNA renders share one scene, one
material set, one camera and one light.

    windows/run.ps1 render_rna_states.py --protocol protocol/teaset_cross_backbone_locked.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np

import ednalib as L

WSL_ROOT = "/mnt/c/Projects/NeuralShaderDemo"


def wsl(script: str) -> str:
    """Run one bash line in the RNA root inside WSL and return stdout."""
    proc = subprocess.run(
        ["wsl.exe", "-d", "Ubuntu-22.04", "-u", "root", "--exec", "bash", "-lc",
         f"cd {WSL_ROOT}/external/relightable-neural-assets && {script}"],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"WSL call failed ({proc.returncode}): {script}\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    return proc.stdout


def best_checkpoint(name: str) -> tuple[str, float]:
    """Validation-best checkpoint of an RNA run, by the official filename's val_psnr."""
    root = L.RESULTS / "rna_teaset" / "ckpt" / name
    best, best_psnr = None, -1.0
    for p in root.rglob("epoch=*-val_psnr=*.ckpt"):
        m = re.search(r"val_psnr=([0-9.]+)dB", p.name)
        if m and float(m.group(1)) > best_psnr:
            best, best_psnr = p, float(m.group(1))
    if best is None:
        raise SystemExit(f"no validation checkpoint under {root}")
    return str(best), best_psnr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--part", choices=("frozen", "refit"), required=True,
                    help="frozen: T0 model at every state and mode (+ T0 seed-B repeat); refit: T3 model at T3")
    args = ap.parse_args()
    proto = json.loads(open(L.EXPERIMENT / args.protocol, encoding="utf-8").read())
    L.init_upstream()  # environment_record reads the Mitsuba version, which needs a variant set
    out = L.RESULTS / proto["rna_output"]
    out.mkdir(parents=True, exist_ok=True)
    inf = proto["rna_inference"]
    rec = {"protocol": L.rel(L.EXPERIMENT / args.protocol), "part": args.part, "environment": L.environment_record(),
           "renders": {}}

    names = proto.get("rna_checkpoint_names", {"frozen": "rna-teaset-T0-common-light", "refit": "rna-teaset-T3-common-light"})
    light_model = inf.get("light_model", "area-sampled")
    rec["light_model"] = light_model
    if args.part == "frozen":
        ck, psnr = best_checkpoint(names["frozen"])
        rec["checkpoint_T0"] = {"path": ck, "val_psnr_db": psnr, "sha256": L.sha256(ck)}
        train_h5 = L.RESULTS / proto["rna_dataset_dir"] / "teaset_T0_train.h5"
        rec["train_h5_sha256"] = L.sha256(train_h5)
        jobs = [(s, m, ck, train_h5, f"{s}_{m}") for s in proto["states"] for m in inf["modes"]]
        # T0 again with independent area-light samples (features T0_B.npz): RNA's own
        # seed-to-seed noise for the decision rule's noise precondition
        jobs.append(("T0_B", "canonical", ck, train_h5, "T0_canonical_B"))
    else:
        ck3, psnr3 = best_checkpoint(names["refit"])
        rec["checkpoint_T3_refit"] = {"path": ck3, "val_psnr_db": psnr3, "sha256": L.sha256(ck3)}
        train_h5 = L.RESULTS / proto["rna_dataset_dir"] / "teaset_T3_train.h5"
        rec["train_h5_sha256"] = L.sha256(train_h5)
        # the refit's canonical frame is T3 itself, so its own state needs no pullback
        jobs = [("T3", "current", ck3, train_h5, "refit_T3_current")]

    for state, mode, ckpt, h5, tag in jobs:
        feats = L.RESULTS / proto["rna_features_dir"] / f"{state}.npz"
        prefix = out / tag
        stdout = wsl(
            f".venv/bin/python ../../experiments/8dna_deformation_replication/rna_infer.py"
            f" --checkpoint {ckpt.replace(chr(92), '/').replace('C:/Projects/NeuralShaderDemo', WSL_ROOT)}"
            f" --train-h5 {WSL_ROOT}/{L.rel(h5)}"
            f" --features {WSL_ROOT}/{L.rel(feats)}"
            f" --mode {mode}"
            f" --training-light-intensity {inf['training_light_intensity']}"
            f" --light-model {light_model}"
            f" --out {WSL_ROOT}/{L.rel(prefix)}")
        img = np.load(str(prefix) + ".npy")
        rec["renders"][tag] = {"state": state, "features": L.rel(feats), "mode": mode, "mean": float(img.mean()),
                               "finite": bool(np.isfinite(img).all()), "shape": list(img.shape)}
        print(tag, stdout.strip().splitlines()[-1] if stdout.strip() else "", flush=True)
    name = "rna_render.json" if args.part == "frozen" else "rna_refit_render.json"
    L.write_json(out / name, rec)
    print("written", L.rel(out / name))
    return 0


if __name__ == "__main__":
    rc = main()
    # Every output is written by now. The interpreter's teardown after DrJit and
    # torch have both been loaded crashes with an access violation on this
    # Windows setup (seen once the renders and the JSON were complete), which
    # would turn a finished step into a failed one; leave without teardown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
