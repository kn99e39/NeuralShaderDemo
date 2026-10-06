"""Does the timing instrumentation change training?  (smoke-scale check, never evidence)

    windows/run.ps1 ../neural_recompute_cost/nrc_identity_check.py --out <dir> --rna-datasets <dir with teaset_T3_{train,val}.h5>

8DNA: the historical launcher train_8dna_state.py is run twice without
instrumentation and once through nrc_8dna_train_timed.py, on T3 with seed 9
and a tiny schedule (protocol "identity_check"); the final last.ckpt weights
are compared bit for bit.  RNA: the official scripts/train.py twice plain and
once through nrc_rna_train_timed.py (WSL, seed 0, tiny schedule on the smoke
datasets); final last.ckpt weights compared by digest.  The two plain runs
measure each trainer's own run-to-run determinism; the instrumented run is
judged against that floor.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "8dna_deformation_replication"))

import nrc_records as R  # noqa: E402
from nrc_wsl import RNA_ROOT, run_detached, wsl, wsl_path  # noqa: E402

ROOT = HERE.parents[1]
EXP8 = ROOT / "experiments" / "8dna_deformation_replication"


def verdict(block: dict, ratio: float) -> dict:
    """Instrumented runs must differ from plain runs no more than plain runs differ from each other (x ratio)."""
    floor = block["plain_a_vs_plain_b"]["max_abs_diff"]
    worst = max(block["plain_a_vs_instrumented"]["max_abs_diff"], block["plain_b_vs_instrumented"]["max_abs_diff"])
    if block["plain_a_vs_plain_b"]["bit_identical"]:
        ok = block["plain_a_vs_instrumented"]["bit_identical"] and block["plain_b_vs_instrumented"]["bit_identical"]
        rule = "plain runs are bit-identical, so the instrumented run must be too"
    else:
        ok = worst <= ratio * floor
        rule = f"instrumented-vs-plain max |dw| <= {ratio} x plain-vs-plain max |dw| (trainer is not bit-deterministic)"
    return {"rule": rule, "plain_floor_max_abs_diff": floor, "instrumented_worst_max_abs_diff": worst,
            "pass": bool(ok)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rna-datasets", required=True)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    proto = json.loads((HERE / "protocol" / "nrc_t3_recompute_v1.json").read_text(encoding="utf-8"))
    ic = proto["identity_check"]
    py = sys.executable
    rec = {"git": R.git_state(ROOT), "protocol_block": ic, "8dna": {}, "rna": {}}

    # --- 8DNA ------------------------------------------------------------------
    base = ["--state", "T3", "--protocol", "protocol/teaset_frozen_locked.json", "--device", "0",
            "--max_epochs", str(ic["8dna_epochs"]), "--seed", "9", *ic["8dna_overrides"]]
    runs = {"plain_a": None, "plain_b": None, "instrumented": None}
    for name in runs:
        d = out / "8dna" / name
        argv = base + ["--log_path", str(d), "--experiment_name", "run"]
        if name == "instrumented":
            cmd = [py, str(HERE / "nrc_8dna_train_timed.py"), "--events", str(d / "events.jsonl"), "--snapshots", str(d / "snap"),
                   "--export", str(d / "export.ckpt"), "--allow-dirty", "--", *argv]
        else:
            cmd = [py, str(EXP8 / "train_8dna_state.py"), *argv]
        with (out / f"8dna_{name}.log").open("w", encoding="utf-8") as fh:
            rc = subprocess.run(cmd, cwd=EXP8, stdout=fh, stderr=subprocess.STDOUT).returncode
        if rc:
            raise SystemExit(f"8DNA {name} failed rc={rc}")
        runs[name] = d / "run" / "last.ckpt"
    import torch

    sd = {k: torch.load(p, map_location="cpu", weights_only=False)["state_dict"] for k, p in runs.items()}

    def cmp(a, b):
        d = [(sd[a][k].float() - sd[b][k].float()).abs() for k in sd[a]]
        return {"bit_identical": all(torch.equal(sd[a][k], sd[b][k]) for k in sd[a]),
                "max_abs_diff": max(float(x.max()) for x in d),
                "mean_abs_diff": sum(float(x.sum()) for x in d) / sum(x.numel() for x in d)}

    rec["8dna"] = {"plain_a_vs_plain_b": cmp("plain_a", "plain_b"), "plain_a_vs_instrumented": cmp("plain_a", "instrumented"),
                   "plain_b_vs_instrumented": cmp("plain_b", "instrumented")}
    rec["8dna"]["verdict"] = verdict(rec["8dna"], ic["tolerance_ratio"])
    print("8DNA", rec["8dna"], flush=True)

    # --- RNA (WSL) ---------------------------------------------------------------
    ds = Path(args.rna_datasets).resolve()
    src = (EXP8 / "configs" / "rna_teaset_T3.yml").read_text(encoding="utf-8")
    cfg = re.sub(r"^num_epochs: .*$", f"num_epochs: {ic['rna_epochs']}", src, count=1, flags=re.M)
    cfg = re.sub(r'^training_dataset: .*$', f'training_dataset: "{wsl_path(ds / "teaset_T3_train.h5")}"', cfg, count=1, flags=re.M)
    cfg = re.sub(r'^validation_dataset: .*$', f'validation_dataset: "{wsl_path(ds / "teaset_T3_val.h5")}"', cfg, count=1, flags=re.M)
    digests = {}
    for name in ("plain_a", "plain_b", "instrumented"):
        d = out / "rna" / name
        d.mkdir(parents=True, exist_ok=True)
        c = re.sub(r'^name: .*$', f'name: "identity-{name}"', cfg, count=1, flags=re.M)
        (d / "config.yml").write_text(c, encoding="utf-8")
        train = ["--config", wsl_path(d / "config.yml"), "--checkpoint_dir", wsl_path(d / "ckpt"), "--seed", "0"]
        if name == "instrumented":
            job = [".venv/bin/python", wsl_path(HERE / "wsl" / "nrc_rna_train_timed.py"), "--events", wsl_path(d / "events.jsonl"),
                   "--snapshots", wsl_path(d / "snap"), "--snapshot-every", "1", "--project-commit", rec["git"]["commit"], "--", *train]
        else:
            job = [".venv/bin/python", "scripts/train.py", *train]
        rc = run_detached(f"identity_rna_{name}", out / "logs", job, RNA_ROOT)
        if rc:
            raise SystemExit(f"RNA {name} failed rc={rc}")
        last = next((d / "ckpt").rglob("last.ckpt"))
        p = wsl(["bash", "-c", f"cd {wsl_path(RNA_ROOT)} && .venv/bin/python -c \"import sys,json; sys.path.insert(0,'{wsl_path(HERE / 'wsl')}'); "
                               f"import nrc_rna_eval_wsl as W; print(json.dumps(W.checkpoint_meta('{wsl_path(last)}')))\""])
        digests[name] = json.loads(p.stdout.strip().splitlines()[-1])
    lasts = {n: wsl_path(next((out / "rna" / n / "ckpt").rglob("last.ckpt"))) for n in digests}
    code = "\n".join([
        "import torch, json",
        f"p = {json.dumps(lasts)}",
        "sd = {n: torch.load(f, map_location='cpu', weights_only=False)['state_dict'] for n, f in p.items()}",
        "def c(a, b):",
        "    d = [(sd[a][k].float() - sd[b][k].float()).abs() for k in sd[a]]",
        "    return {'bit_identical': all(torch.equal(sd[a][k], sd[b][k]) for k in sd[a]),",
        "            'max_abs_diff': max(float(x.max()) for x in d),",
        "            'mean_abs_diff': sum(float(x.sum()) for x in d) / sum(x.numel() for x in d)}",
        "print(json.dumps({'plain_a_vs_plain_b': c('plain_a', 'plain_b'), 'plain_a_vs_instrumented': c('plain_a', 'instrumented'),",
        "                  'plain_b_vs_instrumented': c('plain_b', 'instrumented')}))",
        ""])
    (out / "rna" / "compare.py").write_text(code, encoding="utf-8")
    p = wsl(["bash", "-c", f"cd {wsl_path(RNA_ROOT)} && .venv/bin/python {wsl_path(out / 'rna' / 'compare.py')}"])
    rec["rna"] = {"digests": digests, **json.loads(p.stdout.strip().splitlines()[-1])}
    rec["rna"]["verdict"] = verdict(rec["rna"], ic["tolerance_ratio"])
    print("RNA", rec["rna"], flush=True)
    R.write_json(out / "identity_check.json", rec)
    return 0


if __name__ == "__main__":
    rc = main()
    import os

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
