"""Run the relational-residual prototype end to end (Windows driver).

    windows/run.ps1 ../relational_residual_prototype/rrp_run.py --run-id v1_<commit> [--worklog 28]
    windows/run.ps1 ../relational_residual_prototype/rrp_run.py --run-id smoke --smoke

Stages, strictly in order on the RTX 5080:
  1. rrp_probes.py        (Windows, Mitsuba)  current-geometry probe state, references, stable flags
  2. wsl/rrp_features.py  (WSL, RNA venv)     frozen RNA base and persistent features
  3. wsl/rrp_train.py     (WSL, RNA venv)     both branches x seeds, predictions, latency
  4. rrp_eval.py          (Windows)           historical metrics, accounting, exports
Evidence runs need a clean tree and a run id ending in the commit; --smoke trains
for 20 steps only (an untrained model: no information about any hold-out), allows a
dirty tree, and is never evidence.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "neural_recompute_cost"))

import nrc_records as R  # noqa: E402
from nrc_wsl import RNA_ROOT, run_detached, wsl_path  # noqa: E402

ROOT = HERE.parents[1]
EXP8 = ROOT / "experiments" / "8dna_deformation_replication"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--worklog", default=None)
    ap.add_argument("--from-stage", type=int, default=1)
    args = ap.parse_args()
    git = R.git_state(ROOT)
    R.require_clean(git, allow_dirty=args.smoke)
    if not args.smoke and not args.run_id.endswith(git["commit"][:7]):
        raise SystemExit(f"evidence run id must end with the commit ({git['commit'][:7]})")
    run = ROOT / "results" / "relational_residual_prototype" / args.run_id
    logs = run / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    ev = R.EventLog(run / "run_events.jsonl", "driver")
    ev.emit("run_start", git=git, smoke=args.smoke)
    allow = ["--allow-dirty"] if args.smoke else []
    py = sys.executable

    def local(n, name, argv):
        if args.from_stage > n:
            return
        ev.begin("stage", key=name)
        with (logs / f"{name}.log").open("w", encoding="utf-8") as fh:
            rc = subprocess.run([py, "-X", "faulthandler", *argv], cwd=EXP8, stdout=fh, stderr=subprocess.STDOUT).returncode
        ev.end("stage", key=name, exit_code=rc)
        print(name, "rc", rc, flush=True)
        if rc:
            raise SystemExit(f"{name} failed; see {logs / (name + '.log')}")

    def remote(n, name, argv):
        if args.from_stage > n:
            return
        ev.begin("stage", key=name)
        rc = run_detached(name, logs, [".venv/bin/python", *argv], RNA_ROOT)
        ev.end("stage", key=name, exit_code=rc)
        print(name, "rc", rc, flush=True)
        if rc:
            raise SystemExit(f"{name} failed; see {logs}/{name}.std*.log")

    local(1, "probes", [str(HERE / "rrp_probes.py"), "--out", str(run), *allow])
    remote(2, "features", [wsl_path(HERE / "wsl" / "rrp_features.py"), "--run", wsl_path(run)])
    remote(3, "train", [wsl_path(HERE / "wsl" / "rrp_train.py"), "--run", wsl_path(run), "--project-commit", git["commit"],
                        *(["--smoke-steps", "20"] if args.smoke else [])])
    local(4, "eval", [str(HERE / "rrp_eval.py"), "--run", str(run), *allow, *(["--worklog", args.worklog] if args.worklog else [])])
    ev.emit("run_done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
