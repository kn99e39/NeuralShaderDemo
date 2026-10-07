"""Run the worklog-29 prototype (Windows driver).

    windows/run.ps1 ../radiometric_relation_state_prototype/rrs_run.py --run-id v1_<commit> [--worklog 29]
    windows/run.ps1 ../radiometric_relation_state_prototype/rrs_run.py --run-id v1_<commit> --oracle [--worklog 29]
    windows/run.ps1 ../radiometric_relation_state_prototype/rrs_run.py --run-id smoke --smoke

Stages: 1 remote light (Windows) -> 2 proxy (WSL) -> 3 train geom20/zero/shuffled/real (WSL) -> 4 eval (Windows).
--oracle (only after the real candidate failed the success test): oracle radiance (Windows),
oracle training (WSL), eval again.  --smoke: 20 training steps (untrained, no information about
any hold-out), dirty tree allowed, never evidence.
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
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--worklog", default=None)
    ap.add_argument("--from-stage", type=int, default=1)
    args = ap.parse_args()
    git = R.git_state(ROOT)
    R.require_clean(git, allow_dirty=args.smoke)
    run = ROOT / "results" / "radiometric_relation_state_prototype" / args.run_id
    if not args.smoke and not args.oracle and not args.run_id.endswith(git["commit"][:7]):
        raise SystemExit(f"evidence run id must end with the commit ({git['commit'][:7]})")
    logs = run / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    ev = R.EventLog(run / "run_events.jsonl", "driver")
    ev.emit("run_start", git=git, smoke=args.smoke, oracle=args.oracle)
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

    smoke = ["--smoke-steps", "20"] if args.smoke else []
    wl = ["--worklog", args.worklog] if args.worklog else []
    if not args.oracle:
        local(1, "remote_light", [str(HERE / "rrs_remote_light.py"), "--out", str(run), *allow])
        remote(2, "proxy", [wsl_path(HERE / "wsl" / "rrs_proxy.py"), "--run", wsl_path(run)])
        remote(3, "train", [wsl_path(HERE / "wsl" / "rrs_train.py"), "--run", wsl_path(run), "--project-commit", git["commit"], *smoke])
        local(4, "eval", [str(HERE / "rrs_eval.py"), "--run", str(run), *allow, *wl])
    else:
        local(1, "oracle_radiance", [str(HERE / "rrs_oracle_radiance.py"), "--out", str(run), *allow,
                                     *(["--spp", "4"] if args.smoke else [])])
        remote(2, "oracle_train", [wsl_path(HERE / "wsl" / "rrs_train.py"), "--run", wsl_path(run), "--project-commit", git["commit"],
                                   "--variants", "oracle", *smoke])
        local(3, "eval_with_oracle", [str(HERE / "rrs_eval.py"), "--run", str(run), *allow, *wl])
    ev.emit("run_done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
