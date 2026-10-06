"""Evidence chain for the neural full-recompute cost batch (worklog 27).

    windows/run.ps1 ../neural_recompute_cost/nrc_chain.py --run-id v1_<commit>   [--smoke] [--only ...]

Runs, strictly one at a time on the RTX 5080 (16 GiB; concurrent jobs would
slow each other and corrupt the timing):

  timed stages (the cost being measured; protocol "tracks")
    8dna_train     8DNA rebuild on T3: historical launcher + upstream train.py,
                   30 epochs, seed 9; ends with the load_asset-readable export
    rna_h5_train   RNA T3 training targets (rna_bridge.generate_h5, 200 views)
    rna_h5_val     RNA T3 validation targets (40 views)
    rna_train      RNA training in WSL (official scripts/train.py, 250 epochs, seed 0)
  untimed stages (evaluation; never part of the latency)
    rna_snapshot_copy, eval_8dna, eval_rna, inference_timing

Each stage's launch and exit are chain events (Windows wall clock); each
timed process writes its own phase events.  A stage whose done-marker exists
is skipped, so an interrupted chain can resume -- but a timed stage that was
interrupted must be rerun from scratch (its directory is refused if it holds
partial output), because a resumed training run is not a clean timing.

Evidence runs refuse a dirty project tree.  --smoke uses tiny settings
(declared in the protocol's "smoke" block), allows a dirty tree, and its
outputs are never evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "8dna_deformation_replication"))

import nrc_records as R  # noqa: E402
from nrc_wsl import WSL_ROOT, clock_offset, launch_detached, wait_status, wsl, wsl_path  # noqa: E402

ROOT = HERE.parents[1]
EXP8 = ROOT / "experiments" / "8dna_deformation_replication"
RES8 = ROOT / "results" / "8dna_replication"
RNA_ROOT = ROOT / "external" / "relightable-neural-assets"
PROTOCOL = HERE / "protocol" / "nrc_t3_recompute_v1.json"
def host_record() -> dict:
    import platform

    def q(args):
        try:
            return subprocess.run(args, capture_output=True, text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    return {
        "os": platform.platform(),
        "os_caption": q(["pwsh", "-NoProfile", "-Command", "(Get-CimInstance Win32_OperatingSystem).Caption"]),
        "cpu": q(["pwsh", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"]),
        "ram_gb": q(["pwsh", "-NoProfile", "-Command", "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)"]),
        "gpu": q(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "hostname": platform.node(),
        "wsl_kernel": (wsl(["uname", "-r"]).stdout or "").strip() or None,
    }


def derive_rna_inputs(proto: dict, run: Path, smoke: bool) -> dict:
    """Derived RNA protocol and config: the locked ones with only output locations changed."""
    rdir = run / "rna"
    rdir.mkdir(parents=True, exist_ok=True)
    src_p = EXP8 / proto["tracks"]["rna_common_light"]["dataset"]["locked_protocol"]
    dp = json.loads(src_p.read_text(encoding="utf-8"))
    changed = {"rna_dataset_dir": os.path.relpath(rdir / "datasets", RES8).replace("\\", "/")}
    if smoke:
        changed.update({f"rna_dataset.{k}": v for k, v in proto["smoke"]["rna_dataset"].items()})
        dp["rna_dataset"].update(proto["smoke"]["rna_dataset"])
    dp["rna_dataset_dir"] = changed["rna_dataset_dir"]
    dp["derived_by"] = {"script": "experiments/neural_recompute_cost/nrc_chain.py", "from": R_rel(src_p), "changed": changed}
    dpp = rdir / "derived_protocol.json"
    R.write_json(dpp, dp)

    src_c = EXP8 / proto["tracks"]["rna_common_light"]["training"]["locked_config"]
    text = src_c.read_text(encoding="utf-8")
    name = proto["tracks"]["rna_common_light"]["training"]["run_name"] + ("-smoke" if smoke else "")
    ds_rel = os.path.relpath(rdir / "datasets", RNA_ROOT).replace("\\", "/")
    new = re.sub(r'^name: .*$', f'name: "{name}"', text, count=1, flags=re.M)
    new = re.sub(r'^training_dataset: .*$', f'training_dataset: "{ds_rel}/teaset_T3_train.h5"', new, count=1, flags=re.M)
    new = re.sub(r'^validation_dataset: .*$', f'validation_dataset: "{ds_rel}/teaset_T3_val.h5"', new, count=1, flags=re.M)
    if smoke:
        new = re.sub(r'^num_epochs: .*$', f'num_epochs: {proto["smoke"]["rna_epochs"]}', new, count=1, flags=re.M)
    diff = [(a, b) for a, b in zip(text.splitlines(), new.splitlines()) if a != b]
    allowed = ("name:", "training_dataset:", "validation_dataset:") + (("num_epochs:",) if smoke else ())
    if len(text.splitlines()) != len(new.splitlines()) or any(not a.startswith(allowed) for a, _ in diff):
        raise SystemExit(f"derived RNA config changes more than its output locations: {diff}")
    cfg = rdir / "config.yml"
    cfg.write_text(new, encoding="utf-8")
    return {"protocol": dpp, "protocol_rel": os.path.relpath(dpp, EXP8).replace("\\", "/"), "config": cfg,
            "config_diff": diff, "run_name": name}


def R_rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(ROOT)).replace("\\", "/")


class Chain:
    def __init__(self, run: Path, smoke: bool):
        self.run = run
        self.smoke = smoke
        self.logs = run / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.ev = R.EventLog(run / "chain_events.jsonl", "chain")

    def stage(self, name: str, done: Path | None, argv: list[str], cwd: Path, timed: bool, fresh: list[Path] = ()):
        if done is not None and done.exists():
            print(f"SKIP {name} (have {R_rel(done)})", flush=True)
            self.ev.emit("stage_skipped", stage_name=name, done=R_rel(done))
            return
        if timed:
            for d in fresh:
                if d.exists() and any(d.iterdir()):
                    raise SystemExit(f"{name}: {R_rel(d)} holds partial output from an interrupted run; "
                                     "a timed stage must start clean (move it aside by hand)")
        print(f"START {name}", flush=True)
        self.ev.begin("stage", key=name, timed=timed, argv=argv, cwd=R_rel(cwd))
        with (self.logs / f"{name}.log").open("w", encoding="utf-8") as fh:
            rc = subprocess.run(argv, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT).returncode
        self.ev.end("stage", key=name, exit_code=rc)
        print(f"END {name} rc={rc}", flush=True)
        if rc != 0 or (done is not None and not done.exists()):
            raise SystemExit(f"stage {name} failed (rc={rc}); see {R_rel(self.logs / (name + '.log'))}")

    def wsl_stage(self, name: str, done: Path, job_argv: list[str], fresh: list[Path]):
        """A long WSL job, detached from this client (AGENTS.md), polled through its status file."""
        if done.exists():
            print(f"SKIP {name} (have {R_rel(done)})", flush=True)
            self.ev.emit("stage_skipped", stage_name=name, done=R_rel(done))
            return
        for d in fresh:
            if d.exists() and any(d.iterdir()):
                raise SystemExit(f"{name}: {R_rel(d)} holds partial output; a timed stage must start clean")
        self.ev.emit("clock_offset", when="before", **clock_offset())
        print(f"START {name} (WSL, detached)", flush=True)
        self.ev.begin("stage", key=name, timed=True, argv=job_argv, cwd=R_rel(RNA_ROOT), launcher="wsl_run_detached.sh")
        try:
            status = launch_detached(name, self.logs, job_argv)
        except RuntimeError as e:
            self.ev.end("stage", key=name, exit_code=-1)
            raise SystemExit(str(e))
        rc = wait_status(status)
        text = status.read_text(encoding="utf-8").strip()
        self.ev.end("stage", key=name, exit_code=rc, status=text, poll_interval_s=5)
        self.ev.emit("clock_offset", when="after", **clock_offset())
        print(f"END {name} {text}", flush=True)
        if rc != 0 or not done.exists():
            raise SystemExit(f"stage {name} failed ({text}); see {R_rel(self.logs)}/{name}.std*.log")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", nargs="*", help="run only these stages (others are skipped, not marked done)")
    args = ap.parse_args()
    proto = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    git = R.git_state(ROOT)
    R.require_clean(git, allow_dirty=args.smoke)
    if not args.smoke and not args.run_id.endswith(git["commit"][:7]):
        raise SystemExit(f"evidence run id must end with the commit it runs at ({git['commit'][:7]})")
    run = ROOT / "results" / "neural_recompute_cost" / args.run_id
    run.mkdir(parents=True, exist_ok=True)
    ch = Chain(run, args.smoke)
    py = sys.executable
    want = lambda s: not args.only or s in args.only
    ch.ev.emit("chain_start", git=git, smoke=args.smoke, run_id=args.run_id, protocol=R_rel(PROTOCOL),
               protocol_sha256=__import__("hashlib").sha256(PROTOCOL.read_bytes()).hexdigest(),
               host=host_record(), python=py)
    allow = ["--allow-dirty"] if args.smoke else []
    t8 = proto["tracks"]["8dna_envmap_primary"]["training"]

    # --- 8DNA rebuild (timed) ----------------------------------------------
    d8 = run / "8dna"
    if want("8dna_train"):
        launcher = ["--state", t8["state"], "--protocol", t8["states_protocol"], "--device", "0",
                    "--max_epochs", str(proto["smoke"]["8dna_epochs"] if args.smoke else t8["max_epochs"]),
                    "--seed", str(t8["seed"]), "--log_path", str(d8 / "train"), "--experiment_name", t8["experiment_name"]]
        if args.smoke:
            launcher += proto["smoke"]["8dna_overrides"]
        ch.stage("8dna_train", d8 / f"{t8['experiment_name']}_released_layout.ckpt",
                 [py, "-X", "faulthandler", str(HERE / "nrc_8dna_train_timed.py"), "--events", str(d8 / "events.jsonl"),
                  "--snapshots", str(d8 / "snapshots"), "--export", str(d8 / f"{t8['experiment_name']}_released_layout.ckpt"),
                  *allow, "--", *launcher],
                 cwd=EXP8, timed=True, fresh=[d8 / "train", d8 / "snapshots"])

    # --- RNA rebuild (timed) -----------------------------------------------
    rin = derive_rna_inputs(proto, run, args.smoke)
    ch.ev.emit("rna_derived_inputs", protocol=R_rel(rin["protocol"]), config=R_rel(rin["config"]), config_diff=rin["config_diff"])
    rd = run / "rna"
    for split in ("train", "val"):
        if want(f"rna_h5_{split}"):
            ch.stage(f"rna_h5_{split}", rd / "datasets" / f"teaset_T3_{split}.cameras.json",
                     [py, "-X", "faulthandler", str(HERE / "nrc_rna_h5_timed.py"), "--events", str(rd / "events.jsonl"),
                      "--protocol", rin["protocol_rel"], "--state", "T3", "--split", split, *allow],
                     cwd=EXP8, timed=True)
    tr = proto["tracks"]["rna_common_light"]["training"]
    snap_native = f"/root/nrc27/{args.run_id}/snapshots"
    ckpt_dir = rd / "ckpt"
    if want("rna_train"):
        wsl(["bash", "-c", f"rm -rf {snap_native} && mkdir -p {snap_native}"])
        ch.wsl_stage("rna_train", ckpt_dir / rin["run_name"] / "version_0" / "checkpoints" / "last.ckpt",
                     [".venv/bin/python", wsl_path(HERE / "wsl" / "nrc_rna_train_timed.py"),
                      "--events", wsl_path(rd / "events_train.jsonl"), "--snapshots", snap_native,
                      "--snapshot-every", str(1 if args.smoke else tr["snapshot_every"]), "--project-commit", git["commit"],
                      "--", "--config", wsl_path(rin["config"]), "--checkpoint_dir", wsl_path(ckpt_dir),
                      "--seed", str(tr["seed"])],
                     fresh=[ckpt_dir])

    # --- untimed: copy RNA snapshots out of the WSL-native directory -----------
    if want("rna_snapshot_copy"):
        dst = rd / "snapshots"
        if not dst.exists():
            ch.ev.begin("stage", key="rna_snapshot_copy", timed=False)
            p = wsl(["bash", "-c", f"mkdir -p {wsl_path(dst)}.tmp && cp {snap_native}/*.ckpt {wsl_path(dst)}.tmp/ && "
                                   f"mv {wsl_path(dst)}.tmp {wsl_path(dst)} && rm -rf {snap_native}"])
            ch.ev.end("stage", key="rna_snapshot_copy", exit_code=p.returncode)
            if p.returncode:
                raise SystemExit(f"snapshot copy failed: {p.stderr}")

    # --- untimed evaluation -------------------------------------------------
    smoke = ["--smoke"] if args.smoke else []
    if want("eval_8dna"):
        ch.stage("eval_8dna", run / "eval" / "8dna_eval.json",
                 [py, "-X", "faulthandler", str(HERE / "nrc_eval_8dna.py"), "--run", str(run), *smoke, *allow], cwd=EXP8, timed=False)
    if want("eval_rna"):
        ch.stage("eval_rna", run / "eval" / "rna_eval.json",
                 [py, "-X", "faulthandler", str(HERE / "nrc_eval_rna.py"), "--run", str(run), *smoke, *allow], cwd=EXP8, timed=False)
    if want("inference_timing"):
        ch.stage("inference_timing", run / "eval" / "inference_timing.json",
                 [py, "-X", "faulthandler", str(HERE / "nrc_inference_timing.py"), "--run", str(run), *smoke, *allow], cwd=EXP8,
                 timed=False)
    ch.ev.emit("chain_done")
    print("CHAIN DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
