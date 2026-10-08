"""Evidence chain for worklog 30: run every stage once, in order, from a clean tree.

    pwsh ../neural_recompute_cost/windows/launch_detached.ps1 sro_chain <log dir> ../selective_refit_oracle/sro_chain.py --root <dir>

Stages are subprocesses of this interpreter (the run.ps1 environment is inherited); each writes
<root>/chain/<stage>.log and, on success, <stage>.done, so a rerun resumes after the last
completed stage.  The chain stops at the first failing stage.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def stages(root: Path) -> list[tuple[str, list[str]]]:
    o = root / "oracle"
    mask = str(o / "oracle_cells.npz")
    st = [
        ("oracle", ["sro_oracle.py", "--out", str(o)]),
        ("oracle_repeat", ["sro_oracle.py", "--out", str(root / "oracle_repeat")]),
        ("oracle_seed2", ["sro_oracle.py", "--out", str(root / "oracle_seed2"), "--seed", "30031"]),
        ("tests", ["tests/test_sro.py", "--out", str(root / "tests"), "--mask-dirs", str(o), str(root / "oracle_repeat")]),
        ("synthetic", ["tests/test_sro_synthetic.py", "--out", str(root / "tests")]),
        ("regions", ["sro_regions.py", "--out", str(root / "regions")]),
        ("train_E_mask_cost", ["sro_train.py", "--arm", "E_mask", "--steps", "256", "--mask", mask, "--out", str(root / "cost" / "E_mask_s0")]),
    ]
    for arm, seed in (("C", 0), ("D", 0), ("E", 0), ("S", 0), ("B", 0), ("C", 1), ("D", 1), ("E", 1)):
        run = root / "runs" / f"{arm}_s{seed}"
        st.append((f"train_{arm}_s{seed}", ["sro_train.py", "--arm", arm, "--seed", str(seed), "--mask", mask, "--out", str(run)]))
        st.append((f"render_{arm}_s{seed}", ["sro_render.py", "--run", str(run)]))
    st.append(("metrics", ["sro_metrics.py", "--root", str(root)]))
    return st


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    chain = root / "chain"
    chain.mkdir(parents=True, exist_ok=True)
    ev = (chain / "chain_events.jsonl").open("a", encoding="utf-8")
    for name, argv in stages(root):
        if (chain / f"{name}.done").exists():
            continue
        t0 = time.time()
        ev.write(json.dumps({"t": t0, "stage": name, "event": "launch", "argv": argv}) + "\n"); ev.flush()
        with (chain / f"{name}.log").open("w", encoding="utf-8") as log:
            rc = subprocess.run([sys.executable, "-X", "faulthandler", str(HERE / argv[0]), *argv[1:]],
                                stdout=log, stderr=subprocess.STDOUT).returncode
        ev.write(json.dumps({"t": time.time(), "stage": name, "event": "exit", "rc": rc, "seconds": time.time() - t0}) + "\n"); ev.flush()
        if rc != 0:
            print(f"stage {name} failed rc={rc}", flush=True)
            return rc
        (chain / f"{name}.done").write_text(str(time.time()))
        print(f"stage {name} ok {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
