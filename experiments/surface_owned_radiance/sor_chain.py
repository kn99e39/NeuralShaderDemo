"""Evidence chain for worklog 31: every stage once, in order, from a clean tree (resumable).

    pwsh ../neural_recompute_cost/windows/launch_detached.ps1 sor_chain <log dir> ../surface_owned_radiance/sor_chain.py --root <dir>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def stages(root: Path):
    r = str(root)
    st = [
        ("tests", ["tests/test_sor.py", "--out", str(root / "tests")]),
        ("units", ["sor_units.py", "--root", r]),
        ("heldout", ["sor_heldout.py", "--root", r]),
        ("pretrain_T0", ["sor_arm.py", "--arm", "T0", "--root", r]),
        ("oracle", ["sor_oracle.py", "--units", str(root / "units" / "units.npz"), "--out", str(root / "oracle")]),
        ("oracle_replicate", ["sor_oracle.py", "--units", str(root / "units" / "units.npz"), "--out", str(root / "oracle_replicate"), "--seed", "31032"]),
    ]
    for arm, seed in (("C", 0), ("E", 0), ("DG", 0), ("E5", 0), ("B", 0), ("U", 0), ("C", 1), ("E", 1)):
        st.append((f"arm_{arm}_s{seed}", ["sor_arm.py", "--arm", arm, "--root", r, "--seed", str(seed)]))
    st.append(("metrics", ["sor_metrics.py", "--root", r]))
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
