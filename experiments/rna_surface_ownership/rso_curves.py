"""Training curves of every arm from the Lightning TensorBoard logs (RNA venv; no GPU).

    python rso_curves.py --run <run dir>/<CFG> --out <curves.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    res = {}
    for ev in sorted(Path(args.run).glob("*/tb/*/version_0")):
        arm = ev.parts[-4]
        acc = EventAccumulator(str(ev), size_guidance={"scalars": 0}); acc.Reload()
        res[arm] = {t: [[e.step, e.value] for e in acc.Scalars(t)] for t in acc.Tags()["scalars"]}
    Path(args.out).write_text(json.dumps(res))
    print({a: {t: len(v) for t, v in r.items()} for a, r in res.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
