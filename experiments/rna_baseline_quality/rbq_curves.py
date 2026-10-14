"""Training curves from Lightning TensorBoard event directories (RNA venv; no GPU).

    python rbq_curves.py --out <json> name=<dir containing events.out.tfevents.*> ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    res = {}
    for spec in args.runs:
        name, d = spec.split("=", 1)
        acc = EventAccumulator(d, size_guidance={"scalars": 0}); acc.Reload()
        res[name] = {t: [[e.step, e.value] for e in acc.Scalars(t)] for t in acc.Tags()["scalars"]}
    Path(args.out).write_text(json.dumps(res))
    print({a: {t: len(v) for t, v in r.items()} for a, r in res.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
