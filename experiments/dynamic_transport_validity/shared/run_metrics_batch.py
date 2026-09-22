"""Evaluate a JSON list of metric jobs while reusing the LPIPS network."""

from __future__ import annotations

import argparse
import json
import pathlib
from types import SimpleNamespace

from metrics import evaluate


def path_or_none(value):
    return pathlib.Path(value) if value else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=pathlib.Path)
    args = parser.parse_args()
    jobs = json.loads(args.spec.read_text(encoding="utf-8"))
    results = []
    for job in jobs:
        values = dict(job)
        for key in (
            "reference", "prediction", "canonical_visibility", "current_visibility",
            "canonical_position_reference", "canonical_position_current", "output",
            "review", "mask_output_dir",
        ):
            values[key] = path_or_none(values.get(key))
        values.setdefault("lpips", True)
        values.setdefault("correspondence_tolerance", 0.003)
        result = evaluate(SimpleNamespace(**values))
        results.append(result)
        print(json.dumps(result))
    summary = args.spec.with_name(args.spec.stem + "_results.json")
    summary.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
