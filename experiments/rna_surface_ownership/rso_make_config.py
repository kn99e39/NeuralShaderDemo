"""Write an arm config: the worklog-22 RNA teaset config with this batch's dataset paths and name.

    python rso_make_config.py --train <h5> --val <h5> --name <name> --out <yml>
"""

import argparse
from pathlib import Path

BASE = Path(__file__).resolve().parents[2] / "experiments" / "8dna_deformation_replication" / "configs" / "rna_teaset_T0.yml"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lines = []
    for line in BASE.read_text().splitlines():
        if line.startswith("name:"):
            line = f'name: "{a.name}"'
        elif line.startswith("training_dataset:"):
            line = f'training_dataset: "{a.train}"'
        elif line.startswith("validation_dataset:"):
            line = f'validation_dataset: "{a.val}"'
        lines.append(line)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
