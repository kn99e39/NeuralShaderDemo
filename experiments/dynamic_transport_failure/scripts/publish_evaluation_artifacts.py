"""Copy human-review visual artifacts into one evaluation directory.

Sources remain authoritative.  This tool makes a byte-for-byte review copy
and records each source path and SHA-256 in ``manifest.json`` beside it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil


VISUAL_EXTENSIONS = {".gif", ".mp4", ".webm", ".png", ".jpg", ".jpeg"}


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/evaluation"),
    )
    parser.add_argument(
        "--copy",
        nargs=2,
        action="append",
        metavar=("SOURCE", "REVIEW_NAME"),
        required=True,
        help="Copy one visual SOURCE using REVIEW_NAME in --output-dir.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for source_string, review_name in args.copy:
        source = pathlib.Path(source_string).resolve()
        destination = output_dir / review_name
        if source.suffix.lower() not in VISUAL_EXTENSIONS:
            raise ValueError(f"not a supported visual artifact: {source}")
        if not source.is_file():
            raise FileNotFoundError(source)
        if pathlib.Path(review_name).name != review_name:
            raise ValueError("REVIEW_NAME must be a filename, not a path")
        if destination.suffix.lower() != source.suffix.lower():
            raise ValueError("REVIEW_NAME must keep the source file extension")
        shutil.copy2(source, destination)
        artifacts.append(
            {
                "review_name": review_name,
                "source": str(source),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )

    (output_dir / "manifest.json").write_text(
        json.dumps({"artifacts": artifacts}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
