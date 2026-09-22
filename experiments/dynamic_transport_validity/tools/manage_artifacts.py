"""Keep generated experiment artifacts out of Git and bound local disk growth.

Default mode is a dry run.  Use --apply only after reviewing the printed plan.
Checkpoint pruning retains the highest validation-PSNR checkpoint plus last.ckpt,
so a normal RNA inference workflow remains usable.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil


PSNR = re.compile(r"val_psnr=([0-9.]+)dB")


def checkpoint_keep_set(directory: pathlib.Path) -> set[pathlib.Path]:
    files = list(directory.glob("*.ckpt"))
    keep = {path for path in files if path.name == "last.ckpt"}
    scored = [(float(match.group(1)), path) for path in files if (match := PSNR.search(path.name))]
    if scored:
        keep.add(max(scored, key=lambda item: item[0])[1])
    return keep


def split(path: pathlib.Path, chunk_bytes: int, remove_original: bool) -> list[pathlib.Path]:
    parts: list[pathlib.Path] = []
    with path.open("rb") as source:
        index = 0
        while block := source.read(chunk_bytes):
            part = path.with_name(f"{path.name}.part{index:03d}")
            part.write_bytes(block)
            parts.append(part)
            index += 1
    manifest = path.with_name(f"{path.name}.parts.json")
    manifest.write_text(json.dumps({"original": path.name, "parts": [p.name for p in parts]}, indent=2))
    if remove_original:
        path.unlink()
    return parts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--max-mb", type=int, default=500)
    parser.add_argument("--prune-checkpoints", action="store_true")
    parser.add_argument("--split-large", action="store_true")
    parser.add_argument("--remove-original", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    limit = args.max_mb * 1024 * 1024
    files = [p for p in args.root.rglob("*") if p.is_file()]
    oversized = [p for p in files if p.stat().st_size > limit]
    for path in oversized:
        print(f"oversized: {path} ({path.stat().st_size / 1024 / 1024:.1f} MiB)")
        if args.split_large and args.apply:
            parts = split(path, limit, args.remove_original)
            print(f"split: {path.name} -> {len(parts)} parts")

    if args.prune_checkpoints:
        for directory in sorted({p.parent for p in files if p.suffix == ".ckpt"}):
            keep = checkpoint_keep_set(directory)
            for path in directory.glob("*.ckpt"):
                if path not in keep:
                    print(f"prune: {path}")
                    if args.apply:
                        path.unlink()
    if not args.apply:
        print("dry run; re-run with --apply to change files")


if __name__ == "__main__":
    main()
