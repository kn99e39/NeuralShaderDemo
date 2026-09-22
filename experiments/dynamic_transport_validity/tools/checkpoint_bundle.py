"""Create Git-friendly checkpoint chunks and losslessly reconstruct them.

Checkpoint chunks are intentionally small (50 MiB by default), each is
SHA-256-addressed in a manifest, and reconstruction verifies both every chunk
and the reassembled original.  A checkpoint must be unpacked before it can be
passed to RNA; chunks are an archival transport format, not a runtime format.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil


BUFFER_BYTES = 4 * 1024 * 1024


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(BUFFER_BYTES):
            digest.update(block)
    return digest.hexdigest()


def pack(source: pathlib.Path, output: pathlib.Path, chunk_mib: int) -> pathlib.Path:
    if not source.is_file():
        raise FileNotFoundError(source)
    chunk_bytes = chunk_mib * 1024 * 1024
    if chunk_mib <= 0 or chunk_mib >= 100:
        raise ValueError("--chunk-mib must be between 1 and 99 for GitHub-safe chunks")
    output.mkdir(parents=True, exist_ok=True)
    parts = []
    with source.open("rb") as stream:
        index = 0
        while block := stream.read(chunk_bytes):
            name = f"{source.name}.part{index:03d}"
            part = output / name
            part.write_bytes(block)
            parts.append({"file": name, "bytes": len(block), "sha256": sha256(part)})
            index += 1
    manifest = {
        "format": "neuralshader-checkpoint-bundle-v1",
        "original_filename": source.name,
        "original_bytes": source.stat().st_size,
        "original_sha256": sha256(source),
        "chunk_bytes": chunk_bytes,
        "parts": parts,
    }
    path = output / f"{source.name}.manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def load_manifest(path: pathlib.Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != "neuralshader-checkpoint-bundle-v1":
        raise ValueError(f"unsupported bundle manifest: {path}")
    return data


def verify(manifest_path: pathlib.Path) -> dict:
    manifest = load_manifest(manifest_path)
    for entry in manifest["parts"]:
        part = manifest_path.parent / entry["file"]
        if not part.is_file() or part.stat().st_size != entry["bytes"] or sha256(part) != entry["sha256"]:
            raise ValueError(f"invalid checkpoint part: {part}")
    return manifest


def unpack(manifest_path: pathlib.Path, output: pathlib.Path) -> pathlib.Path:
    manifest = verify(manifest_path)
    output.mkdir(parents=True, exist_ok=True)
    target = output / manifest["original_filename"]
    temporary = target.with_suffix(target.suffix + ".partial")
    with temporary.open("wb") as destination:
        for entry in manifest["parts"]:
            with (manifest_path.parent / entry["file"]).open("rb") as source:
                shutil.copyfileobj(source, destination, BUFFER_BYTES)
    if temporary.stat().st_size != manifest["original_bytes"] or sha256(temporary) != manifest["original_sha256"]:
        temporary.unlink(missing_ok=True)
        raise ValueError("reassembled checkpoint hash does not match manifest")
    temporary.replace(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    pack_parser = commands.add_parser("pack")
    pack_parser.add_argument("--source", required=True, type=pathlib.Path)
    pack_parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    pack_parser.add_argument("--chunk-mib", type=int, default=50)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--manifest", required=True, type=pathlib.Path)
    unpack_parser = commands.add_parser("unpack")
    unpack_parser.add_argument("--manifest", required=True, type=pathlib.Path)
    unpack_parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()
    if args.command == "pack":
        print(pack(args.source, args.output_dir, args.chunk_mib))
    elif args.command == "verify":
        print(json.dumps(verify(args.manifest), indent=2))
    else:
        print(unpack(args.manifest, args.output_dir))


if __name__ == "__main__":
    main()
