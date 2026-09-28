"""Geometry-only audit of predeclared released 8DNA deformation candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "external" / "8dna26"
OUT = ROOT / "experiments" / "8dna_deformation_replication" / "asset_audit.json"

CANDIDATES = {
    "seal": ["scenes/seal/seal.obj", "scenes/seal/jade.vol"],
    "teaset": [
        "scenes/teaset/teaplate.obj",
        "scenes/teaset/teapot2.obj",
        "scenes/teaset/teapot3.obj",
        "scenes/teaset/teapot4.obj",
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_obj(path: Path) -> dict[str, object]:
    vertices: list[tuple[float, float, float]] = []
    faces = 0
    object_names: list[str] = []
    group_names: list[str] = []
    material_names: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("v "):
                fields = line.split()
                vertices.append(tuple(float(value) for value in fields[1:4]))
            elif line.startswith("f "):
                faces += 1
            elif line.startswith("o "):
                object_names.append(line[2:].strip())
            elif line.startswith("g "):
                group_names.append(line[2:].strip())
            elif line.startswith("usemtl "):
                material_names.add(line[7:].strip())
    bounds = None
    if vertices:
        bounds = {
            "min": [min(point[axis] for point in vertices) for axis in range(3)],
            "max": [max(point[axis] for point in vertices) for axis in range(3)],
        }
    return {
        "path": str(path.relative_to(UPSTREAM)).replace("\\", "/"),
        "sha256": sha256(path),
        "vertices": len(vertices),
        "faces": faces,
        "bounds": bounds,
        "object_names": object_names,
        "group_names": group_names,
        "material_names": sorted(material_names),
    }


def main() -> int:
    output: dict[str, object] = {"candidates": {}}
    for candidate, relative_paths in CANDIDATES.items():
        entries: list[dict[str, object]] = []
        for relative in relative_paths:
            path = UPSTREAM / relative
            entry: dict[str, object] = {
                "path": relative,
                "exists": path.is_file(),
            }
            if path.is_file():
                if path.suffix.lower() == ".obj":
                    entry.update(inspect_obj(path))
                else:
                    entry.update({"bytes": path.stat().st_size, "sha256": sha256(path)})
            entries.append(entry)
        output["candidates"][candidate] = entries

    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
