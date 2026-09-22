"""Fail closed until a high-quality static RNA baseline is explicitly accepted."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


REQUIRED_PRESERVE = {
    "materials", "textures", "topology", "surface_identity", "uv", "camera",
    "lighting", "renderer", "rna_architecture", "canonical_checkpoint",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--allow-static-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    errors: list[str] = []
    if manifest.get("static_gate") != "passed":
        errors.append("static_gate must be 'passed' before dynamic evaluation")
    if not manifest.get("canonical_checkpoint"):
        errors.append("canonical_checkpoint is required and must remain frozen")
    if set(manifest.get("preserve", ())) != REQUIRED_PRESERVE:
        errors.append("preserve contract is incomplete or contains undeclared controls")
    if manifest.get("change_only") != ["pose_geometry"]:
        errors.append("only pose_geometry may change in this batch")
    results_root = pathlib.PurePosixPath(manifest.get("results_root", ""))
    if not str(results_root).startswith("results/batch1_hq_dynamic_failure/"):
        errors.append("results_root must be inside the designated ignored result tree")
    if args.allow_static_only:
        errors = [error for error in errors if not error.startswith("static_gate") and not error.startswith("canonical_checkpoint")]
    if errors:
        print("manifest rejected:")
        print("\n".join(f"- {error}" for error in errors))
        return 2
    print(f"manifest accepted for {manifest['asset_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
