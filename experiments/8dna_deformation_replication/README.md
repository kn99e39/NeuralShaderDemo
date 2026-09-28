# 8DNA deformation replication

This directory contains project-owned adapters for the independent 8DNA
replication study. The official source remains immutable under
`external/8dna26/` and is intentionally ignored by Git.

The experiment is gated:

1. reproduce the released static inference path;
2. audit the representation contract from source;
3. inspect candidate assets without viewing deformation error;
4. only then define and lock a correspondence-valid deformation trajectory.

No deformation result is valid unless the static gate in
`baseline_manifest.json` reports `status: pass`.

## Official dependency

- Repository: <https://github.com/lwwu2/8dna26>
- Pinned commit: `4a2157ca24e506c5ac0831f27d656ecc50a64f07`
- Expected checkout: `external/8dna26/`
- Released scene archive SHA-256:
  `e60653896a978fb7a386c09c85f477cf98054082f41aa6745ea8d2538e0d1d19`
- Released checkpoint archive SHA-256:
  `ef134a50bfade0431c97a71fd224dd833a56faeb79bf2ae14312f3a0be0bb13c`

## Commands

From the repository root, using the isolated upstream environment:

```text
external\8dna26\.venv\Scripts\python.exe experiments\8dna_deformation_replication\probe_environment.py
external\8dna26\.venv\Scripts\python.exe experiments\8dna_deformation_replication\inspect_assets.py
external\8dna26\.venv\Scripts\python.exe experiments\8dna_deformation_replication\render_official_baseline.py --mode smoke
external\8dna26\.venv\Scripts\python.exe experiments\8dna_deformation_replication\render_official_baseline.py --mode official
```

The official mode follows the released notebook's `seal`, `scene2`, 256 x
256, 256 spp, seed-0 path. The smoke mode is an implementation check only and
cannot close the baseline gate.
