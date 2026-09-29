# 8DNA replication (scientific replication baseline)

Project-owned adapters for the independent 8DNA replication study. The
official source stays unmodified under `external/8dna26/` (ignored by Git).
8DNA is a scientific replication baseline here, not the development base
(`docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md`).

## Official dependency

- Repository: <https://github.com/lwwu2/8dna26>
- Pinned commit: `4a2157ca24e506c5ac0831f27d656ecc50a64f07`
- Released scene archive SHA-256:
  `e60653896a978fb7a386c09c85f477cf98054082f41aa6745ea8d2538e0d1d19`
- Released checkpoint archive SHA-256:
  `ef134a50bfade0431c97a71fd224dd833a56faeb79bf2ae14312f3a0be0bb13c`

## Runtime (RTX 5080, native Windows)

`windows/setup_env.ps1` builds `external/8dna26/.venv-cu128`. `windows/run.ps1
<script.py> ...` runs an entry point with MSVC 14.38 and CUDA 12.8, so torch
JIT-builds the unmodified upstream extension for sm_120. Deviations from the
upstream environment, none of which touches model code or weights:

- torch 2.8.0+cu128 instead of 2.3.1, whose wheels have no sm_120 kernels;
- DrJit `VCallRecord` on for neural renders (`ednalib.py` explains the
  wavefront-dispatch hang; `probe_vcall_dispatch.py` reproduces it);
- native Windows instead of Linux; under WSL, DrJit 0.4.6 cannot load OptiX
  (`wsl/README.md`).

## Pipeline

| Step | Script | Output (under `results/8dna_replication/`) |
|---|---|---|
| runtime smoke | `smoke_runtime.py` | `smoke/` |
| static baseline vs PT reference | `run_static_baseline.py` | `static_baseline/<tag>/` |
| geometry of candidate states | `measure_teaset_geometry.py` | `gt_design/geometry_*.json` |
| GT-only trajectory design + ROIs | `teaset_gt_states.py` | `gt_design/<rev>/` |
| correspondence tests | `test_correspondence.py` | `correspondence_tests/` |
| locked frozen evaluation | `teaset_frozen_eval.py` | `frozen/teaset_locked/` |

`teaset_parts.py` holds the part-rigid configurations and the correspondence
adapter (`PartRigidAsset`). `protocol/` holds the GT design revisions and the
locked protocol. `teaset_frozen_eval.py` refuses to run on a dirty tree.

The earlier files `render_official_baseline.py`, `probe_environment.py`,
`baseline_*.json` and `environment_probe.json` are the worklog-19 native
torch 2.3.1 attempt and are kept unchanged as its record.
