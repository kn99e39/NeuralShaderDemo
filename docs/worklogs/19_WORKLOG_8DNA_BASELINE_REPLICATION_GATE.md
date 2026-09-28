# Worklog — 8DNA Baseline Replication Gate (2026-09-28)

## Session question

Can the official 8DNA release be reproduced faithfully on the available host
so that an independent frozen-deformation experiment may begin?

Primary classification:

**BASELINE REPRODUCTION FAILED**

The failure occurred before any image was rendered. Per the batch stop rule,
no deformation adapter, deformation state, GT deformation render, frozen 8DNA
deformation evaluation, or failure classification was produced.

## IMPLEMENTATION FACT

- Read `docs/RESEARCH_CENTRIC_TOPIC.md` and `docs/RESEARCH_ROADMAP.md`.
- The requested `docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md` does not exist
  in this checkout. The batch's explicit role contract was preserved: RNA
  remains the development base; 8DNA is only a scientific replication
  baseline.
- Preserved all RNA history and code. No Rain tuning or RNA reinterpretation
  was performed.
- Cloned the official 8DNA repository to the ignored
  `external/8dna26/` directory and pinned commit
  `4a2157ca24e506c5ac0831f27d656ecc50a64f07`.
- Downloaded and hashed the official released scenes and weights. Hashes and
  byte sizes are recorded in
  `experiments/8dna_deformation_replication/baseline_failure.json`.
- Kept upstream source immutable. All added scripts and documentation are
  project-owned under `experiments/8dna_deformation_replication/`.
- Created an isolated Python 3.11.9 environment with the upstream versions:
  PyTorch 2.3.1, Mitsuba 3.5.1, and Lightning 2.1.3. On Windows, the
  unqualified PyTorch requirement installed a CPU wheel, so the same version's
  official CUDA 12.1 wheel was used. Ninja and NumPy 1.26.4 were required to
  clear missing-build-tool and binary-ABI confounders.
- Audited the model, integrator, normalizing flow, encodings, dataset sampler,
  and light-transport sampler. The result is
  `experiments/8dna_deformation_replication/8DNA_REPRESENTATION_AUDIT.md`.
- Inspected two released candidates using only geometry/material/contract
  suitability: `seal` and `teaset`. The geometry inventory is recorded in
  `asset_audit.json`.

## MEASUREMENT

Host/runtime facts:

- GPU: NVIDIA GeForce RTX 5080, 16,303 MiB, compute capability 12.0.
- Driver: 596.49.
- Official model version: PyTorch 2.3.1+cu121.
- Architectures advertised by that wheel: SM 5.0 through SM 9.0; no SM 12.0.
- Final process exit: `-1073740791` (`0xC0000409`) during official
  model/integrator construction.
- Rendered pixels: zero.
- Static image metrics: unavailable.
- Deformation metrics: unavailable.

The official demo configuration that was gated was `seal`, `scene2`, 256 x
256, 256 spp, seed 0, `neuralvolpath`, maximum depth 10. A 64 x 64, 1-spp smoke
run was used before attempting the expensive configuration. It never reached
rendering, so the expensive run was correctly not started.

## OBSERVATION

The first failures were ordinary environment omissions: CPU-only PyTorch from
the unqualified Windows requirement, missing Ninja, an inactive MSVC shell, and
NumPy 2.x ABI incompatibility. Those were removed without changing upstream
source or model weights.

After those repairs, the official import/JIT path reached model construction,
but the process aborted. PyTorch consistently warned that the RTX 5080's SM
12.0 capability is incompatible with the released PyTorch 2.3.1 binary, whose
newest compiled architecture is SM 9.0. Building the small 8DNA extension with
9.0+PTX did not make the rest of PyTorch/Mitsuba compatible.

No visible output, baseline metric, or neural behavior was observed.

## INTERPRETATION

This is a baseline runtime-compatibility failure on the available host. It is
not evidence that 8DNA's static representation is inaccurate, that frozen
deformation is invalid, or that the cross-model deformation phenomenon does or
does not recur.

The representation audit nevertheless identifies a plausible future controlled
path. 8DNA persistently learns a canonical conditional distribution over
internal asset transport, while current external emitter/visibility work is
recomputed. Directly querying deformed positions would be a spatial mismatch;
a valid surface experiment needs an explicit canonical point/direction map.

Before neural output was viewed, `teaset` was geometry-only preferred over
`seal`: its four separate fixed-topology conductor meshes permit exact per-part
rigid correspondence and meaningful relative-configuration change. The seal's
heterogeneous volume would also require a validated material-volume warp.
This selection is provisional and was not turned into an experiment because
the baseline gate failed.

## Negative evidence

- No deformation failure was observed.
- No negative deformation result was observed.
- No coordinate/indexing artifact was observed because no image was rendered.
- No physical GT behavior was measured.
- No comparison with RNA behavior is scientifically available.

Relative to the RNA result, this session adds no cross-model evidence. The Rain
fixed-target result remains **PHYSICAL EFFECT TOO WEAK**; it is neither changed
to RNA robust nor generalized to 8DNA.

## UNRESOLVED QUESTION

Can the exact pinned release reproduce on a compatible SM <= 9.0 GPU with the
same PyTorch 2.3.1/Mitsuba 3.5.1 stack? That is the next gate. Using a newer
PyTorch/Mitsuba build on this RTX 5080 would be a separate compatibility port,
not a faithful official-environment reproduction, and should be declared and
validated as such before it can support the requested study.

If the baseline passes on compatible hardware, the next session should:

1. export the official seal baseline and matched PT reference;
2. verify a canonical static `teaset` checkpoint/render;
3. lock a geometry-only relative-part trajectory before frozen neural results;
4. implement and test exact per-part canonical correspondence; and
5. only then render physical GT and frozen 8DNA states.

## Reproducibility and repository state

- Project base commit used: `e2056e0c92017d6d1fe7303a27e699a9a4560db2`.
- Official 8DNA commit:
  `4a2157ca24e506c5ac0831f27d656ecc50a64f07`.
- The final project state is intentionally uncommitted in this session; the
  exact base commit is reported rather than claiming a nonexistent final
  experiment commit.
- An unrelated pre-existing/user-owned modification to
  `experiments/dynamic_transport_failure/scripts/capture_wsl_gpu_pv_failure.ps1`
  was observed and left untouched.
- No evaluation folder was created because the session produced no
  human-reviewable image, GIF, or video artifact.
