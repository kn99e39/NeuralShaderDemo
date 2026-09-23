# Worklog — Rain F3 Refit Runtime Environment Block (2026-09-23)

## Scope

This session attempted the deferred F3 refit of the evaluated Rain deformation
case.  The corpus and train configuration were already prepared in the prior
session; no corpus, checkpoint, or source asset was altered here.

## Reproduction result

The official RNA `NeuralSurfaceDataModule` was inspected before another full
training attempt.  Its loader materializes all H5 channels through
`ops_device` and retains each processed slice on `device`.  The committed F3
configuration therefore matches the static baselines' GPU-resident policy.

To distinguish GPU residency from an F3-only configuration error, the exact
F3 H5 pair was then initialized read-only with both `device` and `ops_device`
set to CPU.  This preserves the corpus, RNA architecture, and later
batch-to-trainer behavior while avoiding loader CUDA allocation.  The process
ended before its `CPU_STAGED_READY` completion marker, and the Ubuntu WSL
instance restarted: its subsequent uptime was one minute.  Kernel startup
logs reported an unclean prior shutdown and `dxgkio_query_adapter_info`
failures.  The same host-level restart had occurred on the prior GPU-resident
initialization and training attempts.

## Interpretation and limit

The F3 refit is not yet measured.  The block is a WSL runtime/session failure,
not a failed RNA quality result and not evidence against refitting.  No
repeated full-training attempts were made after this independent CPU-staged
reproduction.  Existing Rain primary evidence remains classified only as
**Weakened** in its fixed front-primary regime; it is not promoted to a global
falsification conclusion.

## Next condition

Resume the frozen F3 refit once WSL can retain a Python data-loader process
through the `CPU_STAGED_READY` marker (or run the same committed corpus/config
on a stable Linux/CUDA host).  Then apply the predeclared same-state refit
delta before changing the Rain verdict.
