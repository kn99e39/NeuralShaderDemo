# Worklog — Batch 2 Non-Rigid Study and Artifact Policy (2026-09-22)

## What changed

The Batch 2 framework was prepared for three canonical-only RNA deformation
families: fold creation/disappearance, cross-part approach, and a compound
fold/twist state. The implementation was subsequently organized under the
`batch2`, `shared`, and `tools` packages.

The repository publication policy was also tightened: generated datasets,
checkpoints, EXR buffers, rendered review images/videos, and local dependency
checkouts are treated as local artifacts rather than source-controlled files.
The artifact management tool supports a dry-run checkpoint retention plan that
keeps the best validation-PSNR checkpoint and `last.ckpt`; this preserves normal
inference/recovery while preventing per-epoch checkpoint accumulation.

## Why

The previous Git history included several gigabytes of generated H5 datasets and
checkpoint files. That made a normal GitHub push fail. A lightweight framework
snapshot was therefore published instead, containing source, configuration, and
small Blender assets but not the generated experiment products.

## Evidence and limits

The Batch 2 quantitative report and review package were generated before the
cleanup, but the user intentionally reset the workspace to a clean state.
Those intermediate files are consequently not asserted to be present locally in
this session. Any future numerical verdict must be regenerated from the retained
framework and labeled as that new evaluation regime; this worklog does not treat
the earlier outputs as live reproducible files.

No checkpoint splitting was applied. The observed individual RNA checkpoints
were below 500 MB; the relevant storage risk was cumulative per-epoch retention,
which is addressed by best-plus-last pruning. Splitting a live checkpoint is not
used automatically because it would require reassembly before inference.

## Related commits

* `947857e` — lightweight framework publication snapshot.
* `a709db2` — Batch-oriented experiment organization.
* `40af823` — worklogs moved to the dedicated directory.

## Open items

* Re-run Batch 2 only if fresh rendered evidence is required; its generated
  outputs are intentionally excluded from Git.
* Run the artifact manager in dry-run mode after future training, then apply
  best-plus-last pruning only after reviewing its proposed removals.
