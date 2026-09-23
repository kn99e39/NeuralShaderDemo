# Worklog — Evaluation Artifact Staging (2026-09-23)

## Scope

Established `results/evaluation/` as the single handoff location for visual
artifacts intended for human inspection.  This responds to the distinction
between the many render/metric/debug intermediates and the small subset needed
to review an evaluated claim.

## Current curated set

The directory contains the Rain static baseline GIF, F0--F3 fixed-primary
reference-versus-frozen-RNA GIF, and full-frame plus scarf-ROI F3 comparison
and absolute-error images.  `manifest.json` records each review filename,
authoritative source path, byte count, and SHA-256.

The artifacts are copies.  Source EXRs, measurement PNGs, and JSON metrics
remain in their original experiment directories and are authoritative.

## Reproducibility

Added `publish_evaluation_artifacts.py`.  It accepts explicit source and
review-name pairs, permits only reviewable image/video extensions, copies the
files to the evaluation directory, and regenerates the manifest.  Future
human-facing output should be published through this helper after its source
render and metric validation are complete.  As with all `results/` outputs,
the staged review copies are ignored by Git.
