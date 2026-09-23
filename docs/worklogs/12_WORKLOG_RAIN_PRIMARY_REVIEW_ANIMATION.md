# Worklog — Rain Primary Review Animation (2026-09-23)

## Scope

Added a small deterministic helper to present the already-evaluated fixed
front-primary Rain states F0 through F3.  It creates a labeled side-by-side
reference versus frozen-RNA GIF from the existing EXR pairs.

## Artifact and interpretation

The generated, ignored artifact is:

`results/batch1_hq_dynamic_failure/cloth/deformation/primary/rain_primary_reference_vs_frozen_rna.gif`

Each 1.5-second frame has the current reference on the left and the frozen
static RNA prediction on the right, ordered F0, F1, F2, and F3.  It is a
review aid only: it applies display tonemapping and is not used for metric
calculation or for changing the Rain `Weakened` classification recorded in
the primary measurement files.

## Reproducibility

`make_state_comparison_gif.py` accepts ordered `--state-dir` arguments and
reads `ref_0.exr` and `rna_0.exr` from each directory.  It therefore avoids
copying, renaming, or changing any source render.  The artifact remains
excluded by the repository's `results/` policy.
