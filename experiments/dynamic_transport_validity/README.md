# Dynamic transport validity experiments

This directory contains isolated instrumentation for testing a frozen RNA surface
asset after topology-preserving deformation.  The official implementation is kept
unchanged in `external/relightable-neural-assets`.

The correspondence path stores each canonical vertex position (normalized to
`[0, 1]` with RNA's canonical scene AABB) in a floating-point point-domain mesh attribute.
Cycles rasterizes that attribute into a `canonical_position_aov`, so a pixel on a
deformed triangle receives the barycentric interpolation of the corresponding
canonical triangle.  RNA then receives:

- canonical normalized position for triplane lookup;
- current camera direction, light direction, geometric normal, world position,
  and recomputed visibility from the deformed scene.

This is experimental instrumentation only.  It does not alter RNA's model,
features, loss, or checkpoint.

## Layout and entry points

- `batch1/`: Lego and synthetic folding-sheet controls, including G4 refit.  Its
  `assets/` and `configs/` are exclusive to Batch 1; scene construction and
  correspondence validation scripts also live here.
- `batch2/`: authored A/B/C deformation families.  It owns their assets,
  training/evaluation configs, asset generator, deformation definitions, and
  region-of-interest review script.
- `shared/`: renderer-independent correspondence, the Blender/RNA renderer,
  metrics, compatibility code, and review/batch-metric utilities used by both
  batches.
- `tools/`: environment capture, checkpoint inspection, artifact management,
  deterministic launcher, and TensorBoard summarization.
- `tests/`: focused tests for shared correspondence behavior.

Run a script from its listed directory, for example
`experiments/dynamic_transport_validity/shared/dynamic_renderer.py` or
`experiments/dynamic_transport_validity/batch2/create_assets.py`.  Paths embedded
in the configs remain workspace-relative and have been updated for this layout.

Run commands from the official RNA repository under Ubuntu/WSL unless stated
otherwise. Every render writes its resolved CLI/config/checkpoint and deformation
metadata into the corresponding result directory.
