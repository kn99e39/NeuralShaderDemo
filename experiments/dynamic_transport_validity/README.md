# Batch 1: Dynamic transport validity

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

## Entry points

- `correspondence.py`: renderer-independent barycentric correspondence utilities.
- `dynamic_renderer.py`: Blender/RNA correspondence-preserving renderer.
- `create_synthetic_scene.py`: deterministic, finite-thickness connected folding-sheet control scene.
- `metrics.py`: image metrics and changed-visibility attribution.
- `summarize_training.py`: TensorBoard curve/final validation extraction.
- `build_review_package.py`: review PNGs, aggregate CSV, and sweep videos.
- `build_refit_comparison.py`: shared-scale G4 frozen-versus-refit review panel.
- `tests/`: focused identity, rigid-transform, barycentric, and normal tests.
- `configs/`: immutable experiment configs and the evaluation manifest.

Run commands from the official RNA repository under Ubuntu/WSL unless stated
otherwise. Every render writes its resolved CLI/config/checkpoint and deformation
metadata into the corresponding result directory.
