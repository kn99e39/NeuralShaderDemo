# Batch 1A Static Baseline Runbook

This runbook governs P0 (Einar) and F0 (Rain) only.  It must not be used to
report a P1--P4/F1--F3 deformation result.

## Fixed facts

- The RNA source is the unmodified official commit `66b5b098e2013db32214e44a439a0cd66d331e4d`.
- `canonical_*.blend` is the immutable selected-surface scene; the separate
  `*_aovs.blend` copy is the official AOV-prepared render input.
- Every production train view is 512x512 at 256 Cycles samples.  Train has
  200 views (four 50-view shards); held-out validation has 40 views at the
  same quality.
- Always call the official generator with `--mode rna`.  Its default mode
  produces an incompatible seven-component query for the RNA renderer.
- The official renderer currently assigns `scene.cycles.device = "CPU"` in
  `training_dataset/blender_render_dataset.py`; this is an upstream execution
  constraint, not a local RNA modification.

## Dataset procedure

Run one shard at a time from `external/relightable-neural-assets`:

```console
.venv/bin/python scripts/generate_dataset.py --mode rna --config /mnt/c/Projects/NeuralShaderDemo/experiments/dynamic_transport_failure/configs/rain_static_hq_batch01_generate.yml
```

Validate every completed shard before generating the next one:

```console
.venv/bin/python /mnt/c/Projects/NeuralShaderDemo/experiments/dynamic_transport_failure/scripts/validate_rna_h5.py \
  --dataset /mnt/c/Projects/NeuralShaderDemo/results/batch1_hq_dynamic_failure/cloth/static/dataset/rain_static_train_batch01.h5 \
  --expected-images 50 \
  --output /mnt/c/Projects/NeuralShaderDemo/results/batch1_hq_dynamic_failure/cloth/static/dataset/rain_static_train_batch01_integrity.json
```

After all four train shards validate, merge them with
`scripts/merge_rna_h5.py`.  The merger streams one view at a time, validates
layout and resolution equality first, and writes an ignored result artifact.
Do not commit HDF5 datasets, EXRs, `.blend` run inputs, checkpoints, or
renders.

## Gate discipline

Dataset integrity confirms only that the canonical static RGB/AOV/visibility
data is nonempty and finite.  It is not a neural-render quality verdict.
Static PASS/CONDITIONAL/FAIL requires the later frozen-checkpoint review:
GT/RNA comparisons under unseen light and camera, material-class inspection,
and identity/rigid rendered controls.  Deformation claims remain out of scope
until this static gate is closed.
