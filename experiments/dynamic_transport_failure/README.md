# High-quality dynamic transport failure

This is the clean Batch 1 high-quality experiment.  It uses the published
Adobe Research RNA implementation unchanged at
`external/relightable-neural-assets` and puts all experiment-specific scene
preparation, correspondence instrumentation, evaluation manifests, and review
logic here.

The older `experiments/dynamic_transport_validity` package is not an input,
baseline, source of assets, or evidence for this batch.  Its primitive scenes
remain historical instrumentation only.

## Status and order of execution

1. `asset_registry.json` records candidates and their licence/source before use.
2. The static gate is closed for each asset before any non-rigid result is
   interpreted.  A failed or visibly unstable static gate blocks the asset.
3. Run the numerical correspondence tests and the rendered AOV checks.
4. Train only on G0/F0, freeze the selected canonical checkpoint, and evaluate
   P0--P4/F0--F3 with the same camera, lighting, topology, material, and model
   configuration.
5. Render GT and frozen RNA independently, then create full-resolution pairs,
   flicker, ROI crops, and attribution metrics under
   `results/batch1_hq_dynamic_failure/`.

No dynamic checkpoint may be used before frozen evaluation.  Refit is one
optional, predeclared representative-state control after a Category C candidate
has passed correspondence validation.

## Official RNA input contract

The published `NeuralSurfaceTriPlane` consumes
`(position, camera_dir, light_dir, normal)`.  `position` is used by the learned
TriPlane feature lookup; the remaining three vectors go directly into the RGB
MLP.  The official renderer sources position, normal, tangent, camera direction,
and visibility from Cycles AOV/deep buffers.

For this batch the adapter must therefore feed a *canonical normalized surface
position* to the frozen TriPlane while retaining current-pose camera direction,
light direction, normal, tangent diagnostic, visibility, and GT path tracing.
It must not send a deformed world position to the canonical feature lookup.
`docs/RNA_INPUT_AUDIT.md` has source locations and classifications.

## Asset contract

The required primary cases are deliberately fixed before training:

| Case | Asset | Canonical state | Deformation states | Physical ROIs |
| --- | --- | --- | --- | --- |
| Character | Blender Studio Einar rig (CC-BY) | P0, arm separated | P1 moderate, P2 approach, P3 near-contact, P4 stable contact if valid | arm--torso gap, armpit/elbow cavity, newly occluded/exposed skin or cloth |
| Cloth | separate licence-audited garment/cloth asset | F0 open/weak fold | F1 moderate, F2 deep, F3 near/self-contact if stable | fold interior, cavity, self-contact gap, new shadow |

The character source archive is intentionally external/ignored.  Its source URL,
SHA-256, extracted file names, selected objects, and any preprocessing script
must be added to `asset_registry.json` before data generation.  A cloth source
is not selected until it has the same provenance and quality inspection.

## Commands

Run from this directory with the official RNA virtual environment activated:

```console
python -m unittest discover -s tests -v
python scripts/validate_run_manifest.py --manifest manifests/character.json
python scripts/validate_run_manifest.py --manifest manifests/cloth.json
```

The manifest validator is intentionally conservative: it refuses dynamic states
until a corresponding static gate is marked `passed`, a canonical checkpoint is
identified, all preserve/change controls are locked, and declared output paths
remain under the ignored result root.

## Result layout

```
results/batch1_hq_dynamic_failure/
  environment/
  character/static/  character/P0/ ... character/P4/  character/roi/  character/videos/
  cloth/static/      cloth/F0/ ... cloth/F3/          cloth/roi/      cloth/videos/
  refit_control/  metrics/  review/
```

Results, training datasets, raw renders, temporary EXRs, and original asset
archives remain ignored.  A future selected checkpoint may be committed only as
verified sub-500 MiB bundle parts using the repository checkpoint-bundle tool.
