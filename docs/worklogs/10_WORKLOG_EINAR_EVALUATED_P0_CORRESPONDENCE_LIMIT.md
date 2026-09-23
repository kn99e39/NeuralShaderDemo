# Worklog — Einar Evaluated-P0 Correspondence Limit (2026-09-23)

## Correction to prior baseline handling

This worklog follows the Einar limitation noted in worklog 09. The original canonical attribute stored each source mesh's datablock/world coordinate, but the Einar release scene is already armature-evaluated at P0. Consequently the earlier identity audit was not a valid frozen-correspondence baseline (p99 absolute error 0.2637).

## Change and measurement

`add_canonical_position_aov.py` now captures evaluated P0 vertex positions for meshes whose evaluated topology preserves the source vertex count, retaining the non-spatial `FLOAT_COLOR` payload. Einar v4 reduced identity error to MAE 0.00245 and p99 0.03418 at 512x512. The median, p90 and p95 max-channel errors are 0.00098, 0.00195 and 0.00293 respectively.

One visible helper, `GEO-einar_tire_shoulder_surfacedeform`, changes topology from 218 source vertices to 3127 evaluated vertices through its Armature/Subsurf/Shrinkwrap/Corrective Smooth stack. The adapter reports this explicitly and falls back to source coordinates rather than inventing a vertex-index mapping. That remaining component is material: 1,308 of 64,154 covered pixels exceed 0.01 absolute error and 1,040 exceed 0.03; p99 rises to 0.06348 when measured as per-pixel maximum channel error.

## Verdict and limit

This repair substantially improves the P0 contract but does not validate the whole Einar evaluated domain. Because the topology-changing shoulder helper remains visible and is part of the intended arm/shoulder deformation regime, no Einar frozen-control or dynamic verdict is claimed. The appropriate current classification for Einar is **not attributable / not evaluable under this correspondence adapter**, not support or failure of the RNA hypothesis.

The Rain primary conclusion recorded in worklog 09 is unchanged. This worklog does not alter official Adobe source.