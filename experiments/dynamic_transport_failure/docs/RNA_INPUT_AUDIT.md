# RNA current-geometry input audit

Scope: official Adobe Research RNA checkout at commit
`66b5b098e2013db32214e44a439a0cd66d331e4d`.  This is an implementation fact,
not yet an experimental observation.

| Input or quantity | RNA source | Exact classification | Frozen deformation evaluation requirement |
| --- | --- | --- | --- |
| TriPlane lookup position | `rna/models.py`, `NeuralSurfaceTriPlane.forward` | **EXPLICIT MODEL INPUT** and coordinate of learned-feature lookup; normalized from rendered world position | replace only with barycentrically corresponding canonical normalized position |
| TriPlane feature grid | `rna/neural_textures.py`, `TriplaneFeatures` | **LEARNED FEATURE CONTENT**, canonical | freeze |
| RGB MLP | `rna/models.py`, `rgb_network` | **LEARNED FEATURE CONTENT**, canonical | freeze |
| Camera direction | renderer `camera_dir` AOV, `rna/renderers.py:process_deep_buffers` | **EXPLICIT MODEL INPUT**, renderer-derived | current pose/camera AOV; camera pose is locked across deformation states |
| Light direction | generated from rendered hit point and fixed light, `rna/lights.py` | **EXPLICIT MODEL INPUT**, renderer-derived | recompute from current hit position under fixed lighting |
| Surface normal | renderer `normal` AOV | **EXPLICIT MODEL INPUT**, renderer-derived | recompute/rasterize from current deformed geometry |
| Tangent | renderer `tangent` AOV | **NOT PRESENT** in the selected surface TriPlane `forward`; retained only as a renderer/dataset diagnostic | recompute/rasterize from current deformed geometry and retain in evidence |
| World hit position | renderer `position` AOV | **RENDERER-DERIVED**; normalized to form position input and to construct point-light direction | current pose for light direction and diagnostics; never use as frozen TriPlane identity |
| UV | renderer `uv` AOV | **PRECOMPUTED** topology/material coordinate; **NOT PRESENT** in this TriPlane forward path | preserve; diagnostic and material validation only |
| Direct visibility | `diffuse_direct` ray/deep-buffer pass | **RENDERER-DERIVED branch selector**, not an MLP input; selects one of two predicted RGB branches after forward evaluation | recompute under current geometry; preserve buffer and selected branch in output |
| Shadow / occlusion | no separate surface-model argument | **NOT PRESENT** as an explicit model input | only affects the renderer-derived binary direct-visibility branch selection in this official path |
| Deep buffer | `rna/renderers.py:process_deep_buffers` | **RENDERER-DERIVED intermediate**, not an MLP input | current scene only; use it for feature AOV extraction and diagnostics |
| Distance / near-field term | no surface model argument | **NOT PRESENT** | no fabricated distance feature may be added |
| AABB used for position normalization | `utils.ops.normalize_positions` via official renderer | **PRECOMPUTED** canonical scene extent | freeze canonical AABB; do not recompute after deformation |

## Adapter boundary

The adapter is outside the official repository.  It writes a point-domain float
attribute containing each vertex's canonical normalized world position before
pose application.  Cycles interpolates that attribute over the current triangle
and emits it as a float EXR AOV.  Thus triangle identity plus barycentric weights
select the frozen feature identity, while the official current-pose AOVs remain
the geometry-derived inputs.  In the official renderer, current direct
visibility is recomputed as a renderer-derived selector between the two RGB
branches; it is not a surface-network input.  This affects no learned RNA
parameters, loss, network topology, or canonical training data semantics.

## Required falsification controls

- Identity: canonical AOV and ordinary canonical lookup agree within float EXR
  tolerance.
- Whole-object translation and rotation: barycentric identity remains invariant;
  a coupled camera/light frame control must not create a large mismatch.
- Intrinsic deformation: per-pixel canonical AOV equals interpolation of the
  source triangle's canonical vertices and normals/tangents come from current
  geometry.
- Any failed control is Category A/B and blocks transport interpretation.
