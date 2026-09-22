# RNA current-geometry input audit

Scope: official Adobe Research RNA checkout at commit
`66b5b098e2013db32214e44a439a0cd66d331e4d`.  This is an implementation fact,
not yet an experimental observation.

| Input or quantity | RNA source | Classification in canonical training | Frozen deformation evaluation requirement |
| --- | --- | --- | --- |
| TriPlane lookup position | `rna/models.py`, `NeuralSurfaceTriPlane.forward` | learned feature query, normalized from rendered world position | replace only with barycentrically corresponding canonical normalized position |
| TriPlane feature grid | `rna/neural_textures.py`, `TriplaneFeatures` | learned, canonical | freeze |
| RGB MLP | `rna/models.py`, `rgb_network` | learned, canonical | freeze |
| Camera direction | renderer `camera_dir` AOV, `rna/renderers.py:process_deep_buffers` | renderer-derived | current pose/camera AOV; camera pose is locked across deformation states |
| Light direction | generated from current rendered point and fixed light, `rna/lights.py` | renderer-derived | recompute from current hit position under fixed lighting |
| Surface normal | renderer `normal` AOV | renderer-derived | recompute/rasterize from current deformed geometry |
| Tangent | renderer `tangent` AOV | renderer-derived diagnostic; not a surface TriPlane forward input | recompute/rasterize from current deformed geometry and retain in evidence |
| World hit position | renderer `position` AOV | renderer-derived before normalization | current pose for light direction and diagnostics; never use as frozen TriPlane identity |
| UV | renderer `uv` AOV | precomputed topology/material coordinate | preserve; diagnostic and material validation only for TriPlane path |
| Direct visibility | `diffuse_direct` deep buffer | renderer-derived | recompute under current geometry; preserve buffer in output |
| Deep/ray-query light transport | custom BPy/Cycles path invoked by official renderer | renderer-derived | current deformed scene and fixed lighting; do not reuse canonical visibility |
| AABB used for position normalization | `utils.ops.normalize_positions` via official renderer | precomputed canonical scene extent | freeze canonical AABB; do not recompute after deformation |

## Adapter boundary

The adapter is outside the official repository.  It writes a point-domain float
attribute containing each vertex's canonical normalized world position before
pose application.  Cycles interpolates that attribute over the current triangle
and emits it as a float EXR AOV.  Thus triangle identity plus barycentric weights
select the frozen feature identity, while the official current-pose AOVs remain
the geometry-derived inputs.  This affects no learned RNA parameters, loss,
network topology, or canonical training data semantics.

## Required falsification controls

- Identity: canonical AOV and ordinary canonical lookup agree within float EXR
  tolerance.
- Whole-object translation and rotation: barycentric identity remains invariant;
  a coupled camera/light frame control must not create a large mismatch.
- Intrinsic deformation: per-pixel canonical AOV equals interpolation of the
  source triangle's canonical vertices and normals/tangents come from current
  geometry.
- Any failed control is Category A/B and blocks transport interpretation.
