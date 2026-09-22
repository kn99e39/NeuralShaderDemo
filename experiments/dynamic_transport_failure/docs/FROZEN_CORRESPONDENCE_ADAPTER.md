# Frozen correspondence adapter contract

For every current-pose rasterized surface sample, the adapter uses:

| Quantity | Source | Role |
| --- | --- | --- |
| TriPlane position | `canonical_position_aov` | Persistent identity; canonical world position, normalized by the canonical training AABB. |
| Normal | current `normal_aov` | Current explicit surface-network input. |
| Camera direction | current `incoming_aov` | Current explicit surface-network input. |
| Light direction | current light query | Current explicit surface-network input. |
| Direct visibility | current `diffuse_direct` render | Current renderer-derived RGB-branch selector. |
| GT radiance | current Cycles render | Current physical reference only. |

`add_canonical_position_aov.py` writes a point-domain canonical-world-position
attribute in the canonical state.  Armature/lattice deformation transports the
attribute with the same vertex identity; per-pixel AOV rasterization provides
barycentric interpolation over a current evaluated triangle.  It does not use
the current deformed world position for frozen TriPlane lookup.

Identity control must compare this path with the ordinary RNA renderer at the
canonical pose.  Rigid translation and rotation controls must retain this
canonical lookup while updating the current AOVs and visibility.  A failure in
either control blocks interpretation of non-rigid results.
