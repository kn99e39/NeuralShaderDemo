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

`add_canonical_position_aov.py` writes canonical world XYZ as the RGB payload
of a point-domain `FLOAT_COLOR` attribute in the canonical state.  This is
intentional: a Blender `FLOAT_VECTOR` shader socket is spatial and would apply
the current object transform, corrupting a rigid correspondence control.  The
non-spatial Color socket retains the canonical payload, while armature/lattice
deformation transports it with the same vertex identity. Per-pixel AOV
rasterization provides barycentric interpolation over a current evaluated
triangle. It does not use the current deformed world position for frozen
TriPlane lookup.

Identity control must compare this path with the ordinary RNA renderer at the
canonical pose.  Rigid translation and rotation controls must retain this
canonical lookup while updating the current AOVs and visibility.  A failure in
either control blocks interpretation of non-rigid results.

For a rigid state with world transform `T`, the adapter audit checks
`current_position == T(canonical_position)` on covered pixels. The Rain
control moves the production armature object once; moving both armature and
already-skinned meshes would double-apply the transform and is invalid.
