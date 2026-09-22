# Worklog — HQ Batch 1 Asset and Environment Rebuild (2026-09-22)

## What changed

Started a clean high-quality Batch 1 under
`experiments/dynamic_transport_failure/`.  It is deliberately separate from
the older primitive Batch 1/Batch 2 scenes and does not reuse their output as
evidence.  The new package contains the actual RNA input audit, asset
provenance registry, predeclared character and cloth state contract,
numerical correspondence tests, headless asset inspectors, and a fail-closed
dynamic-run manifest validator.

Cloned the official Adobe Research RNA repository at
`66b5b098e2013db32214e44a439a0cd66d331e4d` into ignored `external/`, created
its Linux Python 3.10 environment, and installed the official release-1.0.0
custom `bpy-3.5.1` wheel.  Its imports succeeded with CUDA-enabled PyTorch on
the local RTX 5080.  No official RNA source was modified.

The implementation audit found that the published surface TriPlane queries
learned features with position and supplies camera direction, light direction,
and normal to the RGB MLP.  The planned frozen-deformation adapter therefore
must query the canonical normalized position emitted through a barycentrically
interpolated AOV, while leaving current-pose normals, camera/light directions,
visibility, and path-traced GT current.  This boundary is documented in
`experiments/dynamic_transport_failure/docs/RNA_INPUT_AUDIT.md`.

## Asset decisions

Selected Blender Studio Einar v1 as the character source: CC-BY, native
Blender 3.5+, full production rig, 799-bone `RIG-einar`, and high-detail
clothing/skin/robot-arm texture assets.  The source archive hash is recorded
in `asset_registry.json`.  Inspection confirmed arm controls suitable for the
P0--P4 approach trajectory.

Selected Blender Studio Rain v2's `GEO-rain_scarf` and `GEO-rain_top` as the
independent secondary garment case: CC-BY, native Blender 3.0 and readable in
the official BPy 3.5.1 environment, with armature/corrective/subdivision
deformation, UVs, dedicated materials, and fabric PBR maps.  It will be
treated as a separately trained F0--F3 neural asset, not as a character
evaluation shortcut.

Rejected the Erika Archer third-party candidate despite its CC-BY listing:
the nested “torso” and “sleeve” FBX assets import as arrow meshes without
usable garment geometry or textures in BPy 3.5.1.  This rejection is retained
in the registry rather than silently replacing it.

## Validation and limits

The new numerical tests passed identity lookup, rigid translation feature
identity, and intrinsic deformation/current-normal behavior.  The dynamic
manifest validator correctly rejected execution before a static gate and
canonical checkpoint exist.

No RNA training, static quality measurement, path-traced GT/RNA comparison,
deformation rendering, ROI metric, or scientific failure classification has
yet been claimed.  The next gating work is asset scene preparation followed by
the high-quality static RNA baseline for each selected asset.  A result remains
`NOT YET ATTRIBUTABLE` until those gates and rendered correspondence controls
are complete.

## Commit status

This worklog accompanies the uncommitted clean-rebuild infrastructure in this
session.  It records no numerical experiment result.
