# Worklog — Batch 1A Static Domain and Dataset Bridge (2026-09-23)

## Scope

This session advanced only the Batch 1A static prerequisite for the Einar v1
character and Rain v2 garment assets.  It did not run, score, or interpret a
P1--P4/F1--F3 deformation experiment.

## Implementation facts

- Einar's selected neural surface domain was reduced to the torso garment and
  the human/robot arm surfaces that can participate in the intended approach.
  Head, legs, footwear, hair, satchel, props, and helpers are excluded.  Rain
  uses the scarf and top as one garment domain.
- `prepare_canonical_scene.py` now builds one `RNA-Canonical` scene that links
  only selected meshes and the source armature.  The source `.blend` and the
  selected mesh topology, UVs, material graphs, and pose remain unchanged.
  This replaced a weaker `hide_render`-only scheme which could expose objects
  from another linked production scene.
- The run scene retains the source World because the official RNA renderer
  accesses its World node tree.  Official AOV preparation is written to a
  separate `*_aovs.blend` file.
- The RNA input audit was corrected: position, camera direction, light
  direction, and normal are the explicit surface-network inputs.  Direct
  visibility is recomputed by the renderer and selects a predicted RGB branch;
  it is not an MLP input.
- Dataset validation and streaming HDF5 merge tools were added.  Generated
  HDF5, query, EXR, canonical `.blend`, checkpoint, and review outputs remain
  ignored by Git.

## Observations and measurements

- Evaluated-mesh engineering checks found exactly 70 Einar and 2 Rain render
  meshes in their isolated run scenes.  In the tested non-scientific poses,
  evaluated vertex/triangle counts, triangle ordering, material slots, and UV
  layers were stable.  Einar's source auto-control did not move the evaluated
  mesh with scripts disabled; direct `DEF-UpperArm.R` motion changed two
  selected surfaces.  Rain `FK-Scarf2` changed both selected garments.
- One-view RNA query/AOV/HDF5 probes completed for both assets.  Required
  color, alpha, position, normal, camera/light direction, UV, and direct
  visibility arrays were present, finite, and nonempty.  These are bridge
  checks, not render-quality measurements.
- A Rain full-quality calibration at 512x512 and 256 Cycles samples completed
  in approximately 27 seconds including process/startup overhead.  The first
  50-view production-quality Rain train shard completed and passed integrity:
  50 slices at 512x512, alpha absolute sum 1,376,116, color absolute sum
  1,016,634.09, and direct-visibility absolute sum 686,328.
- The official renderer explicitly sets `scene.cycles.device = "CPU"` in its
  dataset code.  This upstream execution constraint was recorded; no official
  RNA source was modified.

## Interpretation and limits

The static production data bridge is established and one Rain train shard has
verified integrity.  The complete 200-view train/40-view validation datasets,
static RNA training, frozen checkpoint, unseen-condition GT/RNA comparisons,
and rigid rendered controls remain required before any static PASS,
CONDITIONAL, or FAIL verdict.  Therefore this worklog makes no claim about
neural-render quality, pose robustness, correspondence quality, or dynamic
transport.
