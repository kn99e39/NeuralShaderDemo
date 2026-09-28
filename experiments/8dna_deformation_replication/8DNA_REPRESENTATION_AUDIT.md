# 8DNA Representation Audit

## Scope and source

This audit is based on the official implementation at
`external/8dna26/`, commit
`4a2157ca24e506c5ac0831f27d656ecc50a64f07`. It is a code audit, not an
inference from the paper abstract. The relevant implementation paths are
`models/eight_dna.py`, `models/integrator.py`, `models/normflow.py`,
`models/mlps.py`, `utils/dataset/path_sampling_dataset.py`, and
`utils/light_transport.py`.

The official static baseline did not pass on the available host. Therefore,
this audit records the implementation contract but does not authorize a
deformation experiment or support an empirical claim about 8DNA behavior.

## Persistent learned state

The checkpoint contains one asset-specific `EightDNA` model with:

- a tri-plane feature grid encoding the incident position `xi`;
- cubemap feature grids encoding the incident direction `wi` and sampled exit
  position direction `xo`;
- a conditional normalizing flow for `p(xo | xi, wi)`;
- a conditional normalizing flow for `p(wo | xo, xi, wi)`; and
- an RGB albedo/throughput MLP conditioned on `xi, wi`.

In the code, camera-side quantities are named incident (`xi`, `wi`) and
light-side quantities are named exitant (`xo`, `wo`), explicitly opposite to
the paper's naming convention. The checkpoint is asset-specific and contains
no pose, deformation, topology, primitive-ID, UV, or material-ID input.

For the released `bounce: 2` configuration, dataset samples with path depth
below two are zeroed. The neural component is therefore trained to represent
the indirect/internal part selected by that path-depth policy, while ordinary
BSDF sampling remains available in the integrator.

## Current/query-time inputs

At a camera ray's current surface hit, inference obtains:

- `xi`: the current Mitsuba surface-interaction position;
- `wi`: the current world-space incident direction;
- the current asset instance transform (`m2w`);
- the asset bounding box;
- current scene emitters, ordinary BSDFs, and ray visibility; and
- Monte Carlo samples and an RGB channel selection.

`xi` and `wi` are transformed into the asset frame before querying the frozen
network. The model samples `xo` and `wo`. `xo` is a unit-direction
parameterization, projected from the bounding-box center to the asset bounding
box; it is not a persistent surface ID. The sampled outgoing direction is
transformed back to world space.

## Geometry assumptions

Training paths are generated in one concrete Mitsuba scene. The learned
distribution is fitted to canonical path samples containing the first asset
intersection, the last internal/path intersection projected to the canonical
bounding box, its outgoing direction, and path throughput. Consequently, the
learned state bakes in the canonical asset's internal visibility, scattering,
and endpoint distribution.

The inference adapter supports an asset-level transform. It does not implement
non-rigid deformation or surface correspondence. Its coordinate conversion
uses the instance's 3x3 transform as a rotation-like map and does not constitute
a general inverse deformation map. `prepare_scene()` also caches instance
transforms and bounding boxes on first use, so separate scene/integrator
construction is required for safely evaluated locked states.

A semantically defensible frozen non-rigid evaluation would need, at minimum:

1. a topology-preserving map from each current first-hit point to its canonical
   surface point;
2. a justified pullback of the incident direction into the canonical local
   frame;
3. a fixed, documented proxy-envelope convention for the sampled exit
   direction; and
4. current GT geometry/materials that implement the same deformation.

For volumetric assets, material preservation additionally requires a validated
warp of the released volume field. Moving only the boundary mesh while leaving
the density/albedo grid fixed would not be a material-preserving deformation.

## Lighting assumptions and physically recomputed work

The network does not receive a light identifier or direct light position. It
represents a lighting-independent conditional distribution over internal asset
transport. At inference, the renderer samples/evaluates the current external
emitters at the proxy exit interaction, traces ordinary BSDF paths, evaluates
current emitter visibility, and performs MIS. This is what permits near-field
relighting without retraining.

The renderer therefore recomputes transport outside the neural asset and the
ordinary/direct lobe, but it does not recompute the learned internal conditional
distribution when the asset's intrinsic geometry changes.

## Deformation-feasibility conclusion

The released representation does assume fixed intrinsic geometry, but a frozen
deformation experiment is not automatically meaningless. A controlled surface
asset with explicit per-surface correspondence could query the canonical
network while rendering current geometry, deliberately testing the validity of
the frozen internal transport distribution. Feeding deformed world positions
directly into the canonical tri-plane would instead be an uncontrolled spatial
query mismatch.

Two release candidates were inspected before any neural deformation output:

- `seal`: 334,189 vertices and 668,378 faces plus a 24 MiB heterogeneous
  `jade.vol`. It is the official demo baseline, but a material-preserving
  deformation requires a validated volume warp and is unsuitable as the first
  deformation candidate.
- `teaset`: four separate rough-conductor meshes (tray, milk pot, biscuit tin,
  and tea pot), each with fixed topology. Relative rigid part motion offers
  exact per-part correspondence and a physically meaningful interreflection /
  approach family. It is the geometry-only preferred primary candidate.

No `teaset` trajectory was implemented or evaluated because the official
baseline gate failed first.
