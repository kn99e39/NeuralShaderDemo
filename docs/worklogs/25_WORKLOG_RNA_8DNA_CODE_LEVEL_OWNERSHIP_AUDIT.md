# Worklog — Code-Level Root-Cause Audit of RNA and 8DNA (2026-10-02)

## Session question

Worklogs 21–24 established that, in the accepted teaset case, a stationary
surface's GT radiance changes when the milk pot moves (worklog 24: mostly the
milk pot occluding tray-reflected light, plus light reflected by the milk pot;
direct light ≈ 0), and that frozen RNA and frozen 8DNA do not track it. Why do
two structurally different representations fail to express this change? Is
that an accident of two implementations, or a predictable consequence of how
the methods assign geometry-dependent transport to persistent learned state?

Analysis only. No RNA/8DNA source, checkpoint, dataset, protocol or earlier
worklog was changed. Code paths below are the pinned checkouts
(`external/relightable-neural-assets` @ `66b5b09`, `external/8dna26` @
`4a2157c`). Paper statements are from the arXiv versions (RNA:
arXiv:2312.09398, TOG 44(1) 2025; 8DNA: arXiv:2604.25129, SIGGRAPH 2026).

## IMPLEMENTATION FACT

- One diagnostic probe was run, `probe_query_invariance.py` (`e1235e1`),
  read-only over the worklog-22 feature buffers, to answer one question that
  static reading could not settle numerically: *do the learned components'
  runtime inputs change between T0 and T3 at the stationary interaction ROI?*
  Record: `results/8dna_replication/code_audit/probe_query_invariance.json`.
- The buffers compute the canonical query position exactly as the project's
  8DNA adapter does (subtract the hit part's translation;
  `teaset_parts.PartRigidAsset`), so one comparison answers the query-input
  question for both backbones. The adapter is this project's, not upstream's.
- Incidental: RNA's `validation_step` writes `<name>_{pred,gt}_<i>.exr` into
  the working directory (`rna/interfaces.py:272-296`); that is why the RNA
  checkout's root holds many EXR files from this project's training runs. They
  are outputs, not source.
- No `results/evaluation/25/` folder: this batch produced no reviewer images.

## CODE-LEVEL DATA FLOW

### RNA

| stage | code | what flows |
|---|---|---|
| data generation | `training_dataset/blender_render_dataset.py:201-239` | one Blender scene; per image a random camera and a per-pixel random sun direction (`light_radius` 0 in the official configs); Cycles `max_bounces = 32`, direct samples clamped at 20, no denoising → `color` is the **full global-illumination radiance** of the fixed asset, every bounce |
| visibility channel | `blender_render_dataset.py:317-349` | all materials set to white diffuse, `max_bounces = 0`, 1 spp, thresholded > 0 → `diffuse_direct` = **direct-light visibility only** |
| loading | `rna/datasets.py:126-163` | per pixel: color, light_dir, camera_dir, normal, visibility, position (normalised by the H5's AABB); only pixels with alpha > 0 |
| model | `rna/models.py:160-177`, `rna/neural_textures.py:92-122` | `TriplaneFeatures(position)` (three learned 512² × 8 planes, bilinear lookup, summed) ⊕ camera_dir ⊕ light_dir ⊕ normal → 4×512 MLP → **6 outputs** |
| training | `rna/interfaces.py:182-214`, `:43-49` | `where(visibility > 0, out[:3], out[3:])` against `color`; L2 on log(1+x); shadowed pixels weighted by `visibility_loss_weight` (10 in the configs used here) |
| inference | `rna/renderers.py:115-172`, `:523-626` | position, normal, camera_dir from the *current* Blender scene; position normalised by the **current scene AABB** (`blu.get_scene_bounding_box`); visibility from a current `diffuse_direct` pass; light direction and weight from the light model; radiance = selected branch × light weight |

There is no input that refers to any surface other than the query point:
`forward(position, camera_dir, light_dir, normal)` is the whole interface.

### 8DNA

| stage | code | what flows |
|---|---|---|
| training scene | `utils/dataset/path_sampling_dataset.py:62-73, 90` | `scenes.<asset>.get_scene(H)` — one fixed geometry; envelope = the asset instance's bounding box |
| training samples | `path_sampling_dataset.py:93-151`, `utils/light_transport.py:227-315` | rays from points on a unit sphere around the asset, cosine-distributed about the sphere normal, traced into the asset with BSDF sampling only (no emitter); `xi` = **first asset hit** `x1`, `wi` = direction back to the ray origin; trace until the path leaves; exit projected to the bbox → `xo` (as a direction from the box centre), `wo`; `throughput` = product of BSDF weights |
| scattering-order split | `configs/default.yaml` `bounce: 2`; `path_sampling_dataset.py:126-127` | paths with fewer than 2 asset vertices get zero throughput: the network learns **only paths with ≥ 2 vertices inside the asset** |
| model | `models/eight_dna.py:57-88` | cond = Triplane(xi) ⊕ GridCube(wi); `albedo(cond)`; normalising flows `p(xo|cond)`, `p(wo|cond, xo)` |
| training loss | `train.py:55-72` | MSE(albedo, throughput) − mean(throughput · log p(xo, wo \| xi, wi)) |
| render: direct | `models/integrator.py:153-163` | at an asset hit, Mitsuba emitter sampling with the **real BSDF** and a visibility test in the **current** scene |
| render: learned lobe | `eight_dna.py:262-340` (`eval_asset`) | (xi, wi) → asset frame by the instance's m2w; albedo(xi, wi) · p(xo, wo \| xi, wi); `xo` projected to the bbox; emitter sampled **from xo** in the current scene |
| render: continuation | `eight_dna.py:176-258` (`sample_asset`) | a BSDF direction is tested against the current scene; if it re-hits the same instance the path continues only through the learned lobe (from `xo`, `wo`), otherwise stochastically through the BSDF (escape) or the lobe |

Inside the asset, the only current-geometry operations are the direct
emitter test at the hit point and the self-visibility test of BSDF
directions. Everything that happens between two asset vertices is the learned
`albedo · p`.

### Probe result (stationary interaction ROI, common light, 66 720 query samples)

| runtime input | T0 → T3 |
|---|---|
| first-hit part | identical for 99.86% of samples (0.14%, at the ROI edge, now see the milk pot) |
| canonical position, normal, camera direction (= 8DNA xi, wi; RNA position, normal, view) | **identical** (max difference 0) on those samples |
| light direction (RNA) / lighting (both) | identical (fixed light) |
| 8DNA envelope for the hit part (attached adapter: canonical bbox + t_teapot4 = canonical bbox) | identical |
| RNA current-mode normalisation (official renderer: current scene AABB) | identical here — the milk pot moved inside the scene AABB |
| per-light-sample direct visibility | **changes**: 1.22% of light samples (light-sample seed alone: 0.11%); lit fraction 0.998 → 0.987, i.e. ≈1.1% of samples newly shadowed by the milk pot |

## METHOD-LEVEL OWNERSHIP

### A. RNA information ownership

| information | persistent / current | training source | inference source | geometry-dependent? | changes T0→T3 at the ROI? | represented at runtime? |
|---|---|---|---|---|---|---|
| query position | current input | H5 `position` (fixed scene) | current hit, AABB-normalised | yes (own surface) | no | yes |
| normal, view direction | current input | H5 | current scene | own surface only | no | yes |
| light direction | current input | per-pixel random | light model | no | no | yes |
| direct-light visibility | current input (branch selector) | `diffuse_direct` pass | current `diffuse_direct` / shadow test | yes (any occluder) | yes (≈1.1% of light samples) | yes |
| material response, all orders | persistent (triplane + MLP) | `color` (32-bounce GI) | — | via fixed scene | — | only through the four inputs above |
| **indirect-path visibility (milk pot occluding the tray's light)** | **persistent, baked** | inside `color` | none | yes (cross-part) | **yes** (WL24: −0.048 at T3) | **no** |
| **radiance arriving from the milk pot** | **persistent, baked** | inside `color` | none | yes (cross-part) | **yes** (+0.035 at T3) | **no** |
| position of other parts | absent | fixed in the scene | none | — | yes | no |

### B. 8DNA information ownership

| information | persistent / current | training source | inference source | geometry-dependent? | changes T0→T3 at the ROI? | represented at runtime? |
|---|---|---|---|---|---|---|
| xi, wi (asset frame) | current input | first asset hit of training rays | hit point via m2w (per part in this project's adapter) | own surface | no | yes |
| envelope (bbox), xo/wo parameterisation | persistent frame | asset bbox at training | instance bbox (adapter: + t_part) | asset extent | no (for teapot4) | yes |
| direct scattering at x1 | current | excluded from training (`bounce: 2`) | real BSDF + current visibility | yes | yes (small) | yes |
| self-visibility of BSDF directions | current (sampling selector) | — | current scene ray test | yes | yes (only where the direct escape term is affected) | yes |
| lighting beyond the envelope | current | — | emitter sampling from xo | lighting only | no | yes |
| albedo(xi, wi), p(xo, wo \| xi, wi) — all ≥2-vertex intra-asset transport | **persistent** | path statistics of the fixed asset | none (function of xi, wi only) | yes (internal structure) | inputs no ⇒ output no | inputs only |
| **indirect-path visibility (milk pot occluding the tray's light)** | **persistent, inside p and albedo** | training paths | none | yes | **yes** | **no** |
| **radiance arriving from the milk pot** | **persistent, inside p and albedo** | training paths | none | yes | **yes** | **no** |
| part identity / internal configuration | absent (one instance, one envelope) | — | none upstream (part ids exist only in this project's adapter, used for the coordinate pullback) | — | yes | no |

### C. Shared failure

| physical quantity (WL24, T3) | RNA ownership | 8DNA ownership | current update path | consequence under T3 |
|---|---|---|---|---|
| occlusion of tray-reflected light by the milk pot (share ≈ 0.72 common light / 0.58 envmap) | triplane + MLP, baked from the fixed-scene `color` | albedo · p, baked from fixed-asset path statistics | none in either | canonical (unoccluded) tray reflection is reproduced |
| light reflected by the milk pot (≈ 0.28 / 0.43) | same | same (≥2-vertex paths) | none | the milk pot's reflection stays at its canonical position |
| direct light and its shadowing (≈ 0.00 / 0.02) | binary visibility selects shadowed/lit branch | real BSDF, current visibility | yes in both | tracked, but carries almost no signal here (the near-mirror lobe does not connect the light to the camera on this ROI) |

### D. Methodology comparison

| representation assumption | RNA realisation | 8DNA realisation | shared / different |
|---|---|---|---|
| an asset's transport is learned for one fixed internal configuration | one Blender scene for the whole dataset | one `get_scene` for all training samples | **shared** |
| the learned query depends only on the query point's own state (+ view/light) | position, normal, view, light | xi, wi (+ exit evaluated in the scene) | **shared** |
| geometry-dependent information supplied at runtime | direct-light visibility only | direct scattering and self-visibility only | **shared in kind** (direct-order only) |
| which orders are learned | all orders, including direct shading | only ≥2 asset vertices; direct is analytic | different |
| what the learned state is indexed by | world position via a scene-AABB triplane | asset-frame xi via triplane, wi via grid | different |
| lighting model | distant light, one per sample | near-field, via the envelope exit | different |
| where the path ends | at the first hit (no continuation) | continues from the envelope exit into the scene | different |
| stated scope | assets are static (geometry retained for primary rays; no deformation discussed) | "accurate only if the internal scattering structure does not change"; "assumes no additional geometry lies inside the asset's convex hull" | both outside scope here |

## CROSS-MODEL COMMONALITY

### Causal chains

**RNA**
1. Fixed training geometry — one scene per dataset (`blender_render_dataset.py`). *Supported.*
2. Inter-part transport, including the milk pot's occlusion of the tray's light and its own reflection, appears consistently in the supervision — `color` is 32-bounce GI of that scene. *Supported.*
3. The representation can store it in persistent state — the triplane is indexed by position and the MLP by direction, so radiance that is a function of (position, view, light) at the fixed configuration is representable. *Supported by the code; how much of it RNA actually fits on this near-mirror material is limited (G1 margin 4%, worklog 22).*
4. Inference exposes only position, normal, view, light and direct visibility (`renderers.py`, `models.py:160`). *Supported.*
5. The geometry relation changes (milk pot moves). *Given.*
6. Indirect-path visibility and the milk pot's radiance change (worklog 24). *Measured.*
7. No runtime variable communicates them — the probe shows the query inputs are identical; only direct visibility changes. *Supported.*
8. The learned contribution is the same function of the same inputs, so it remains the T0 transport: valid for G0, stale for G1. *Supported.*

**8DNA**
1. Fixed training geometry — one `get_scene` (`path_sampling_dataset.py`). *Supported.*
2. Inter-part transport appears consistently in the supervision — training paths with ≥2 asset vertices carry exactly the occluded tray reflection and the milk-pot reflection of the canonical configuration (`bounce: 2`). *Supported.*
3. The representation stores it persistently — albedo(xi, wi) and p(xo, wo | xi, wi) are trained to those path statistics (`train.py:55-72`). *Supported; the worklog-21 envmap refit shows the architecture can represent T3 when retrained (case A).*
4. Inference exposes xi, wi (asset frame), the envelope, current direct scattering and self-visibility (`integrator.py`, `eight_dna.py`). *Supported.*
5. The relation changes. *Given.*
6. Indirect-path visibility and the milk pot's radiance change. *Measured.*
7. No runtime variable communicates them — xi, wi and the envelope are identical; direct and self-visibility tests are current but act on the direct term only. *Supported.*
8. albedo · p is evaluated on identical inputs, so it returns the canonical internal transport: valid for G0, stale for G1. *Supported.*

### What is common and what is not

- **Common:** both methods assign *all inter-surface transport within the
  asset* to persistent learned state that is a function of the query point's
  own state, learned from a single internal configuration, and supply only
  direct-order geometry information at runtime. Two different mechanisms
  instantiate this — a position-indexed radiance function with a direct
  visibility switch, and a conditional exit distribution with analytic direct
  scattering.
- **Different:** what is learned (all orders vs ≥2 vertices), how it is
  indexed, the lighting model, whether the path continues past the asset.
  None of these differences reaches the missing quantity.
- **Why current direct visibility does not help:** both models do recompute
  direct visibility, but on this ROI the direct term carries ≈ 0 of the change
  (worklog 24), and neither model routes current visibility into its learned
  indirect component. RNA's switch selects between two *learned* outputs; it
  cannot introduce information about which indirect paths are blocked.

### Failure levels

| issue | level | note |
|---|---|---|
| single fixed configuration in training | methodology / representation contract | both methods, by design |
| learned query = own-state only; no cross-part input | methodology / representation contract | both; the failure follows for *any* weights once the inputs are identical |
| runtime geometry limited to direct order | methodology / representation contract | RNA: visibility hint; 8DNA: analytic direct + self-visibility |
| RNA position normalised by the current scene AABB | implementation choice | no effect in this case (AABB unchanged); a coordinate confound when the AABB changes |
| RNA validation writes EXRs to the working directory | implementation choice | irrelevant to transport |
| teaset authored as one instance containing all parts | asset-authoring choice (upstream asset) | makes inter-part transport intra-asset; 8DNA's paper names convex decomposition as a remedy, but its own "no geometry inside the convex hull" condition is violated whenever parts approach |
| per-part coordinate pullback | this project's adapter | removes the coordinate confound; not part of either method |
| capacity to represent the moved state | not the cause for 8DNA (envmap refit recovers, WL22) | RNA: common-light refit near threshold; unresolved |
| implementation bugs causing the failure | **none found** | |

### Paper/method cross-check

| finding | classification |
|---|---|
| RNA bakes all shading and scattering, including self-occlusion and interreflection, into the asset | explicitly intended ("all shading and scattering is precomputed and included in the neural asset") |
| RNA's visibility hint selects shadowed vs unshadowed output by an explicit test | explicitly intended |
| RNA's hint covers direct light only | implied by the method and the code (`max_bounces = 0` diffuse pass); not discussed as a limitation |
| RNA assumes a fixed geometry | implied (one scene per asset; distant lights stated); no deformation discussion found |
| 8DNA learns only indirect transport; direct scattering is kept analytic | explicitly intended, and consistent with `bounce: 2` |
| 8DNA is accurate only if the internal scattering structure does not change; no geometry inside the convex hull | explicitly stated limitation |
| 8DNA notation (paper F(xo, ωo, xi, ωi), α(xo, ωo)) vs code (albedo(xi, wi), p(xo, wo \| xi, wi)) | consistent once the documented direction swap is applied (`eight_dna.py:19-21`) |

Neither paper claims validity under internal configuration change; the
teaset experiment tests both outside their stated scope. The point of this
audit is what their contracts assume, not fault.

## INTERPRETATION

1. **Where geometry-dependent transport lives in RNA:** in the triplane
   feature planes and the MLP — a single learned function of (position, view,
   light, normal) per visibility branch, fitted to full-GI radiance of one
   scene.
2. **Where it lives in 8DNA:** in albedo(xi, wi) and the flows
   p(xo, wo | xi, wi) — the learned statistics of every path with ≥2 vertices
   inside the asset's envelope, for one internal configuration.
3. **Current at runtime:** RNA — the query's position, normal, view, the
   light, and direct-light visibility. 8DNA — the query's xi, wi, direct
   scattering with current visibility, self-visibility of BSDF directions, and
   lighting outside the envelope.
4. **Persistent / frozen:** all inter-surface transport inside the asset in
   both, including inter-part occlusion and inter-part reflection.
5. **Why T0 and T3 differ physically while the learned query stays the
   same:** the change lives in ≥2-vertex transport that depends on another
   part's placement; the learned components are deterministic functions of
   inputs that, for a stationary query, are identical at T0 and T3 (probe).
   They therefore return the T0 transport whatever their weights; only the
   direct-order terms are current, and they carry almost none of the change.
6. **Level:** for this failure class — a stationary query under a changed
   cross-part relation inside one learned asset — the failure is a
   **methodology-level consequence of the representation contract**, not an
   implementation accident: it follows from (a) single-configuration training,
   (b) own-state-only learned queries and (c) direct-order-only runtime
   geometry, and it holds for any trained weights. It is not an architecture
   limitation of capacity (8DNA recovers when retrained on T3). This is shown
   for these two methods only; it is not a claim about all neural transport
   methods — families that condition on current geometry exist.
7. **Minimum category of current information a future method must
   represent:** something that varies with the *current relation between the
   query point and other surfaces of the asset* — at least the current
   visibility of the query's indirect incidence (which inter-surface paths are
   blocked now), and the radiance currently arriving from other surfaces —
   or, equivalently, transport state rebuilt for the current configuration.
   Local surface state and direct visibility are provably insufficient for a
   stationary query.

**Representation-level assumption that must change:** "inter-surface
transport inside an asset is a property of the asset that can be learned once
and queried by the query point's own state." Under changing cross-part
relations it is a property of the *configuration*; a method that remains valid
must either expose the current configuration to that component or stop owning
it persistently. Which of these to do is a method-design question, not
answered here.

## UNRESOLVED QUESTION

- How much of the inter-part transport RNA actually encodes at T0 on this
  material (its static quality is weak); the ownership route is established
  from code, the fidelity is not.
- Whether splitting the teaset into per-part 8DNA assets would help at all:
  inter-asset transport would then be traced with current geometry, but the
  milk pot entering the tray's convex hull violates 8DNA's stated condition.
  Not tested.
- The audit covers the released code paths used in this project; RNA's hair
  path, the UV-based modules and 8DNA's volumetric integrator were not traced.
- Whether the current-AABB normalisation of the official RNA renderer causes a
  separate coordinate confound in other trajectories (it did not here).

## Commits

| item | commit |
|---|---|
| probe code | `e1235e1` |
| probe record (`code_audit/probe_query_invariance.json`) | run at `e1235e1` |
| this worklog | see `git log` |

Execution host: RTX 5080 (probe only).
