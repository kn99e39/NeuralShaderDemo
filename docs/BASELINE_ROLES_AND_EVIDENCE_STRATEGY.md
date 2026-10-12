# Baseline Roles and Evidence Strategy — Object-Level Selective Neural Transport Recomputation

**Updated:** 2026-10-12  
**Status:** Active interpretation/control contract; object-level selective recomputation has NOT been experimentally validated  
**Project:** NeuralShaderDemo / Selective_Recompute

## 1. Central Scope Correction

User-defined "selective recomputation" refers to **objects with stale learned transport**, not partial feature grids within a single object/model.

- An invalid object model N_i may be **fully recomputed**, including all its MLP/triplane/directional weights, from scratch or by complete warm-start refitting.
- A valid object N_j must retain its complete pretrained transport model.
- A geometry-edited object may affect other **stationary** receiver objects. Discover the radiometric dirty set, not just the moved set.
- Geometry dirty D_G differs from transport dirty D_T.
- Current **per-object** ownership and scene-level recomputation semantics are the decision frontier.

Previous Worklog 30 tested *inside one 8DNA whole-asset model*: update selected triplane cells, freeze all shared flows, and compare with full-model updating. Negative result valid, but entirely **off-target** as a test of selecting whole object models. Do not include WL30's gate as a blocker for this direction.

Prior separate working architectures retained:
- docs/history/ARCHITECTURE_LIFECYCLE_SEPARATION_2026-10-07.md
- docs/history/ARCHITECTURE_INTRA_OBJECT_FEATURE_REFIT_2026-10-08.md

## 2. Historical Baselines and Evidence Roles

| Lineage | Scientific role | What it does NOT show |
|---|---|---|
| RNA/Rain | correspondence, nonrigid failures with known local confounds; fixed-target physical change too weak | generalized failure attribution |
| 8DNA/Teaset, WL21–22 | stationary receiver's GI changes as another part moves; stable whole-rigid-motion control | that objectwise neural selective recomputation works |
| RNA/Teaset | independent-backbone repetition of frozen staleness in same scene/light regime | newly proposed model's cross-backbone transfer |
| WL24–25 | physical attribution: nonlocal indirect-path occlusion plus moved-part reflected radiance | online detector for invalid model owners |
| WL26–27 | historical full-asset neural and physical ray tracing cost context | same-method per-object quality-matched speedup |
| WL28–29 | sparse K=32 dynamic transport state candidates negative | selective object rebuilding negative |
| WL30 | **intra-object** triplane-cell-only refit negative (8DNA triplane with flows frozen) | **whole-object full-model refit** negative |

Do not delete, overwrite, reinterpret, or silently integrate these different questions.

## 3. Definition of the Current Substrate

Per-object ownership must be real, not nominal. Each object is assigned an independently trainable **complete model N_i** with its own checkpoint and clear surface/radiometric semantics.

Internal model basis can reuse RNA, triplanes, 8DNA, vertex/surface features, directional bases or new components if justified. Never reject an internal representation merely because its partial feature update is difficult.

When evaluating a model's quality, ask:
- Does its output include correct **cross-object** GI at T0/G0?
- Can it be fully refit on G1 using full-scene reference transport while **other object models** stay frozen?
- Can the renderer compose object models without double-counting/omitting indirect light?
- Does its contract support rigid moves, stationary receiver reillumination and valid-object preservation?
- Are several genuine object owners present, so selection has a nontrivial numerator/denominator?

If object-level training or GI composition is inadequate, this is **O0 failure**; stop before invalidation algorithm design.

## 4. Matched Scene-Level Experimental Controls

Use a genuine multi-object scene, high-quality full-scene transport reference, stable per-object correspondence, physically meaningful T0→T3 indirect transport change, and at least one unaffected object.

| Arm | Model update scope | Explanation |
|---|---|---|
| A — Frozen | no object model updated | shows stale transport baseline |
| B — All-object scratch | every object model fully retrained from scratch | new-scene capacity/expensive recompute |
| C — All-object warm-start | all model parameters for every object permitted to refit | strongest obvious global neural adaptation |
| D — Geometry-dirty only | fully refit directly changed geometry objects | tests mover-only error |
| E — Oracle transport-dirty | fully refit complete N_i for every radiometrically dirty owner; freeze valid N_j | intended selective object-level upper bound |
| F — Optional simple spatial selector | fully refit full models of spatially close owners | baseline for dependency selection |

Between C and E **change only which full object models are selected**. Do not freeze shared layers/flow weights within selected models to make them "local". Ensure the comparison uses one partition, architecture, reference, training regime, optimizer and camera/light/material protocol.

Scratch and warm-start times should be measured independently. A warm refit of *all* objects is not the same as scratch all-object regeneration.

## 5. Oracle Object Transport-Invalidation and Physical Closure

A reference-derived oracle may identify objects whose transport validity changed due to an edit. The classification is not simply "pixels near moving object"; evaluate stable samples on the entire affected object and indirect GI even where its own geometry is static.

Track:
- directly modified geometry objects D_G;
- all transport-invalid object models D_T, including stationary indirect receivers;
- objects whose transport remains valid;
- cross-object dependency paths and higher-bounce propagation;
- potentially dense dirty closure and cases where no selective cost saving is possible.

Predeclare noise floor, affected criterion, surface-sample coverage, object aggregation and independent evaluation split. If output selection validity is uncertain, conservatively invalidate; never claim 100% oracle completeness without justification.

The oracle detector cost may be reported separately for a conditional upper bound, but cannot disappear in deployment-oriented comparisons.

## 6. Cost and Quality Metrics

**Quality:** physical reference GI, object-conditioned static T0, all-object T3 oracle comparison, per-object affected/valid error, signed transport-change gain on stationary receivers, glossy reflection bands, global render quality, compositional consistency, light-transport noise.

**Preservation:** complete model tensor identity for retained N_j, same-object prediction stability with appropriate current geometry and light, and scene-level validity. Reusing weights unchanged does not imply physically correct output; transport validity must be checked.

**Accounting:** object count/size/training steps, |D_G|, |D_T|, updated object percentage and parameter count, per-object reference target generation, path tracing over whole scene, full per-owner optimization, checkpoint traffic, validation, dependency/oracle time, rendering/inference, GPU peak and actual end-to-end wall-clock to matched quality.

**Scaling:** scene with low dirty fraction and high dirty fraction; multi-bounce propagation may invalidate many objects. Never use parameter count, object count or traditional reported 8DNA scratch time alone as speedup.

**Visual inspection:** moving object, indirectly affected stationary receiver, truly unaffected object, glossy/high-frequency regions, masks and model ownership overlay, image difference maps. Synthetic fixtures for correctness only; real-scene replay needed for method feasibility.

## 7. Method and Novelty Comparators (After the Oracle Gate)

Traditional selective/hierarchical GI dependency (Drettakis/Sillion, later incremental GI, Enlighten) is explicit prior art. Objectwise neural representation/composition with changing geometry/lighting (NeLT, Superposed Deformable Feature Fields) is also prior art.

The defensible technical difference must concern **validity of learned object-owned GI models, correct nonlocal dependency closure, complete object-model incremental refit while retaining others, and measured current-geometry quality/cost improvement**. Neither an object hierarchy nor "per-object neural shaders" is inherently new.

Current neural dynamic systems that render new trained transforms without per-object retraining remain important quality/performance baselines; do not assume none exist.

## 8. Interpretation and Stop Conditions

- A scene with only one neural-owner model cannot test object-level selection; split ownership or select a new scene.
- Objectwise GI may be unrepresentable or not composable at enough quality; stop at O0.
- If a stationarily located receiver has changed indirect transport, a geometry-only selector is insufficient; include it or report error.
- Oracle update of dirty objects may still require updating most objects; then lack of savings is meaningful negative evidence.
- A model's whole parameters may be rewritten; selective **within-object** weights are not a required contract.
- If object models cannot be trained in isolation from each other without inconsistency, attribute the issue and stop before adding a graph.
- Negative results must be scoped to tested scene/ownership and not generalized to all scene-level learned GI.

## 9. Present Next Question

> Can object-owned high-quality neural GI models be recomputed **in their entirety only for the transport-invalid scene objects** while all valid object models remain frozen, with fidelity close to all-object reconstruction and meaningful quality-matched update savings?

This is an untested research question. It is not answered by WL30's triplane-cell masking experiment.
