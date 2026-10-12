# Research Centric Topic — Object-Level Selective Neural Transport Recomputation

**Last updated:** 2026-10-12  
**Status:** Active problem and explicitly clarified architectural direction; full method not validated  
**Target:** SIGGRAPH 2027 (official Technical Papers deadline to be verified separately)

## 1. Fundamental Research Problem

High-quality neural shading / learned global illumination fitted to a fixed geometry configuration can retain stale interreflection, indirect-path occlusion, glossy and multi-bounce transport after a geometry edit. Re-evaluating the frozen shader with current local surface inputs is not necessarily sufficient.

**Fundamental question:** Can we reuse *entire pretrained neural transport models belonging to scene objects whose transport remains valid*, while fully recomputing/refitting the **entire neural transport model of every object whose transport has become invalid**?

The selection unit is the **scene object**, not a cell, latent feature, vertex, parameter or partial model layer. All learned parameters belonging to an invalid object may be recomputed. No intra-object parameter masking is required by the research intent.

The approach is selectively *which object models* are rebuilt, not selectively *which weights within a model* are retrained.

## 2. Critical Distinction: Geometry-Dirty and Transport-Dirty

Let O={O_1,...,O_n} denote a declared object partition, G0/G1 the former/current scene geometry, and N_i the **complete owned neural transport model** of object i, fitted to the scene's transport configuration.

- Geometry dirty D_G: objects directly transformed, edited or deformed.
- Transport dirty D_T: objects whose **learned transport model is invalid under G1**, including stationary receivers whose indirect lighting changes when another object moves.

**D_G is not equal to D_T.** In the canonical teaset case, moving the milk pot changes reflections and indirect occlusion on stationary teapot/tray surfaces. Recomputing the milk pot alone cannot repair those other objects' baked transport if their models store full scene-conditioned GI.

The intended update is:

~~~text
Current scene edit G0 -> G1
    -> discover affected *object transport dependencies*
    -> compute conservative transport-invalid object set D_T
    -> for every i in D_T: fully refit/retrain N_i on current-scene transport
    -> for every i not in D_T: keep N_i unchanged
    -> compose current scene using the mixture of updated and reused object models
~~~

"Fully recompute object" means recomputing the object's **complete learned transport representation**; it does not require scratch initialization. Both complete-object retraining from scratch and complete-object warm-start may be tested; all parameters owned by an invalid object remain eligible. Supervision may require rays that traverse other objects or the whole scene.

One globally shared trainable decoder with no object-level ownership cannot simply be updated for one object without affecting other objects. That is a model-boundary issue to solve/test, **not evidence that triplanes or RNA are impossible**. A per-object RNA/8DNA/triplane model is a permissible base.

## 3. Established Evidence and Correct Scope

- Worklogs 21–22: a changed relative placement within the canonical teaset produces nonlocal indirect transport changes at stationary surfaces; frozen 8DNA and RNA do not track it adequately. Relation-preserving rigid scene motion is an essential control.
- Worklogs 24–25: indirect-path occlusion and radiance reflected from the moved part dominate; current local geometry and direct visibility do not fully determine the learned GI.
- Worklog 26: physical ray-traced recomputation is expensive on its own benchmark; not quality-matched to any proposed method.
- Worklog 27: historical **asset-level scratch** neural refits are expensive and vary by light regime; not object-by-object rebuild timings.
- Worklogs 28–29: two compact relational state attempts fail their held-out near-mirror regime; lifecycle-separated state remains a different on-hold candidate.
- Worklog 30: a **mis-scoped intra-object experiment** tested selective masking of 8DNA triplane cells while freezing shared weights. It fails on that one substrate, but this does **not test or falsify the object-level selective recomputation proposed here**. The experiment remains a valid historical result for feature-level selection only.

The canonical teaset was previously authored as one learned 8DNA asset. For an object-level test, the scene partition and per-object neural ownership must be defined explicitly; calling a single all-part asset an "object" makes the object-level selection test degenerate.

## 4. Prior Art and Novelty Boundary

Object-level incremental GI, hierarchical dependency/visibility structures and selective recomputation already exist (e.g. Drettakis and Sillion 1997; Luksch et al. 2019; Enlighten workflows). Neural per-object composition and dynamic inference already exist (NeLT 2023; Superposed Deformable Feature Fields 2024).

Do **not** claim novelty for "a hierarchy", "only update affected objects", or "per-object neural representation" alone.

The research opportunity, if experimentally supported, is to make **scene-wide learned GI object models** reusable across unforeseen edits using meaningful transport-validity dependencies, conservative update closure, accurate full-object neural rebuilds, and favorable quality/cost scaling compared with rebuilding every scene object's learned transport.

Compared with dynamic-object inference conditioned on trained transforms, the intended operation is **post-training repair** when the previous learned transport has become invalid. Compare fairly rather than asserting that neural GI cannot handle motion.

## 5. Object Ownership Is the Key Design Boundary

Define for each object:
1. The identity and transform/correspondence of O_i.
2. Its entire neural model N_i (all trainable parameters and optionally any object-local cache).
3. What transport N_i predicts: outgoing surface radiance under specified lighting, incident transport, transfer coefficients, or other semantics.
4. Whether and how N_i encodes **scene context** including other objects.
5. How recomputing N_i can use the **current entire scene** for supervision without changing unchanged objects' learned models.
6. How final rendering composes all object contributions, avoiding double-counting or missing cross-object transport.

If N_i represents transport affected by other objects, its geometry may remain unchanged yet N_i becomes transport dirty.

Scene-level dependencies can propagate over multiple bounces. One mover may invalidate many or all objects; do not assume that affected sets are always small or spatially close.

The granularity can later be hierarchical objects or parts if physically and semantically justified, but **a partial feature subset within N_i is not the required meaning of selective**.

## 6. Immediate Scientific Frontier

**Before a deployable dependency predictor**, test object-level recomputation with an offline reference-derived oracle object invalidation set:

- Establish a multi-object scene and a *per-object* learned transport model capable of good fixed-scene high-quality GI.
- Pretrain all N_i at G0.
- Change geometry (prefer the physical teaset T0/T3 edit if a meaningful object partition and scene supervision can be constructed).
- Identify transport-dirty objects D_T^oracle from reference radiance changes on stable/current object surface samples. Distinguish geometry dirtiness from receiver transport dirtiness.
- Fully rebuild all parameters of each selected object model, with unchanged object models held bit-identical.
- Compare with full objectwise rebuild of every model, changed-geometry-only rebuild, frozen scene, and whole-object warm-start/scratch controls.
- Test rendering fidelity, multi-bounce consistency, cross-object effects and total wall-clock (supervision generation included).

The first gate tests **object ownership and whole-object update economics**, not feature-level localizability or mask efficiency. Once that works, test automated dependency detection. If updating almost every object is necessary, report that negative locality result.

## 7. Distinct Sources of Update Cost

Separate:
- discovery of dirty object set (oracle cost vs runtime detector);
- current-geometry ray tracing/reference/supervision generation per object (may traverse the entire scene);
- fully optimizing the selected object's entire model, from scratch or warm start;
- checkpoint load/store, validation, inference/rendering composition;
- any multi-object consistency iteration and dependent recomputations.

Object-level selective retraining can be useful even when within-object computation is entirely dense. Lower object count is not, by itself, proof of lower actual update latency.

## 8. Research Conduct / Stop Conditions

- Preserve historical scripts, Worklogs 21–30 and prior architecture snapshots; never reinterpret Worklog 30 as object-level failure.
- Synthetic correctness + actual multi-object real-scene evidence; quantify object counts, dirty fractions, affected receiver error, negative controls and total cost.
- Do not hide far-away invalidated objects to make selective speedups look larger.
- If object models are not independently trainable/composable with physically meaningful GI, stop and attribute a **model ownership** problem before implementing a hierarchy.
- If the oracle dirtiness set is effectively the whole scene, stop and report the transport coupling rather than optimizing a mask threshold.
- A hierarchy is a candidate acceleration structure, not the primary architecture hypothesis.
- No automatic proceeding from an oracle gate to a full dynamic framework or paper-success claim.

## 9. One-Line Research Definition

> **Preserve high-quality neural transport scene-wide by tracking which objects' pretrained transport has become invalid after geometry changes, fully recomputing only those object-owned neural models, and reusing all models that remain radiometrically valid.**
