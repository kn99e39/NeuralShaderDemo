# Architecture — Object-Level Selective Neural Transport Recomputation

**Updated:** 2026-10-12  
**Status:** Architecture intent CLARIFIED; oracle object-level feasibility NOT YET TESTED  
**Branch:** Selective_Recompute  
**Historical references:** docs/history/ARCHITECTURE_LIFECYCLE_SEPARATION_2026-10-07.md and docs/history/ARCHITECTURE_INTRA_OBJECT_FEATURE_REFIT_2026-10-08.md

## 1. The Actual Architecture Question

After an edit to scene geometry, which **objects have transport-invalid learned neural models**, and can we rebuild the **complete models for only those objects** while preserving the complete models of transport-valid objects?

**Selective unit: Object O_i. Recompute unit: Entire object-owned learned neural transport model N_i.**

The user did NOT request selective recomputation of individual triplane cells, feature channels, learned vertices, MLP layers, or a subset of parameters within a neural asset.

If O_i's learned transport is invalid, its whole model may be retrained/refit, including all triplane features, shared-within-object MLPs, decoder, directional parameters and relevant object-owned caches. If O_j remains valid, its entire N_j must be preserved. Refit from pretrained weights or scratch within an invalid object are both possible; compare rather than silently assuming one.

## 2. Scene / Ownership Model

Represent the scene as objects O={O_1,...,O_n} with geometry G_i and associated **independently owned** learned transport state N_i.

~~~text
T0 scene geometry G0
   -> learn a separate high-quality transport model N_i^0 for each O_i
      (each model supervised by physically correct transport in the WHOLE scene)

A geometry edit produces G1
   -> identify which objects are geometry dirty: D_G
   -> identify which objects' LEARNED TRANSPORT is stale: D_T
      (includes stationary receivers affected by other objects)
   -> for i in D_T: full-object recomputation N_i^0 -> N_i^1 on G1
   -> for j not in D_T: preserve N_j^0, including all its parameters
   -> compose final GI from updated and preserved object models at G1
~~~

D_G and D_T are not identical. Example: moving the milk pot can invalidate transport on stationary teapot/tray models due to indirect visibility and glossy reflection. It is incorrect to recompute only the moved object when stationary receivers have changed transport.

**Object-level reuse is the intended contribution boundary**, not a promise that a model's features are local.

Mathematical notation: let R(G,{N_i}) denote the renderer composing object-owned learned transport under current geometry G; D_T(G0,G1) is a conservative set of objects for which retaining N_i^0 would violate a radiometric validity criterion. Define the candidate updated models as

~~~text
N_i^1 = FULL_REFIT(N_i^0, G1, supervision_i(G1))    when i belongs to D_T
N_i^1 = N_i^0                                       otherwise
~~~

Here FULL_REFIT means **all model parameters of that object may change**, not necessarily initialization from scratch. The supervised target for O_i may involve light paths through every other object of the scene.

## 3. What a Valid Per-Object Neural Model Must Mean

The architecture cannot merely split geometry into names while retaining one indivisible scene-wide MLP.

For each object establish:

- **Ownership**: a distinct neural model/checkpoint N_i, with all trainable tensors attributable to O_i. Optionally a globally frozen reusable encoder/decoder, if its output is valid under updates; an editable shared scene-wide model defeats strict per-object freezing unless safely partitioned.
- **Radiometric semantics**: identify whether N_i predicts outgoing radiance, incident GI, transport function, or another scene-conditioned quantity; whether changing scene lights requires an update; which transport orders it covers.
- **Cross-object transport**: N_i can encode effects originating from other objects. Per-object storage does NOT imply per-object radiometric independence.
- **Geometry/correspondence**: robust object identity across rigid transforms, changes of geometry, visibility and interaction.
- **Composition**: renderer must evaluate each N_i on its owning surface and combine current geometry, direct transport, learned indirect transport and any cross-object terms without double-counting.
- **Update contract**: freeze all N_j for j not in D_T (bitwise model parameters), optimize **all** trainable parameters in invalidated N_i, and evaluate downstream consistency.
- **Multi-bounce**: geometry edit may create indirect dirty receivers arbitrarily far away or across chains. A recomputation closure must be considered; object hierarchy alone does not guarantee sparse D_T.

RNA, 8DNA, triplanes, vertex features, octree directional maps and spatial-angular primitives are candidate **internal representations of each N_i**. Do not preemptively reject them because of intra-object feature locality results. No requirement to invent a new spatial/angular basis. Per-object data assembly and transport/lighting semantics, not a feature-level encoder, are the primary design tasks.

## 4. Prior Evidence Reinterpreted Without Changing Historical Results

### Worklogs 21–27 — VALID PROBLEM EVIDENCE

The canonical teaset first-hit receiver is unchanged yet nonlocal indirect GI changes after a part moves. The current frozen RNA and 8DNA variants lack an updated representation of intra-asset transport and fail to respond appropriately. Historical all-asset retraining can be expensive. This establishes motivation, not an object-model update result.

The historical 8DNA teaset model represented the *whole multi-part asset* as one learned transport owner. Therefore the old T3 refit is not a test of scene-level object selection. It cannot measure preservation of separate valid object models because none were partitioned that way.

### Worklogs 28–29 — SEPARATE ON-HOLD DYNAMIC-STATE HYPOTHESIS

K=32 sparse current relational/radiometric-state candidates did not recover held-out near-mirror T3. Do not tune those results as a substitute for the object-level selective test.

### Worklog 30 — MIS-SCOPED FOR THE CURRENT GOAL

Worklog 30 validly tested triplane **cell-level masking** in one 8DNA model with shared flow/cubemap parameters frozen. It failed on that feature-level architecture. In the intended object-level scenario, an invalid N_i can have **all its triplane and flow/MLP weights fully refitted**, so the "D triplane-only did not approach C" conclusion does not invalidate object-level selective recomputation.

Retain Worklog 30, its code, and previous intra-object Architecture in history. Do not rewrite Worklog 30's measurements as successes or pretend the gate tested the present hypothesis.

## 5. Current Design Hypothesis — Object Transport Ownership

> Scene-scale learned transport can be maintained incrementally when learned transport models are owned by identifiable objects and only models with invalid radiometric scene dependencies are fully recomputed after geometry edits, rather than globally rebuilding the entire scene's learned transport.

This is **unvalidated** and depends on:
1. good baseline fixed-scene quality for object-owned neural models;
2. fair objectwise all-rebuild reference;
3. correct radiometric dirty-object closure after edits;
4. compositional consistency of updated and reused model predictions;
5. meaningful end-to-end cost savings when D_T is smaller than the whole scene.

Failure of any one is material. A scene where almost every object becomes dirty is a valid negative result.

## 6. Next Controlled Gate — Object-Level Oracle Selective Rebuild

**Central Intent:** Does object-level selection itself work if the complete transport-invalid object set is provided by an offline reference oracle?

### Substrate / scene contract

Select or adapt a neural transport model whose entire learned weights can belong to one scene object. A per-object RNA/triplane is permitted, as are other existing codebases. No requirement of independently maskable feature subsets.

Construct at least one **multi-object** controlled scene, with several radiometrically distinguishable owners:
- moving geometry object;
- stationary indirectly affected receiver(s);
- stationary transport-valid object(s) as a real preservation/control case;
- high-quality glossy or multi-bounce reference signal above noise;
- separately reconstructible object-owned models supervised against the **whole current scene**.

Prefer reuse of canonical teaset physical edit if the partition and independent models are scientifically sound. If the canonical scene has no truly transport-valid object, select an additional valid control object/scene rather than quietly dropping the preservation test.

### Oracle object dirty set

Use a reference-derived, conservative, **object-level** D_T^oracle. Compare high-quality current-geometry reference transport for each object using stable object-identity surface samples, including indirect effects on stationary objects. The actual threshold, noise handling and classification protocol must be predeclared before final runs.

The oracle is for scientific upper-bound attribution only. It is not a production transport dependency tracker; measure its time separately. Recognize that changed pixels and model-invalid objects are not identical; verify output validity per owner after composition.

### Matched comparison arms

| Arm | Training / update | Purpose |
|---|---|---|
| A Frozen scene | preserve all T0 pretrained N_i; render G1 | stale-baseline quality |
| B All-object scratch | train ALL N_i from scratch at G1 | matched new-configuration capacity/cost reference |
| C All-object warm start | fully refit ALL N_i from their T0 checkpoints on G1 | strongest obvious full-scene adaptation comparator |
| D Geometry-dirty only | fully refit N_i for i in D_G; freeze others | demonstrate why mover-only selection may miss stationary receiver transport |
| E Oracle transport-dirty | fully refit N_i for i in D_T^oracle; freeze all other complete N_j | intended selective architecture feasibility |
| F Optional naive spatial proximity | fully refit models selected by geometric closeness, with declared equivalent budget/criteria | simpler dependency baseline |

**Changed between C and E: only which WHOLE OBJECT MODELS are selected for complete refit.** Each selected object's internal training architecture, trainable parameter coverage and supervision remain the same as C.

Do NOT perform an intra-object mask, freeze selected object MLPs to mimic triplane-only update, or force 8DNA/RNA's historical whole-asset checkpoint to masquerade as several independent neural models.

### Evaluation

Quantitative:
- object counts total, D_G count, oracle D_T count, valid retained count;
- per-object and cross-object transport-change magnitude;
- radiance reconstruction for directly edited, indirectly affected stationary, and unaffected objects separately;
- global image quality, signed transport-change tracking, glossy/multi-bounce behavior;
- model parameter identity for valid retained objects, output-level validity after scene composition;
- missed affected objects / dependency closure, fidelity vs all-object adaptation;
- total supervision generation, optimizer training, model load/store, rendering, scene acceleration/update, oracle cost separately, full wall-clock to matched quality, peak GPU memory;
- scene-size and affected-object fraction scaling if feasible.

Qualitative:
- changed mover;
- stationary changed receiver;
- unchanged valid object;
- indirect reflection/occlusion and high-frequency features;
- error/difference maps and object ownership visualization.

Synthetic contracts plus **real-scene multi-object** evaluation; synthetic PASS does not establish method viability.

## 7. No Hierarchy Yet: Dependency Tracking Is the Next Question

If and only if oracle selective whole-object refit works, test a deployable geometry-to-transport dependency/invalidation mechanism. Use classic incremental GI, light-path/visibility links and object hierarchies as prior-art comparators, not as new discoveries.

A hierarchy is optional optimization, not a prerequisite. Where multi-bounce paths connect nearly all objects, report dense invalidation rather than hiding quality errors.

## 8. Explicit Non-Goals for the Next Batch

Do NOT:
- add per-feature masks or optimizer subset tuning;
- reattempt Worklog 30 triplane-locality controls as the primary gate;
- invent a new neural spatial-angular representation merely to guarantee parameter locality;
- restrict updates to geometry-dirty objects when stationary receivers are transport dirty;
- omit the full scene during supervision tracing for an object whose correct radiance depends on other objects;
- promise realtime update without measuring training/target generation;
- claim novelty for classic object hierarchy / incremental GI;
- implement deployable dependency tracking before object-level oracle evidence;
- silently treat object-wise scratch and warm-start as identical costs;
- proceed automatically to full dynamics / final method if the gate passes.

## 9. Completion Condition

> **When oracle identifies all transport-invalid object models, can completely recomputing only those object models preserve the valid models and recover current-scene high-quality GI at substantially lower end-to-end cost than recomputing all scene-object models?**

A negative result requires precise attribution to scene partition, model semantics, dependency closure, insufficient quality, or update economics—not intra-object feature locality by default.
