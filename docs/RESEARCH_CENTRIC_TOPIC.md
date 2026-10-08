# Research Centric Topic — Reusable High-Quality Neural Light Transport Under Geometry Edits

**Last updated:** 2026-10-08  
**Status:** Active research problem; conditional selective-reconstruction direction; method NOT validated  
**Target:** SIGGRAPH 2027 planning target (official submission date and acceptance not assumed)

## 1. Central Problem (Stable Across Architecture Changes)

High-quality pretrained neural light-transport representations can compress interreflection, self-shadowing, near-field transport, glossy reflection, complex appearance, and higher-order scattering in a fixed geometry configuration. Some of that compressed transport may cease to be valid when geometry relationships change after training.

**Central problem:**

> Under a geometry edit, which high-quality learned transport information remains valid, which becomes stale, and how can the valid information be reused without regenerating the entire transport representation?

This is a **light-transport validity and reuse** problem, not generic dynamic geometry rendering. It is not a claim that every neural renderer fails on dynamic scenes.

A controlled stale-state failure requires stable surface identity, correct current geometry/correspondence and lighting, a physically meaningful indirect transport change, and a frozen learned response that fails to follow the change. Implementation or feature-coordinate errors alone do not establish the phenomenon.

## 2. Present Research Direction (Conditional)

The current primary **candidate to evaluate** is:

> Preserve a high-quality pretrained **per-scene or internally multi-part learned transport state**, identify transport-state units invalidated by an unforeseen geometry edit, and reconstruct only those units while preserving unaffected high-quality transport.

Conceptual sequence:

~~~text
High-quality pretrained learned transport
    + geometry edit
    -> affected learned-transport state
    -> selective invalidation
    -> selective neural-state reconstruction / refit
    -> updated transport with unaffected pretrained state preserved
~~~

The scientific contribution, if achieved, must concern **learned-state dependency, invalidation semantics, local refit and quality/cost preservation**. An object hierarchy alone is not novel and cannot be used as the contribution claim.

**Present gate is narrower:** before constructing an operational dependency hierarchy, determine whether selective reconstruction is viable even with an offline reference-derived oracle affected-state mask. A positive oracle result is a necessary feasibility indication, not validation of the complete method.

The previous **Persistent Appearance + Current Configuration-Dependent GI State + Shared Transport Operator** direction is **ON HOLD**, not falsified. Its historical specification is preserved in docs/history/ARCHITECTURE_LIFECYCLE_SEPARATION_2026-10-07.md; Worklogs 28–29 tested two particular K=32 representations and do not falsify all lifecycle-separated representations.

## 3. What the Existing Evidence Actually Establishes

- **Worklogs 21–22:** stable stationary teaset receiver queries display a large physical T0→T3 indirect transport change after another internal part moves, while frozen 8DNA and RNA respond poorly; the relation-preserving T1 is an important control.
- **Worklogs 24–25:** the physical change involves indirect-path occlusion and moved-part radiance; stable local query inputs and current direct visibility do not supply the missing nonlocal information.
- **Worklog 27:** historical scratch T3 regeneration is offline (8DNA envmap first isolated oracle-identified recovery 38.9 min, full schedule ~1.76 h; RNA common-light full schedule ~3.24 h without rule-satisfying recovery). These are not warm-start/selective-refit timings.
- **Worklog 26:** physical Cycles after geometry change took ~0.22 / 0.76 / 3.0 seconds at 1080p, 16 / 64 / 256 spp on a different scene. These are not quality-matched to the neural results.
- **Worklog 28, CASE C:** the first K=32 geometric relational sidecar fails held-out T3; a richer reference-derived oracle can recover the change through a shared operator.
- **Worklog 29, CASE D:** aligned frozen-RNA radiometric proxy is not better than ZERO/SHUFFLED controls and is slow (~3.2 s update); even exact radiance on the same K=32 support fails. Angular support, aggregation and training coverage remain confounded. These failures do not disprove selective neural-state reconstruction.

Evidence scope: one clean teaset asset family and cross-backbone problem replication; not a general theorem for dynamic neural GI, all materials, or arbitrary deformation. Rain served as a mechanism-development bench but its clean fixed-target physical signal was too weak.

## 4. Novelty Boundary From Literature Kill Search

**Already established prior art:**
- Classical incremental/hierarchical GI dependency and selective recomputation (Drettakis–Sillion 1997; Bala et al. 1999; Luksch et al. 2019).
- Object-oriented learned transport and dynamic object-conditioned inference (NeLT, TOG 2023; Superposed Deformable Feature Fields, SIGGRAPH Asia 2024).
- Production dependency-aware incremental precompute at authoring time (Enlighten).
- Online neural GI adaptation/caching, static/dynamic residual rendering, dynamic PRT and geometric acceleration structure refit.

**Potential surviving technical questions (not proven novelty claims):**
1. Construct or infer radiometrically meaningful dependencies **for learned transport state units**, which do not necessarily have explicit path provenance.
2. Map a geometry edit to stale learned-state units without updating nearly the entire representation or missing distant indirect effects.
3. Selectively refit directional / glossy / multi-bounce learned transport without damaging the unaffected state, with measurable cost benefit.
4. Demonstrate that learned-state validity and local parameter support can be reconciled under actual neural optimization.

Do not claim novelty for object hierarchy, spatial locality alone, generic cache invalidation, partial GI recomputation, object-wise composition, or persistent-plus-dynamic partitioning.

Some dynamic neural systems already predict high-frequency GI for moving objects within trained configuration distributions. Our potential claim must be **unforeseen post-training geometry edits and incremental learned-state repair**, not simply "dynamic neural GI is impossible."

Selected references: [Drettakis–Sillion 1997](https://maverick.inria.fr/Publications/1997/DS97/dret.pdf), [NeLT 2023](https://doi.org/10.1145/3596491), [Superposed DFF 2024](https://doi.org/10.1145/3680528.3687680), [GLTE preprint](https://arxiv.org/abs/2510.18189), [Neural Radiosity](https://arxiv.org/abs/2105.12319). Preserve uncertainty where paper details are unverified; do not invent absence of prior art.

## 5. Scientific Meaning of State and Dependency

Keep these distinct:

- **Geometry unit:** object, part, patch, voxel, triangle, or spatial support affected by motion.
- **Receiver transport sample:** a surface position/direction whose physically correct outgoing radiance can change.
- **Learned state unit:** independently identifiable and possibly trainable grid entry, patch feature, latent block, etc.
- **Parameter support:** which predictions depend on a learned unit (including interpolation/collisions).
- **Radiometric dependency:** how a geometry edit can invalidate predictions associated with a state unit through direct or multi-bounce light paths.
- **Update set:** units actually modified during refit.
- **Oracle mask:** offline reference-derived affected estimates, not a deployable dependency detector or automatically a correct parameter mask.

A changed pixel set is not automatically a learned-state invalidation set. Simple object containment or Euclidean distance cannot represent every nonlocal transport dependency.

## 6. Current Scoped Investigation

**Immediate test:** an existing spatially/structurally localizable, high-quality **per-scene** neural GI representation; controlled T0 pretrained state and held-out T3 geometry edit; oracle-assisted selective refit vs frozen, full retrain, global warm-start and globally refit eligible local units. Preserve shared weights in controls that test localizable-state updating.

Begin with rigid cross-part relative motion and a stationary indirect receiver. Require a radiometrically meaningful reference signal and genuine glossy/multi-bounce effect. The original teaset is preferred if the selected substrate can model it; do not force an incompatible renderer to reproduce the scene.

Later, only if feasibility passes: develop actual dependency/invalidation, expand to independent scenes, more geometry edits, contact/folds/nonrigid motion, scale and temporally varying updates.

No requirement to implement object hierarchy, general neural renderer, NURBS, full inverse rendering, a new BRDF model, or cross-representation framework in the first gate.

## 7. Evidence and Attribution Rules

- Synthetic fixtures verify contracts, not real-scene viability.
- Compare method costs and quality **on the same selected substrate, scene, reference and training regime**. The historical RNA/8DNA scratch costs cannot be a speedup denominator for another model.
- Compare affected, unaffected and boundary regions independently; show signed change maps, forgetting/seams, update-set coverage, quality-vs-wall-clock and actual updated-state fraction.
- An oracle may use held-out GT to define a diagnostic mask but this cost and reference dependence must be disclosed; hold back separate evaluation data and do not call oracle timing real runtime performance.
- An optimization-only speedup is not an end-to-end speedup; count supervision, data movement, validation and update overhead separately.
- Negative evidence is scoped. An unsuitable substrate or parameter coupling does not universally refute selective neural transport reconstruction.
- Do not tune thresholds, edit scenes, add heuristics or change evaluation splits after looking at the final result.

## 8. Stop / Advancement Rule

**Do not build a learned transport hierarchy unless oracle-guided partial neural refit first achieves meaningful affected-region recovery, unaffected-state preservation and quality-matched cost reduction relative to appropriate global controls.**

If oracle selective refit fails: attribute whether the cause was insufficient static baseline, learned-parameter nonlocality, invalidation mapping, multi-bounce closure, optimizer coupling, or lack of cost benefit. Stop and report; do not automatically redesign a new substrate in the same batch.

If it passes: open a separate controlled batch for practical geometry-to-learned-state radiometric dependency construction. Success is not a full method or paper acceptance.

## 9. One-Sentence Research Definition

> **Develop and validate a mechanism for preserving high-quality pretrained neural transport under unforeseen geometry changes by identifying and selectively reconstructing invalid learned transport state, without paying the full cost of recomputation or corrupting valid state.**
