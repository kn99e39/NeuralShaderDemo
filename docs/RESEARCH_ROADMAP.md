# Research Roadmap — Object-Level Selective Neural Transport Recomputation

**Updated:** 2026-10-12  
**Branch:** Selective_Recompute  
**Status:** Object-level intent clarified; first matched whole-object oracle experiment NOT YET RUN  
**Venue planning:** SIGGRAPH 2027, official paper deadline to be separately verified.

## 0. Research Decision / Scope Correction

The user explicitly clarified:

> Selective recomputation means **fully recompute neural transport for the scene objects whose learned transport was invalidated by a geometry change**, preserving the complete neural transport models of objects that remain valid.

**Selection unit = scene Object; update unit = entire object-owned neural model**, which may include every feature grid, triplane, MLP, flow, or directional state owned by that object.

This corrects the previous assistant interpretation of "selective" as an **intra-object subset of triplane cells**. Worklog 30 is a valid negative test of its narrower feature-level hypothesis but **NOT a test or rejection of this object-level approach**. The prior intra-object architecture is archived at docs/history/ARCHITECTURE_INTRA_OBJECT_FEATURE_REFIT_2026-10-08.md.

Existing dynamic-state architecture history remains at docs/history/ARCHITECTURE_LIFECYCLE_SEPARATION_2026-10-07.md.

Read docs/RESEARCH_CENTRIC_TOPIC.md, docs/Architecture.md and docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md as the active contracts.

## 1. Central Intent

> Does a scene composed of independently owned high-quality neural transport models admit correct and cheaper updates when geometry edits invalidate only a subset of **object models**, provided we fully retrain/refit each invalidated model and keep valid models frozen?

Scientific target:
1. **Transport-dirty detection:** distinguish directly changed geometry objects D_G from radiometrically stale object models D_T. Stationary surfaces can become stale.
2. **Whole-object ownership:** an affected object can update every parameter in its object model; unchanged object models are reused exactly.
3. **Transport correctness:** each object must model its incoming/outgoing indirect transport conditional on the current *whole scene*, not on its geometry alone.
4. **Update economics:** compare end-to-end objectwise selective rebuild against recomputing all object models at quality matched to physical reference.

**A dependency hierarchy may later accelerate discovery of D_T.** Dependency-guided selective GI recomputation is classic prior art; novelty must be a credible neural learned transport specific mechanism/evidence, not "objects update selectively."

## 2. Historical Evidence: Preserve / Interpret Correctly

| Worklogs | Established | Not established |
|---|---|---|
| 21–22 | Canonical teaset stationary receiver GI becomes stale when internal part moves; frozen RNA and 8DNA signatures | Objectwise learned transport update |
| 24–25 | Dominant change is indirect occlusion and moved-part reflection; absent current cross-part information | Every dynamic neural GI method must fail |
| 26–27 | Physical and historical neural whole-asset recomputation can be costly (different scenes/regimes) | Quality-matched per-object incremental cost |
| 28–29 | Two sparse K=32 current relational-state candidates fail held-out near-mirror T3 | Entire lifecycle split falsified |
| 30 | Selecting only 8DNA triplane cells with shared weights frozen fails (quality/preservation/time) | **Object-level full-model selective recomputation fails** |

WL30's triplane-only D/E are not substitutes for the new oracle **object set** selection experiment; the historical multi-part teaset's learned model was single-owner.

## 3. New Ordered Gates — Do Not Jump Ahead

| Gate | Status | Question | Stop condition |
|---|---|---|---|
| P — Stale transport phenomenon | CLOSED (21–25) | Does nonlocal GI become stale under cross-part editing? | Historical result retained |
| C — Cost motivation | CALIBRATED (26–27) | Is full recomputation expensive? | Not a method-level denominator |
| H — Earlier compact state | CLOSED NEGATIVE for tested cases (28–29) | Did K=32 candidates recover T3? | No patch/sweep |
| X — Intra-object feature masking | CLOSED NEGATIVE for 8DNA triplane (30), **off-target** | Are selected feature cells sufficient? | Not an objectwise decision gate |
| O0 — Per-object learned transport feasibility | **NEXT** | Can independent object-owned neural models represent correctly composed full-scene GI? | If high-quality composition impossible, stop |
| O1 — Oracle objectwise selective full refit | **NEXT, conditional on O0** | Can every dirty object's whole neural model be retrained while valid objects remain fixed, at lower quality-matched cost than all-object rebuild? | If failed, stop, attribute |
| O2 — Actual transport dependency / hierarchy | FUTURE; requires O1 evidence and user's decision | Can D_T be determined online without oracle? | O1 not supported |
| O3 — Integrated larger scenes and independent benchmarks | FUTURE | Does complete system scale and preserve glossy/multi-bounce transport? | O2 not supported |
| O4 — Paper-level validation | FUTURE | Is there robust novelty and evidence for SIGGRAPH? | Insufficient breadth, quality, controls |

## 4. Immediate Bounded Batch — O0 + Conditional O1

### Direction

Test scene-level selection of **whole object models** after learned transport invalidation. Reuse RNA/triplane/8DNA or other available neural GI components as appropriate; coding difficulty alone should not force the substrate choice.

### Purpose

Establish whether object-owned learned transport can be trained independently yet correctly reflect full-scene interactions, and whether retraining only radiometrically dirty object models saves real work.

### Central Intent

> If full-object neural recomputation is allowed, does selecting only transport-invalid objects preserve high-quality GI and reduce quality-matched scene update cost?

### Required model/scene design

A genuine multi-object scene with separately trainable complete model N_i per object O_i. A single large model containing many objects is *not* a valid object-level control unless separable ownership is implemented.

Each object model is supervised using reference transport in the **entire current scene** (including cross-object effects). Prefer the controlled teaset if an object partition can preserve the physical GI question. Add or choose a genuinely unaffected object to verify nontrivial preservation.

Object boundaries and transport predictions (incident/outgoing radiance, near-field, transfer, lighting conditioning) must be defined before training. If it is impossible to compose per-object models at credible T0 quality, **STOP at O0**.

### Required matched experimental arms

- **A — Frozen:** T0 object models evaluated after the G1 geometry edit, with current geometry/direct visibility as the chosen representation permits.
- **B — All-object scratch:** rebuild full models of every object for G1 from scratch; capacity/recompute-cost reference.
- **C — All-object warm-start:** fully refit every object's trainable parameters from T0 on G1.
- **D — Geometry-dirty only:** fully refit models whose geometry directly changed; intentionally test omission of stationary receivers.
- **E — Oracle transport-dirty:** fully refit the complete models of every radiometrically invalid object under G1, preserving all noninvalidated model parameters.
- **F — Optional naive proximity:** full refit on proximity-selected objects using a transparent matched rule; baseline, not a required mechanism.

Change **only which whole models are selected** between C and E. The optimizer may modify all parameters inside a selected object, not only feature cells.

Use the same substrate, object partition, scene, reference, target quality, optimizer rules and resource allocation across arms. Use fixed-budget and matched-quality comparisons. Report alternate warm-start/scratch choices transparently.

### Oracle / affected closure

Before final testing, define a conservative reference-derived oracle transport dirty-object set D_T (versus geometry-dirty D_G), based on actual physical GI differences and model validity by object. Include stationary receivers and far indirect effects. Handle uncertainty/noise and test splits explicitly. Report cases where dirty closure reaches the full scene.

Any derived cost excluding oracle construction is an **upper bound** on a future practical selective mechanism. Measure reference/GT generation and per-object target generation separately; tracing one object can still traverse every scene object.

### Quantitative + qualitative evidence

Quantitative: object count, dirty set fractions, model sizes, per-owner geometry/transport changes, per-object affected/unaffected image/surface error, signed GI change tracking, multi-bounce propagation, whole-image error, forgotten unchanged object output, cost-to-quality including scene supervision, training, checkpoint and model composition, GPU peak, reference noise, comparison with all-object rebuilt quality.

Qualitative: images and difference maps of a moved object, indirectly changed stationary receiver, truly valid preserved object, glossy reflection and other representative high-frequency features; show actual model IDs/owner provenance as useful.

Synthetic tests establish semantics only. Real-scene multi-object replay and reviewer-facing exports are required for architectural viability.

### Preserve

- Worklogs 21–30 unchanged, including their negative conclusions in their original experimental scope.
- Existing static RNA/8DNA and original scripts/checkpoints as historical baselines.
- Worklog 30's feature-masking code for archival reproducibility.
- Canonical physical reference/regime wherever scientifically compatible.

### DO NOT

- repeat triplane cell masks, frozen-within-object MLP controls or Jacobian-locality arguments as requirements for "selective";
- demand that object-internal feature-level computation be sparse;
- rebuild only the moving object if unaffected geometry receives changed GI;
- claim an unchanged geometry object always has a valid transport model;
- use one all-parts 8DNA checkpoint as multiple owned models without actual partition;
- implement scene hierarchy / dependency predictor before O1 passes;
- assume changing one object's model requires changing every other object model;
- invent new representation mechanisms unless necessary for credible independent ownership/composition, and report if scope must expand;
- tune thresholds after final metrics or automatically proceed to O2.

### Completion condition

> Can a scene of separately owned neural GI models fully recompute transport-invalid objects only and retain transport-valid models without materially losing high-quality global illumination, at a lower quality-matched end-to-end recomputation cost than rebuilding every object's model?

If no, attribute ownership failure, scene-level coupling, oracle coverage, physical quality or cost.

## 5. O2 Dependency Mechanics — Only After O1

Research actual geometry-to-object radiometric dependencies with multi-bounce closure. Compare against simple spatial proximity, object-to-object visibility/path link heuristics, classical incremental GI, and oracle. Hierarchy is optional. Claim novelty only for new learned-model validity/dependency semantics, not standard selective GI recomputation.

## 6. SIGGRAPH 2027 Planning

Replan experiment durations based on actual training/reference wall-clock and availability of coding agents. **Do not use human coding burden as the primary veto** for a scientifically better architecture. Scientific uncertainty and GPU training/quality-matching remain real costs.

Official submission dates must be independently checked. Publication aim cannot supersede predeclared stop conditions or scientific validity.

## 7. Document Synchronization and Testing

Keep docs/Architecture.md, docs/RESEARCH_CENTRIC_TOPIC.md and docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md aligned on **object-level selection**; no "feature subset F1 is the required next gate" language.

Meaningful implementation units: focused tests first; full suite only for shared production contract change or major closure. Worklogs numbered append-only and reviewers' quantitative and qualitative artifacts both required.

The next implementation batch is **not authorized automatically by this document update**; this document specifies a proposed controlled question, to be reviewed before implementation.
