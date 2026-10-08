# Baseline Roles and Evidence Strategy — Selective Neural Transport Reconstruction

**Original research operating decision:** 2026-09-28  
**Last updated:** 2026-10-08  
**Status:** Active; oracle selective-refit gate F1 run once (worklog 30): negative on the 8DNA teaset triplane

## 0. Why Baseline Roles Must Be Separated

This project has strong evidence that learned transport can go stale under changed internal geometry, but **no positive evidence yet that selective neural-state refit is viable**.

Keep three distinct proof obligations:

1. **Phenomenon:** Does a pretrained neural transport representation fail to follow a physical configuration-dependent GI change?
2. **Selective-reconstruction feasibility:** Can only affected learned-state units be refit while preserving unaffected transport and saving quality-matched cost?
3. **Deployable invalidation:** Can an actual non-oracle geometry-edit detector find invalid units efficiently and accurately?

Passing one does not establish the others. A synthetic correctness test is not a real-scene architecture proof.

---

## 1. Historical Problem-Evidence Lineages (Preserve)

### 1.1 RNA / Rain — Historical diagnostics

RNA/Rain exposed frozen-deformation failures but had local query/canonical-location confounds. Its fixed-target mechanism probe had a physically weak radiance signal and must not be presented as clean causal proof of missing nonlocal transport.

Role: implementation/correspondence/control methodology and historical phenomenon motivation. Do not reopen it just to obtain a positive result.

### 1.2 8DNA / Teaset — Primary physical stale-state anchor

WL21: stationary receiver surface maintains identity/query while another rigid part changes relation. GT indirect transport changes substantially; frozen 8DNA barely tracks it. Relation-preserving T1 acts as a control. WL27 reproduced an envmap same-state T3 scratch reconstruction near T0 quality; early threshold crossing at 38.9 min was retrospectively identified, not a deployable time-to-recovery. Full schedule ~1.76 h.

Role: clean causal stale-state mechanism and historical reconstruction/cost evidence.

### 1.3 RNA / Teaset — Same-scene independent-backbone phenomenon replication

WL22: frozen RNA replicates the nonlocal failure signature under common lighting. RNA static-quality margin is weaker than 8DNA. Common-light rebuild success was not robust: WL22 RNA had borderline recovery, WL27 independent full schedule (~3.24 h) had no rule-satisfying checkpoint.

Role: cross-backbone **problem replication**, not proof that a new solution transfers between two representations.

### 1.4 WL24–25 physics/code attribution

Missing transport mixes indirect visibility/occlusion and reflected radiance from the moving part; unchanged receiver local geometry and current direct-visibility features do not carry the needed nonlocal GI. Neither asset result justifies claims about every neural renderer.

### 1.5 WL28–29 previous architecture candidates

- WL28: K=32 geometry-only relation state insufficient for held-out T3; richer reference-derived transport oracle through frozen RNA shared operator works.
- WL29: same K=32 plus aligned runtime frozen-RNA radiometric proxy offers no benefit over zero/shuffled; even exact directional radiance on the same directions fails; proxy state extraction ~3.2 s.
- Local state, angular support, aggregation, and training-relation coverage are not interchangeable causes. These results falsify the specific tested representations, **not** the whole lifecycle split and **not** the new selective-refit candidate.

Previous RNA sidecar is not automatically the new substrate: partial neural-state refit needs identifiable localizable **trainable GI units**, unlike an unchanged RNA asset with a sidecar residual.

---

## 2. The New Method-Feasibility Substrate — ONE TESTED (WL30), NEGATIVE

**Worklog 30:** the released 8DNA teaset asset served as the F1 substrate (criterion-based choice recorded in `experiments/selective_refit_oracle/protocol/sro_v1.json`); Neural Radiosity was audited at code level only and deferred (runtime port and T0/T3 quality unvalidated on this hardware). 8DNA's role therefore extends from problem evidence to **one tested F1 substrate (negative)**; it is still not the method-development base. The negative result is scoped to its triplane state and shared-decoder ownership (section 7, "D fails while C succeeds").

The eligibility list below still applies to any further substrate:

Prefer an **existing pretrained per-scene neural GI model** with:
- credible T0 static transport fidelity relative to the changed physical signal;
- high-frequency glossy/interreflection or multi-bounce transport;
- localizable learned GI parameters (grid, surface/vertex features, patch blocks, etc.);
- shared/global parameters that can be held fixed;
- measurable parameter support under interpolation and hashing;
- reproducible full train, warm-start refit and targeted optimizer masking;
- tractable training and reference data on available hardware.

Audit at most two candidate implementations before selection in the F0+F1 batch.

Do not choose by model publicity or select RNA/8DNA only because they were historical baselines. Do not create a new neural GI representation in order to satisfy this first gate.

If none qualifies, STOP and report a substrate blocker. This is neither proof nor disproof of general selective neural transport refit.

---

## 3. Required F1 Controls (One Substrate / One Scene)

| ID | Baseline | Initialization | Trainable parameters | Scientific role |
|---|---|---|---|---|
| A | Frozen T0 | pretrained T0 | none | establishes stale-state baseline |
| B | Scratch full T3 | scratch, same architecture | all ordinarily trainable state | new-configuration reconstruction capacity / quality upper reference |
| C | Global warm-start T3 | exact pretrained T0 | all ordinarily trainable state | strongest obvious update/continual adaptation alternative |
| D | Global localized-state T3 | exact pretrained T0 | all eligible local learned-state units, global shared phi frozen | isolates freeze-shared-parameters from selective masking |
| E | Oracle selective-state T3 | exact pretrained T0 | only reference-derived selected local units, phi and unselected units frozen | tests whether selective refit is possible |

Optional diagnostic: equal-size simple spatial-neighborhood mask, without introducing a new hierarchy.

**Fairness:** compare C/D/E with matched supervision, data splits, optimizer schedule, reference, hardware and evaluator. D and E must match trainable-state family and frozen-global contract; only the update mask differs.

Report separately at fixed training budget and at matched recovered quality. Do not select checkpoints or thresholds based on final held-out T3 evaluation samples.

**Static-quality prerequisite:** T0 pretrained fidelity must resolve the GT T0→T3 signal. If not, stop prior to candidate verdict. A full T3 scratch fit must demonstrate the chosen substrate can represent the target configuration; otherwise selective failure may be capacity failure.

---

## 4. Oracle and Invalid-State Accounting

The oracle can consult high-quality GT at T0/T3 to create a diagnostic affected receiver/query map. This information is allowed in F1 **only to choose the experimental update set**, not as a method-level runtime feature.

The receiver change mask must be mapped to real learned parameter support; report any feature interpolation/hash collisions, global parameter dependencies, and geometry/radiometric influence beyond immediate mover bounds.

A reference-derived affected map is NOT guaranteed to be perfect, especially for:
- weak-but-nonzero transport changes near the reference noise floor;
- distant glossy/multi-bounce influence;
- state units shared across affected and unaffected evaluations;
- feature hash collisions and learned decoder coupling.

Use disjoint render/validation samples when constructing mask and final quality evaluation. Report mask size, affected query coverage, potentially missed state units and over-invalidation.

Do not use final T3 evaluation radiance to tune mask thresholds, optimizer parameters or checkpoint selection. Report the oracle-building cost separately and do not include it silently in a deployable efficiency claim.

---

## 5. Performance / Physical Reference Baselines

**Physical full computation (historical context):** WL26 BMW27 1080p Cycles ~0.22/0.76/3.0 s at 16/64/256 spp on RTX 5080. Different scene/reference and not quality matched to selective-refit tests.

**Neural full regeneration (historical context):** WL27 8DNA envmap T3 scratch first isolated oracle-identified recovery ~38.9 min; fixed schedule ~1.76 h. RNA common-light scratch fixed schedule ~3.24 h with no criterion pass. These pipelines do NOT constitute a global warm-start baseline.

**Required method-level comparators:** same-substrate same-scene B/C/D, end-to-end wall-clock to matched quality, plus optical/physical reference where quality permits. For practical comparisons in later F3/F4, additionally assess NRC/NIRC online cache adaptation, static+dynamic residual GI/Hybrid Rendering, per-frame GI probes and relevant incremental MC as the claims demand.

No method can be declared faster by measuring only optimizer kernel time while ignoring target generation, transfer, masks, validation or loading. Nor is lower parameter count automatically faster.

---

## 6. Metrics and Reviewer Exports

**Transport validity:** affected stationary receiver signed T0→T3 change error, relative physical-change error and GT noise floor.

**Reconstruction:** full T3, global warm refit, global-localized refit, and selective refit absolute and change errors in the same regime.

**Preservation:** unaffected query error delta, unintended radiance change, frozen parameter identity, overlap leakage, boundary seams and temporal stability if tested.

**State locality:** eligible unit count, selected count and percentage, genuinely changed units, support coverage, multi-bounce invalidation expansion and memory.

**Cost:** full quality-vs-wall-clock curves; optimization/target generation/mask generation/validation/memory/inference disaggregated; actual matched-quality end-to-end savings vs C and D. A screening target of ~2x vs D with near-D quality is optional only if predeclared.

**Qualitative:** same-exposure GT/frozen/full/global/localized/selective renders, reflection and indirect-occlusion bands, signed difference/error maps, mask/state-support visualizations, unrelated stationary regions and seams.

No synthetic-only architecture pass; no "looks similar" without provenance and error accounting.

---

## 7. Interpretation / Stop Rules

- **F0 failure:** no existing substrate with enough quality and localizable state → report substrate limit; do not build another renderer.
- **B fails:** substrate cannot fit T3 → cannot attribute E failure to selectivity.
- **D fails while C succeeds:** fixed shared parameters / local-only representation are insufficient → conditional substrate-ownership issue, not oracle mask proof.
- **E fails while D succeeds:** mask support, parameter locality, multi-bounce closure, data/optimization budget must be disentangled; do not automatically expand masks.
- **E succeeds but requires nearly all units:** selective locality may be economically meaningless.
- **E quality succeeds, speed fails:** update mechanism is not a practical selective-recompute result even if parameter masking is correct.
- **E passes quality and cost:** authorize only a new bounded investigation of actual non-oracle learned-state dependency. Do not declare an entire hierarchy, framework or SIGGRAPH success.

Always separate IMPLEMENTATION FACT, MEASUREMENT, OBSERVATION, INTERPRETATION, UNRESOLVED QUESTION and ARCHITECTURE DECISION.

Stop when the predeclared gate condition is met; do not rescue a negative mechanism with unapproved heuristics, threshold tuning, more model capacity or repeated unbounded optimization.

---

## 8. Generality and Publication Claims

One well-controlled teaset case can establish a mechanism, not broad dynamic-GI generality.

A final selective-recompute method should eventually face independent scenes, high-frequency glossy paths, far-field/multi-bounce dependencies, several movers, edit magnitude variation and temporal animation. Nonrigid/contact/fold changes are important long-term but not F1.

The strongest defensible paper claim must show:
- novelty beyond classic hierarchy / incremental GI / NeLT / Superposed Deformable Feature Fields;
- genuine learned-state invalidation and partial refit rather than using oracle at runtime;
- high-quality GI restoration;
- quantitatively bounded missed dependencies and unaffected-state corruption;
- cost advantage over same-substrate global adaptation at matched quality.

Cross-backbone problem evidence from WL21–22 is not cross-backbone validation of a new selective-recompute method.

---

## 9. Current Decision

**DONE (worklog 30):** one bounded F0+F1 oracle selective-state refit experiment, 8DNA teaset, envmap regime. Outcome: G0, signal and scratch-ceiling gates pass; D (all localized state, shared frozen) recovers only 0.20× of global warm start's stale-error reduction (gate `G_D_meaningful` fails in both seeds); E ≈ D, with unaffected-region degradation (+7.5%) and no cost advantage (2× slower than D to the D-matched target). Cause: shared-decoder ownership plus triplane projection coupling — a substrate result.

**PENDING USER DECISION:** whether to audit a second substrate with 3D-local, decoder-independent state under the same protocol, or to reassess the direction. F2 is not authorized.

**NOT ACTIVE:** K=32 direction-state tuning; RNA sidecar extension; object hierarchy; dependency-predictor implementation; full new renderer; whole-scene dynamic neural GI framework.

**Completion question:**

> With reference-derived oracle invalidation guidance, can we selectively refit a strict subset of pretrained high-quality neural GI state and recover current transport while preserving valid state, with meaningful quality-matched update-cost savings?

A negative answer is a valid result and must be interpreted before selecting the next research hypothesis.
