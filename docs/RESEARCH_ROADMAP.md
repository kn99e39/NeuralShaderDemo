# Research Roadmap — Selective Neural Transport Reconstruction

**Last updated:** 2026-10-08  
**Branch:** Selective_Recompute  
**Status:** New direction selected for feasibility testing; **F1 tested on one substrate (worklog 30): NEGATIVE** for the 8DNA teaset triplane. Not validated; next decision is the user's  
**Publication planning target:** SIGGRAPH 2027 (exact Technical Papers deadline must be separately verified; do not represent the estimated January deadline as official)

## 0. Document Purpose and Authority

This living document declares the active decision frontier and stop conditions. It should be read with:
- docs/RESEARCH_CENTRIC_TOPIC.md — stable research problem and current method hypothesis;
- docs/Architecture.md — concrete candidate/feasibility contract;
- docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md — control groups and claim scope;
- docs/history/ARCHITECTURE_LIFECYCLE_SEPARATION_2026-10-07.md — preserved previous working architecture;
- numbered docs/worklogs — append-only historical evidence.

This document supersedes the old roadmap's instruction to continue K=32 dynamic-state representation batches. It does **not** declare the earlier research intrinsically incorrect.

## 1. Central Intent and Architecture Decision

**Stable research purpose:** maintain high-quality pretrained neural light transport when unforeseen changes to geometry invalidate some previously compressed GI.

**Current conditional candidate:** learned transport dependency + selective neural-state invalidation/reconstruction + preservation of unaffected pretrained transport.

**Novelty boundary:** object hierarchy and selective GI recomputation are prior art. The surviving research problem, as suggested by the literature kill-search, is linking geometry-change-induced radiometric validity to *learned* state units and refitting those units without losing global transport quality.

**The immediate question is upstream of dependency hierarchy:**

> Even with offline oracle guidance on invalid state, does selective neural-state refit actually preserve quality at meaningful quality-matched cost savings?

If the answer is no for the selected substrate/regime, stop this line of implementation and interpret the failure before designing a hierarchy.

The earlier persistent appearance + compact dynamic GI state + reusable operator is now **ON HOLD**. WL28 and WL29 falsified only their specific K=32 realization for held-out near-mirror teaset T3.

## 2. Accepted Evidence / What Has and Has Not Been Solved

### 2.1 Problem existence and attribution — CLOSED in canonical regime

- WL21: 8DNA teaset, stationary receiver, moved rigid internal part, strong reference nonlocal change with frozen near-zero response; relation-preserving T1 control.
- WL22: same-scene RNA replication under common lighting; cross-backbone **problem evidence**, not cross-backbone method validation.
- WL24: primary physical change attributed to indirect-path occlusion and moved-part radiance, rather than direct receiver illumination.
- WL25: local identities, frames, view/light and direct visibility do not provide current nonlocal learned transport input.
- Rain nonrigid work: useful diagnostics; fixed-target reference change too small for clean causal claim.

### 2.2 Cost motivation — BASELINE CALIBRATED, quality-matched method speedup NOT ESTABLISHED

- WL26 physical Cycles on BMW27: 1080p at 16 / 64 / 256 spp ~0.22 / 0.76 / 3.0 s; persistent-geometry update itself much cheaper; GI sampling dominates.
- WL27: historical neural **scratch** T3 pipelines: 8DNA envmap first criterion-crossing checkpoint at 38.9 min only identified in hindsight (do not call operational recovery), 1.76 h full schedule; RNA common-light ~3.24 h with no checkpoint meeting predeclared criterion. RNA supervision generation dominated cost.
- Historical methods, lighting and scenes are not quality matched to the future selective-refit model; never use their times as selective-method speedup denominators.

### 2.3 Previous dynamic-state prototypes — CLOSED NEGATIVE, NOT UNIVERSAL FAILURE

- WL28 CASE C: frozen RNA + K=32 current geometric relation sidecar fails T3; richer reference-derived oracle succeeds through shared residual operator.
- WL29 CASE D: same K=32 plus remote outgoing-radiance proxy ≈ ZERO/SHUFFLED, and exact directional radiance on the SAME K=32 support also fails; runtime proxy update ~3.2 s. Angular resolution, aggregation and training-state coverage remain entangled.
- Do not continue another K sweep, angular probe tuning, MLP expansion, or local residual patch merely because this new branch exists.

### 2.3b Oracle selective refit (F0+F1) — CLOSED NEGATIVE ON ONE SUBSTRATE (WL30)

- Substrate: released 8DNA teaset asset (localized state = triplane, 12 288 cells / 98 304 of 669 321 parameters); Neural Radiosity audited at code level and deferred (unvalidated runtime and T0 quality). Envmap regime; G0, signal and scratch-ceiling gates pass.
- Oracle: exact common-random-number comparison of T0/T3 training path samples; 8.9% of valid samples affected, 24% of the affected weight ≥ 0.15 from the mover; M95 = 1 195 cells (9.7% of all, 39% of occupied cells), supporting 61% of all T3 samples; seed-stable (Jaccard 0.994).
- Matched arms, 32 768 steps, 2 seeds: global warm start C recovers (R_aff 0.085 frozen → 0.045; tracking gain 0.6); all-localized-state D with shared networks frozen recovers only 0.20× of C's stale-error reduction (gain 0.22); oracle-selective E ≈ D (0.080 vs 0.077) but fails preservation (+7.5% unaffected error) and cost (E reaches the D-matched target 2× slower; restricting work to touched samples removes 5%).
- Attribution: the shared flows/cubemaps own the configuration-dependent transport (post-hoc swap: C's shared tensors alone carry its interaction-ROI recovery), and triplane projections couple every cell to distant surfaces. A substrate result; not evidence against selective refit for 3D-local, non-owning state.

### 2.4 Literature kill-search — CLOSED ENOUGH FOR THIS DECISION, NOT PROOF OF WORLDWIDE ABSENCE

- Classic incremental GI/BVH/Enlighten establish generic dependency, hierarchy and partial recomputation.
- NeLT and Superposed Deformable Feature Fields support current dynamic object relations/high-frequency GI using pretrained function inference; they are not evidence for selective refit of stale learned state.
- Closest opportunities involve learned-state provenance, radiometric invalidation and selective optimization for high-frequency GI.
- Any later related work requires continued verification, particularly new 2025–2027 systems.

## 3. Research Phase Plan (Gates, Not an Automatic Pipeline)

| Gate | Status | Scientific question | Do not proceed if |
|---|---|---|---|
| E0 Problem and attribution | CLOSED, WL21–25 | Is learned GI stale when relations change? | Historical evidence preserved |
| E1 Recompute baseline context | CLOSED, WL26–27 | Are existing rebuilds expensive? | Not a same-substrate speedup |
| E2 Compact state pilot | CLOSED NEGATIVE, WL28–29 | Did tested K=32 current state recover T3? | Do not patch/sweep |
| F0 Substrate suitability | DONE for one substrate, WL30 (8DNA teaset selected; Neural Radiosity deferred) | Does an existing per-scene neural GI model combine credible high-quality GI and localizable learned units? | No viable substrate without major redesign |
| F1 Oracle selective-refit feasibility | **CLOSED NEGATIVE on 8DNA triplane (WL30)**; other substrates untested | With oracle affected-unit guidance, can partial refit match global quality while preserving unaffected state with lower cost? | No local recovery, preservation, or cost advantage |
| F2 Actual dependency/invalidation | NOT AUTHORIZED (F1 not supported) | Can geometry edits identify stale learned units efficiently and radiometrically? | F1 not supported |
| F3 Integrated reconstruction | FUTURE / GATED | Does actual selective refit beat appropriate baselines end to end? | F2 not supported |
| F4 Scene breadth / SIGGRAPH closure | FUTURE / GATED | Does method generalize, stay efficient, look correct and tell a strong paper story? | Canonical-only or nonphysical improvement |

A future agent must **not** proceed automatically from F1 to F2 even after success. The user reviews evidence and approves a new architecture decision.

## 4. Batch F0 + F1, One Bounded Architecture Experiment — COMPLETED (WL30, negative on 8DNA teaset triplane)

### Direction

Determine the existence of a *selectively optimizable transport-state subset*, not the design of a hierarchy.

### Purpose

Reject the hierarchical approach early if perfect affected-region guidance cannot make pretrained learned GI partially reconstructible.

### Central Intent

> Hold the pretrained transport model, training/evaluation supervision and current T3 scene fixed. Change **only the allowed learned-state update set**.

### Work to perform

1. **Audit at most two available candidate substrates**, select one existing reproducible high-quality per-scene neural GI model whose trainable state units can be enumerated and masked. Do not start a new renderer. Check that same-scene physical GI and T0 static quality make the test meaningful.
2. Define a controlled T0→T3 geometry edit with stable receiver correspondence. Prefer the teaset, but permit a scientifically equivalent scene if compatibility is demonstrably poor.
3. Create an **offline reference-derived oracle affected-query signal**, with separate reference/evaluation samples and defensible mapping to real trainable-state support. Report over-/under-invalidation.
4. Train and compare **five same-substrate controls**:
   - Frozen pretrained T0;
   - scratch full T3 retrain (quality ceiling);
   - global T0→T3 warm-start retrain;
   - global update of all eligible localized state while freezing the same shared weights as selective arm;
   - oracle-selected local-state-only update, keeping unaffected units and shared weights bit-identical.
5. Report affected/unaffected/seam quality, T0→T3 signed transport-change tracking, selected state count/fraction, optimization/total latency, target generation, frozen parameters, cost-vs-quality and qualitative crops.
6. Stop after the viability verdict and append a numbered Worklog, machine-readable metrics and human-review artifacts. Update living documents only for a substantive gate decision.

### Preserve

- Worklogs 21–29, original RNA/8DNA baselines and historical assets untouched;
- original optimizer/model architecture and training semantics in the selected substrate;
- same lighting, material, reference, data splits, budgets and scene across arms;
- shared/global weights frozen in matched all-local and oracle-selective refit;
- held-out evaluation distinct from oracle-mask/training selection.

### Change ONLY

The subset of otherwise eligible trainable learned-state units permitted to update.

### Do NOT

- build a BVH/scene/object/transport hierarchy;
- implement a production invalidation predictor;
- introduce a new MLP, network width or learned state representation;
- swap substrate mid-run to chase a positive result;
- change material roughness to improve score;
- tune affected thresholds or evaluation rules on final T3 results;
- use reference oracle radiance as a runtime method;
- run multiple new architecture hypotheses in sequence;
- broaden immediately to deforming characters, cloth or topology changes.

### Preliminary screening criteria (set before the final run)

- affected-region accuracy approaches same-substrate globally updated local-state control (a 10% relative error margin is a *screening proposal*, not physics);
- no meaningful unaffected-region forgetting or seam;
- a nontrivial state fraction is genuinely preserved;
- a meaningful quality-matched cost reduction (2x screening target), versus global-local-state update, with global warm-start and scratch reported honestly;
- no disqualifying oracle mask leakage.

These numbers must not be relaxed after seeing a result. No claim of practical interactive rendering follows from oracle-assisted speedup.

**F1 completion question:**

> Can oracle-guided selective refit reconstruct held-out current-geometry GI at near-global-refit quality, preserve unaffected learned transport and materially reduce measured quality-matched update cost?

If negative, attribute whether static fidelity, chosen substrate, parameter coupling, mask support, multi-bounce closure, or optimization/runtime dominates. No automatic next batch.

## 5. F2 — Learn Transport Dependencies (CONDITIONAL FUTURE)

If and only if F1 passes:

- Construct or estimate radiometric geometry-edit-to-learned-unit dependencies **without oracle GT at runtime**.
- Evaluate directed multi-bounce effects and faraway receivers; distinguish false-negative invalidation from false-positive over-invalidation.
- Compare actual dependency with naive spatial distance/containment, classical geometric/path-link tracking, reference oracle upper bound and full invalidation.
- Measure dependency build memory, runtime query, update latency and changed fraction.
- Choose whether a hierarchy is even necessary, or whether a sparse graph/index is superior; **hierarchy is an implementation choice, not a required output**.

Completion: does learned-state radiometric dependency support selective refit reliably enough to outperform simpler invalidation?

## 6. F3/F4 — Integration, Breadth and Paper-Level Proof (FUTURE)

Once gates pass, evaluate full-method quality, timing and robustness on additional independent scenes and material/transport regimes. The canonical teaset is not a complete paper. Require at least:
- glossy/high-frequency and multi-bounce receiver effects;
- several geometry edit magnitudes, positions and temporal events;
- affected and unaffected region metrics, flicker/seam/forgetting;
- scale-up: mover count, dependency density, state size and full invalidation fraction;
- real reference and comparative baseline quality/cost.

Claims of cross-backbone portability require actual second-method implementation, not two backbones failing on the same scene.

## 7. Paper Planning — SIGGRAPH 2027

**Planning only.** Verify official SIGGRAPH 2027 Technical Papers deadlines before treating a date as binding. Working assumption: a January 2027 full-paper submission window; do not assert that this is confirmed.

Internal ambition:
- **8–23 October 2026:** F0/F1 feasibility, with early stop if representation/locality fails.
- **Late October–mid-November:** F2 actual dependency and reconstruction experiments only if F1 succeeds.
- **Mid-November–early December:** F3 integrated experiments and F4 breadth.
- **December:** figures, videos, method paper draft, baseline comparisons and ablations.
- **January 2027 (provisional):** final paper/supplement review and submission.

This schedule is aggressive. Feasibility success does not guarantee timely final method. Missing the gates requires a publication-target reassessment rather than weakening scientific standards.

## 8. Roles, Artifacts and Testing

Use docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md for control hierarchy.

Every substantive batch records:
- input/scene/renderer/model/checkpoint commit + hashes;
- synthetic correctness fixture (if applicable);
- **real-scene** matched evaluation;
- quantitative state/coverage/error/time accounting;
- qualitative render exports;
- interpreted outcome and unresolved alternatives;
- numbered append-only Worklog and results/evaluation/<N>/ artifacts.

Focused tests after meaningful implementation units. Full regression only for changes to shared production/evaluation contracts or major shared contract closure.

Do not silently alter central documents to describe an unvalidated branch as completed. Explicitly distinguish **method concept / diagnostic oracle / deployed mechanism / observed result / decision**.

## 9. Immediate Action

**F0+F1 ran once (worklog 30) and is NEGATIVE on the 8DNA teaset triplane substrate.** The batch stopped at the predeclared gate. F2 is not authorized.

The next step is a **user decision**, not an automatic batch. Defensible options:

1. a separate F0 audit + F1 run on a substrate whose localized state has 3D-local support and whose shared decoder does not own configuration transport (e.g. Neural Radiosity dense grid), with the same protocol and controls, after its runtime, T0 quality and T3 ceiling are validated;
2. reassess the selective-refit direction against the on-hold lifecycle-separation direction, given that on the only tested substrate global warm start recovers and localized state does not.

Do not reuse the 8DNA result as a general falsification, and do not rescue it with larger masks, cubemap-as-local-state redefinitions or longer budgets without a new protocol.

## 10. Lasting One-Line Rule

> Before learning where to invalidate transport, demonstrate that any nontrivial subset of high-quality pretrained neural transport can actually be reconstructed independently and efficiently when its invalidity is already known.
