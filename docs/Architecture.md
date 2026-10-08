# Architecture — Learned Transport Dependency & Selective Neural Reconstruction

**Last updated:** 2026-10-08  
**Status:** CONDITIONAL architecture candidate; oracle selective-refit gate F1 tested once (worklog 30): **NEGATIVE on the 8DNA teaset triplane substrate**; untested on 3D-local substrates  
**Active branch:** Selective_Recompute  
**Historical direction:** docs/history/ARCHITECTURE_LIFECYCLE_SEPARATION_2026-10-07.md

---

## 1. Architecture Problem

A fixed-configuration neural GI representation may store both valid reusable transport and transport that becomes stale after a geometry edit. The candidate research problem is to reconstruct invalid learned state without discarding unaffected high-quality learned transport.

This document describes a **testable hypothesis**, not an implemented method or proven novelty.

The mechanism, if ultimately viable, must deliver all three:
1. Determine which learned-state units are radiometrically invalidated by a geometry change.
2. Selectively reconstruct those units, with parameter and output preservation outside the affected support.
3. Match current-configuration transport quality sufficiently well at substantially lower measured update cost than global alternatives.

Object hierarchy is only a possible search/index structure. Hierarchical selective GI recomputation, dynamic PRT and object-wise neural transfer already exist as prior art. A successful paper must prove **learned-state dependency and partial reconstruction**, not mere hierarchy construction.

## 2. Core Candidate Contract

~~~text
Geometry G0 + quality-supervised pretraining
          |
          v
High-quality pretrained learned-transport state S0
          |
  geometry edit G0 -> G1
          |
          v
Radiometric geometry-to-learned-state dependencies [FUTURE]
          |
          v
Stale learned-state set I(G0,G1) [ORACLE ONLY IN FIRST GATE]
          |
          +-- affected learned units: selectively reconstruct S1[I]
          |
          +-- unaffected learned units: S1[~I] = S0[~I]
          |
          v
Current-geometry neural GI using S1, unchanged reusable parameters
~~~

Let F(S, phi; G, x, omega) denote the chosen learned transport evaluator with localizable learned state S, shared/global parameters phi, geometry G and directional query (x,omega). The feasibility question is whether some strict proper subset I of learned-state units can be optimized from S0 on geometry G1 with phi and S[~I] frozen, yielding a current reconstruction close to suitable global-refit quality and preserving unaffected outputs.

**Parameter identity is not sufficient for output identity.** A shared decoder, interpolation footprint, hash collision or GI coupling can change predictions outside the nominal updated region. Measure both parameter invariance and radiometric output preservation.

No exact unit granularity, hierarchy, predicate, neural optimizer, radiometric threshold or number of update iterations is currently frozen.

## 3. Scientific Units / Ownership (Do Not Conflate)

| Entity | Meaning | Contract |
|---|---|---|
| Geometry edit | Mover/part transformation or deformation | A causal physical change, not a learned-state key |
| Receiver query | Surface/direction whose outgoing light is evaluated | Must maintain correspondence across G0/G1 for signed change |
| Learned state unit | Identifiable parameter group (cell/vertex/patch/latent block) | Eligible for measured selective update |
| State support | Evaluations influenced by that unit | Can extend beyond cell boundaries; measure collisions/interpolation |
| Transport dependency | Physical/radiometric reason a geometry edit invalidates a learned prediction | Not equivalent to spatial overlap or object ownership |
| Oracle affected mask | Reference-derived offline guide to likely invalid state | Upper bound control, not valid deployable detector |
| Refit set I | Actually trainable units in candidate update | Must be accounted for in units/bytes/fraction |
| Frozen state ~I | Unit parameters not optimized | Bit-identical parameters and measured output stability |
| Shared/global phi | Global decoder, global weights, normalizations, reusable rules | Frozen in selective experiment and capacity-matched global-localized control |

An oracle image/receiver change mask is **not automatically** a valid learned-parameter mask. A mapping from receiver support to trainable units and its false-negative/over-invalidation risk must be justified.

## 4. Critical Ordering: Prove Reconstruction Before Hierarchy

### Gate F0 — Substrate and physical-signal audit

Select **one existing, reproducible, sufficiently high-quality per-scene neural GI substrate** with identifiable localizable trainable state. Audit at most two candidates in one bounded batch.

Eligibility:
- T0 baseline quality sufficient to separate static model error from GT T0→T3 change;
- genuine indirect/glossy or multi-bounce changed transport;
- shared weights can be frozen;
- independently trainable state units are identifiable;
- reference sampling, optimizer masking and partial checkpointing are feasible;
- realistic training and evaluation cost on available hardware.

RNA/8DNA remain historical **problem-evidence** controls for intra-asset relational staleness. Neither is automatically suitable for the new **selective learned-state refit** substrate: normal rigid inter-asset host transport and neural state ownership differ.

If no preexisting substrate satisfies these conditions without building a new renderer, stop and report a substrate blocker.

### Gate F1 — Oracle-guided partial neural refit (CURRENT GATE)

Use offline reference transport on G0/G1 to construct a conservative affected-query signal and map it to learned-state units, with separate held-out evaluation. Do not represent this as a runnable online dependency detector.

Keep one fixed substrate, T0/T3 state, losses, training data quality, lighting, camera, materials and output evaluation across all arms. Test:

| Arm | Start state | Trainable scope | Question |
|---|---|---|---|
| A Frozen T0 | pretrained T0 | none | How stale is current transport? |
| B Scratch T3 | same architecture, scratch | all standard trainable state | What is the reconstruction capacity/quality ceiling? |
| C Global warm-start | same pretrained T0 | all standard trainable state | Is selective work better than ordinary fine-tuning? |
| D Global local-state refit | same pretrained T0 | all eligible local units; global phi frozen | Control for the effect of frozen shared parameters |
| E Oracle selective refit | identical pretrained T0 | only oracle-selected local units; global phi and other units frozen | Does selective learned-state reconstruction work? |

An equal-size spatial-neighborhood update may be included as a diagnostic, not substituted for these controls.

This batch changes ONLY **which eligible learned-state units may update**, while construction of the oracle mask is kept an explicitly separate diagnostic. It does not propose or implement a hierarchy.

The goal is a *causal* measurement of localizable neural refit, not a performance-optimized selective system.

### F1 result — worklog 30 (8DNA teaset, envmap regime)

| arm (32 768 steps, 2 seeds) | trainable | affected-region error R_aff | stationary tracking gain | unaffected R_unaff |
|---|---|---|---|---|
| A frozen (attached) | 0 | 0.085 | −0.03 | 0.025 |
| B scratch (matched / WL27 ceiling) | all | 0.050 / 0.040 | 0.67 / — | 0.034 / 0.026 |
| C global warm start | all 669 321 | 0.045 | 0.56–0.63 | 0.026 |
| D all triplane cells, shared frozen | 98 304 | 0.077 | 0.22 | 0.028 |
| E oracle M95 cells, shared frozen | 9 560 | 0.080 | 0.17 | 0.027 (+7.5% vs step 0) |

Measured contract consequences for this substrate:

- **Ownership:** the configuration-dependent transport is owned mainly by the shared decoder (flows, direction cubemaps). D recovers 0.20× of C's stale-error reduction; a post-hoc swap shows C's shared tensors alone carry its receiver-tracking recovery.
- **Support:** triplane axis projections make a 95% oracle cover (1 195 cells, 39% of occupied cells) touch 95% of training samples and 97% of unaffected pixels; restricting work to touched samples saves ≈5%, and E is 2× slower than D to the D-matched quality.
- **Oracle:** exact and stable (CRN, Jaccard 0.994 across seeds); 24% of the affected weight lies ≥ 0.15 from the mover — invalidation is genuinely nonlocal.

The result constrains the candidate contract: selective refit requires a substrate whose **localized state, not its shared decoder, owns the configuration-dependent transport, and whose state units have 3D-local support**. Whether such a substrate exists among existing per-scene neural GI models (e.g. dense-grid Neural Radiosity) is unaudited.

### Gate F2 — Learned transport dependency (NOT AUTHORIZED: F1 negative on the only tested substrate)

Infer or construct a deployable geometry-edit-to-invalid-learned-unit dependency. Compare to:
- oracle invalid set (upper bound);
- spatial proximity / containment baseline;
- explicit geometry/path-link or support-based traditional dependency;
- conservative full invalidation.

Measure false negatives on distant receiver effects, false positives / over-invalidation, dependency storage, query/update cost and multi-bounce propagation.

Only after deciding dependency semantics should a BVH/object/transport hierarchy be considered as an implementation index.

### Gate F3 — Integrated method and scene breadth (FUTURE)

Compare real incremental system with matching global fine-tune/full rebuild and physical/current-GI alternatives. Add independent material/scene families, different geometry changes, temporal consistency and scaling. Evaluate high-frequency glossy interactions, more movers and realistic frequency of edits.

Do not jump from synthetic test or oracle gate to framework or SIGGRAPH success claims.

## 5. Feasibility Measurements and Evidence

**Physical relevance**
- T0/T3 path-traced or otherwise trusted transport reference and noise floor;
- stable receiver correspondence;
- actual signed indirect transport-change maps, not just absolute image quality;
- affected / unaffected / seam subsets defined before final testing.

**Quality & preservation**
- affected-region error relative to GT and full/global refit;
- untouched-region quality change from T0 to post-refit;
- boundary artifacts, forgetting, signed reflection/occlusion tracking;
- output errors despite frozen weights;
- reference-driven oracle false negatives, if measurable.

**Update and cost**
- total eligible unit count, selected fraction, modified units and bytes;
- optimizer-only time and quality-vs-wall-clock;
- supervision generation, data transfer, state selection, optimizer setup, validation, checkpoint, peak memory and inference;
- end-to-end quality-matched comparison vs C (global warm-start) and D (all-local-unit refit) on SAME representation and scene.

Historical cost context (not quality-matched denominator):
- Worklog 26 physical Cycles: ~0.22 / 0.76 / 3.0 s at 1080p 16 / 64 / 256 spp on BMW27;
- Worklog 27 historical neural scratch: 8DNA envmap earliest oracle recovery 38.9 min (not operational recovery), full schedule 1.76 h; RNA common light 3.24 h and no rule recovery;
- Worklogs 28/29: the two fixed K=32 relational-state candidates failed held-out near-mirror T3; Worklog 29's runtime radiometric proxy cost ~3.2 s.

**Predeclared feasibility screening**, to be set before the final evidence run:
- near-global-localized-refit affected quality (a ~10% relative error margin is a provisional screening guide, not a universal success theorem);
- no meaningful unaffected-region degradation;
- nontrivial preserved learned-state fraction;
- a meaningful quality-matched latency win (2x is a provisional engineering screening goal), with total end-to-end cost reported honestly.

Do not retrospectively alter these criteria.

## 6. Known Risks / Kill Conditions

1. **No localizable pretrained high-quality GI substrate.** Stop at F0; do not invent a new architecture in the gate batch.
2. **Local state is not radiometrically local.** Updated units affect distant output; simple parameter freezing cannot ensure unaffected preservation.
3. **Multi-bounce invalidation closure expands globally.** The changed fraction approaches one; hierarchy cannot rescue the underlying locality assumption.
4. **Affected transport requires shared-weight optimization.** E fails even when D with frozen phi is viable or global C improves; the chosen substrate is incompatible.
5. **Oracle misses necessary units.** Distinguish mask mapping / coverage failure from optimization failure.
6. **Quality mismatch.** Partial refit never approaches eligible global refit at the same reference regime.
7. **Cost mismatch.** Reduced parameter count does not yield a meaningful quality-matched update latency improvement.
8. **Poor static fidelity / weak GT change.** No scientific attribution is possible; stop before method evaluation.
9. **Leakage or tuning.** Test GT used in optimizer/checkpoint/threshold selection invalidates inference.

If F1 fails, do not optimize a threshold sweep, enlarge model, add heuristic dependency links, create a new transport representation, or proceed to F2 in the same batch.

Failures are substrate- and test-regime-scoped, not universal architecture falsifications.

**Worklog 30 (8DNA teaset triplane):** kill conditions 2 (local state not radiometrically/computationally local), 3 (the 95% invalidation cover touches 95% of training samples, through triplane projections), 4 (affected transport requires shared-weight optimization: D ≪ C), 6 (partial refit never approaches global quality) and 7 (no cost advantage) materialized; 1, 5, 8 and 9 did not (substrate quality adequate, oracle exact and stable, signal strong, no test-GT use in training or selection). Scoped to this substrate.

## 7. Novelty / Prior-Art Guardrails

Prior-art foundations (not ours):
- selective GI recomputation and dependency hierarchy: Drettakis–Sillion 1997, Bala 1999, Luksch 2019;
- object-oriented neural transport and dynamic high-frequency inference: NeLT 2023 and Superposed Deformable Feature Fields 2024;
- incremental build/dependency tracking for production precomputed GI: Enlighten;
- global adaptation, neural caching, precomputed transfer and static+dynamic residual paths.

Potential technical difference to validate: **learned transport state with geometry-dependent validity, local parameter/feature support, causal invalidation and selective retraining under glossy and multi-bounce GI**.

Do not assume learned transport provenance can be recovered from object bounds alone. Do not claim multiple-backbone generality unless actually demonstrated.

## 8. Historical Branch and Research Invariants

- Archived previous lifecycle-separation specification: docs/history/ARCHITECTURE_LIFECYCLE_SEPARATION_2026-10-07.md
- Worklogs 21–29 and existing frozen RNA/8DNA baselines: unchanged and reproducible.
- The stable fundamental question (valid vs invalid learned transport across geometry changes) is retained.
- The implementation direction is **conditionally pivoted** to selective refit; the original compact current-state path is on hold, not disproven.
- First gate is F1 only. No actual hierarchy, learned invalidation predictor, large-scale scene, attention model, NURBS, nonrigid deformation or cross-backbone port in this batch.

## COMPLETION CONDITION — CURRENT GATE

> **With oracle guidance on invalid learned state, can the chosen high-quality pretrained neural transport representation recover a held-out geometry edit through partial refit while preserving unaffected transport at meaningfully lower quality-matched cost than global refit?**

A positive answer authorizes *research on actual dependency detection*, not a completed method.

**Answered for one substrate (worklog 30): no** — the 8DNA teaset triplane cannot be selectively refit to near-global quality, does not preserve unaffected output under the predeclared criterion, and gives no cost advantage. The question remains open for substrates with 3D-local, decoder-independent transport state.
