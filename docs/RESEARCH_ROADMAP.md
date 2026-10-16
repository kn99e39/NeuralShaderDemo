# Research Roadmap — Surface-Owned Selective Neural Transport Recomputation

**Updated:** 2026-10-17 (worklog 35)  
**Branch:** Selective_Recompute  
**Status:** Architecture Batch 1 (RNA-based independent units) reported — ownership contract exact; stopped on the baseline-credibility precondition; Batch 2 NOT authorised  
**Venue planning:** SIGGRAPH 2027, official paper deadline to be separately verified.

## 0. Decision Record

- **2026-10-12:** the user clarified that "selective" means recomputing complete owned models, never intra-model parameter subsets. The archived object-level-only framing is at docs/history/ARCHITECTURE_OBJECT_LEVEL_SELECTION_2026-10-12.md.
- **2026-10-13:** the user adopted ten decisions, listed in full in docs/RESEARCH_CENTRIC_TOPIC.md §2:
  - RNA is the initial baseline;
  - ownership may be finer than objects;
  - Surface-Owned Transport Units are the active hypothesis;
  - each invalidated unit's full state may be recomputed;
  - object-level recompute is the coarse comparison;
  - at least five architecture batches come before a broad viability judgment;
  - a single negative result does not falsify the hypothesis;
  - 2007–2013 dynamic GI is the priority design reference;
  - the 1997 line-space hierarchy is not the starting point;
  - efficiency and scaling come after feasibility.

Active contracts: docs/RESEARCH_CENTRIC_TOPIC.md, docs/Architecture.md, docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md.

## 1. Central Intent

> Can learned neural transport be owned by independently reconstructible surface units such that, after a geometry edit, recomputing the complete state of only the transport-invalid units restores high-quality full-scene GI while all valid units are reused unchanged?

## 2. Evidence So Far

| Worklogs | Established | Not established |
|---|---|---|
| 21–25 | stale nonlocal GI at stationary receivers under a part edit; indirect occlusion and moved-part reflection dominate | that every dynamic neural GI method fails |
| 26–27 | physical and historical whole-asset neural recompute are costly (different regimes) | a quality-matched unit-level cost |
| 28–29 | two sparse K=32 relational-state candidates fail held-out near-mirror T3 | that owned-unit recompute fails |
| 30 | triplane-cell masking inside one 8DNA model (shared flows frozen) fails | anything about complete owned units |
| 31 | explicit per-part surface units (no learned decoder) reach 8DNA-level T0 and recover T3 when refitted; object-level D_T = all objects on the compact teaset; no wall-clock saving in that implementation | unit-level D_T sparsity; neural units |
| 35 | RNA qualified on its published asset (Lego). Native generation and training with unchanged upstream code. Retraining matches the released checkpoints within 0.2 dB on the same val set. The gap to Table 1 (−1.9 HQ / −0.8 small) is explained by 1024- vs 4096-spp validation targets (+1.7 / +0.6 dB). Overall COMPARABLE | performance claims (production renderer unreleased) |
| 34 | Batch 2: seams = independent per-unit error fields (decorrelated across cuts) exposed by hard selection; not supervision starvation. P16-PU composition removes 86% of the seam excess on held-out C2 with lower boundary error, exact output-influence accounting (S_i ∪ B_i), and REINIT intact | recovery of P16's interior fidelity loss (−0.20 dB, logL1 1.18×); C1 accuracy gain |
| 33 | RNA baseline audit: integration and evaluation faithful to upstream (no defect); C1 failure = RNA representation limit on near-mirror Ni (by exclusion: not noise, estimator, distribution, capacity, resolution, convergence or data density 100-200 views); C2 gate miss dominated by 128-spp target noise (A vs converged ref 26.7 dB on 4 views) | a credible static reference (needs a prospective protocol); a positive test of the C1 representation limit |
| 32 | RNA split into complete independent units: exact parameter/output preservation and full-state unit reconstruction; P16 costs 1.15× HDR log error and visible seam steps vs RNA (C2), not fixed by 16× capacity | static fidelity under a credible baseline (gate failed: 17.97 / 24.92 dB); recovery after a physical edit; cost |

## 3. Architecture Batch Program (≥ 5 batches before a broad viability judgment)

| Batch | Status | Question | Outcome / stop |
|---|---|---|---|
| 1 — RNA independent units (worklog 32) | **REPORTED; stopped on the baseline-credibility condition** | Can RNA be split into independently owned units with full-state reconstruction and preservation, without unacceptable static-GI loss? | Ownership exact; fidelity verdict not adjudicated; costs: HDR log +12–15%, seam steps |
| 2 — boundary fidelity (worklog 34) | **REPORTED** | Why do independent units lose fidelity at boundaries, and can one representation change fix it without breaking reconstruction? | H2 + H3 supported, H1 starvation contradicted; P16-PU fixes seams, not interior loss; Batch 3 needs a user decision (recommended: immutable shared decoder + owned features under PU) |
| 2 (original plan) | superseded by the above | Recommended, after resolving credibility: radiometric validity of unit-level full recompute under a physical T0 → T3 edit (oracle dirty units vs all units vs whole objects). Alternative: a representation variant (frozen shared decoder + owned features; or blended overlapping units) | — |
| 3–5 | FUTURE | Chosen from evidence: representation variants, D_T sparsity on scenes where it is physically plausible, non-RNA substrates | — |
| After feasibility | FUTURE | Dependency/invalidation discovery (deployable), hierarchy as an optional accelerator, efficiency and scaling (decision 10) | requires feasibility evidence and a user decision |

Gates retained from the object-level framing:
- **P:** phenomenon, CLOSED.
- **C:** cost motivation, CALIBRATED.
- **H:** compact state, CLOSED NEGATIVE for the tested cases.
- **X:** intra-object masking, CLOSED NEGATIVE and off-target.
- **O0/O1:** object-level feasibility and oracle refit. Worklog 31 answered them on an explicit substrate (O0 pass; O1 dense at object granularity). They remain a coarse comparison for unit-level results.

## 4. Precondition Before Batch 2: Static Credibility

Worklog 32's original RNA failed the predeclared 25 dB static-credibility bar:
- C1 (canonical near-mirror teaset): 17.97 dB;
- C2 (rough nickel α 0.3): 24.92 dB.

The batch therefore stopped on its first stop condition. Worklog 33 attributed the failure:
- **C1:** RNA's representation limit on near-mirror conductors. No integration defect; more data, capacity or resolution did not help.
- **C2:** target noise in the measurement. 37.5% of A's error is 128-spp GT noise; against converged targets A scored 26.7 dB on 4 views.

The user chooses, before any Batch-2 training, between:
- **(a)** a predeclared scene/material configuration where original RNA clears the bar; or
- **(b)** a prospectively revised credibility criterion, written as a new protocol and never applied retroactively to Batch 1. Worklog 33's recommendation: the C2 material regime scored against converged (≥ 8× spp) validation targets, pooled, final weights.

Do not select scenes, gates or thresholds after seeing candidate results.

## 5. Rules for Every Batch

- **Controls.** Compare A (original substrate) against B (owned candidate), changing only the ownership boundary and the changes strictly necessary for it. Account for capacity openly; label a comparison capacity-confounded instead of forcing equivalence.
- **Static GI evaluation:**
  - full-image metrics plus per-unit, boundary/interior and seam metrics;
  - an HDR-appropriate metric beyond PSNR;
  - high-frequency/glossy regions;
  - parameters, convergence, load/inference time and memory.
- **Unit reconstruction.** Report parameter preservation, output preservation and radiometric validity separately. A synthetic intervention proves semantics, not GI recovery.
- **Exports.** Reviewer exports go in `results/evaluation/<worklog>/` with a manifest. Tests must cover mapping, ownership, isolation, checkpoint, dispatch, boundary evaluation and composition.
- **Interpretation.** Do not treat one negative candidate as falsifying the hypothesis. Do not claim speedups from parameter counts. Do not advance automatically.

## 6. Preserve

- Worklogs 21–32 unchanged; worklog 30 as the historical intra-object masking result.
- Original RNA/8DNA outputs, checkpoints and scripts as historical baselines.
- Archived architectures under docs/history/.

## 7. SIGGRAPH 2027 Planning

Replan durations from measured training and reference wall-clock. For reference, one P16 RNA arm takes about 1.5 h on the RTX 5080 and about 2.8 h on LabServer63. Human coding burden is not a veto for a scientifically better architecture. Publication aims never override predeclared stop conditions.

## 8. Document Synchronisation

Update this file, docs/Architecture.md and docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md when a batch closes, a hypothesis is adopted or rejected, a baseline role changes, or implementation is authorised.

This document does not authorise Batch 2.
