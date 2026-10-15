# Baseline Roles and Evidence Strategy — Surface-Owned Selective Neural Transport Recomputation

**Updated:** 2026-10-15 (worklog 33)  
**Status:** Active interpretation/control contract. Owned-unit recomputation is NOT validated; Batch 1 established the ownership contract on RNA but stopped on baseline credibility  
**Project:** NeuralShaderDemo / Selective_Recompute

## 1. Scope

- "Selective recomputation" means recomputing the **complete owned learned state of each transport-invalid Surface Transport Unit**. A unit may be finer than an object; a whole object is the coarsest unit and serves as the **coarse comparison**.
- Never select a subset of cells, channels or layers inside a jointly owned model. That is worklog 30's separate, negative hypothesis.
- D_G ≠ D_T: stationary receivers can hold invalid transport.
- RNA is the **initial implementation baseline** (decision 1), not the required final model.
- At least five architecture batches come before a broad viability judgment.
- One negative candidate does not falsify the general ownership hypothesis.

## 2. Historical Baselines and Evidence Roles

| Lineage | Scientific role | What it does NOT show |
|---|---|---|
| RNA/Rain | correspondence, nonrigid failures with known local confounds | generalized failure attribution |
| 8DNA/Teaset, WL21–22 | stationary receiver's GI changes as another part moves; stable whole-rigid-motion control | that owned-unit recompute works |
| RNA/Teaset (WL22, WL32 C1) | independent-backbone repetition of frozen staleness; **static ceiling below the 25 dB credibility bar on the near-mirror teaset** | that RNA is credible on this asset |
| WL24–25 | physical attribution: indirect-path occlusion plus moved-part reflection | online detector for invalid owners |
| WL26–27 | historical full-asset neural and physical recompute cost context | same-method unit-level quality-matched speedup |
| WL28–29 | sparse K=32 dynamic-state candidates negative | owned-unit recompute negative |
| WL30 | **intra-object** triplane-cell-only refit negative (8DNA, flows frozen) | complete-unit or whole-object refit negative |
| WL31 | explicit per-part surface units: O0 pass, T3 recovered by refit, object-level D_T = all objects | neural units; unit-level D_T sparsity; cost advantage |
| WL34 | P16 seams = decorrelated independent unit errors exposed by hard selection; P16-PU composition (no learned-state change) removes most of the excess with exact output-influence accounting | interior per-unit fidelity recovery; a shared-decoder ownership variant |
| WL33 | integrated RNA pipeline = faithful upstream RNA (eval identical, contract audited, estimator clamping not explanatory); teaset C1 = RNA representation-limited; C2 gate measurement noise-limited | that RNA cannot be credible on rough conductors; a positive proof of the C1 representation limit |
| WL32 | RNA independent complete units: exact parameter/output preservation and unit reconstruction; HDR/seam fidelity cost; **baseline-credibility gate failed** | static fidelity under a credible baseline; recovery after a physical edit |

Do not delete, overwrite, reinterpret, or silently merge these questions.

## 3. Baseline Roles in the Current Program

| Role | Current instance | Use |
|---|---|---|
| **Implementation / quality control (A)** | original RNA (`66b5b098`, upstream config, seed 0) on the same data | the static-fidelity reference that every owned candidate is compared with; must first pass a predeclared credibility bar |
| **Owned candidate (B)** | worklog 32: K complete RNA units (P1/P4/P16 matched, P16w full-size) | change only the ownership boundary |
| **Capacity control** | matched-texel units (P16) and full-size units (P16w); P1 isolates local framing | separate capacity from ownership effects |
| **Composition control** | the same learned units under hard selection vs P16-PU (worklog 34) | isolates composition effects from learned-function effects |
| **Ownership-semantics control** | REINIT: re-initialise one unit, retrain it fully with all others frozen | parameter and output preservation; reconstruction to prior quality |
| **Coarse comparison** | whole-object full recompute (worklog 31 on an explicit substrate) | the granularity that finer units must beat or match |
| **Physical reference** | Mitsuba full-scene path-traced radiance (worklog-22 producer), T0 and T3 | supervision and evaluation; never simplified to local transport |

Implementation baselines are not scientific replications. A credibility failure of A is a **baseline limitation**, reported as such, and never read as a candidate failure.

## 4. Static-Credibility Precondition

Before a candidate's fidelity is judged, A must clear a predeclared static bar on the chosen scene. Worklog 32 used 25 dB RNA val PSNR on final weights; A reached 17.97 dB (C1) and 24.92 dB (C2). The bar and the scene are chosen before training. A failure is reported, not repaired by changing the threshold, the metric, the checkpoint selection or the scene after the fact. Any revision is a new, prospective protocol.

Lessons from worklog 33 for every future credibility gate:
- **Score against converged targets.** At 128 spp the GT noise floor was 29.4 dB, which caps any model's measured PSNR.
- **State the aggregation.** Pooled vs view-mean differs by up to 0.6 dB.
- **State the checkpoint rule.** Best-by-val selects on the evaluation set.
- **Check the substrate's material regime first.** RNA on near-mirror conductors under a delta light is representation-limited (worklog 33), so a gate failure there is a baseline limitation, not a candidate failure.

## 5. Controlled Comparison Rules

- **Preserve:** scene geometry, materials, lighting, camera distribution, full-scene targets, splits, target semantics, evaluation regions, coordinate conventions, seeds, and hardware/runtime accounting.
- **Change only:** the ownership boundary and what it strictly requires.
- Capacity is accounted for (total, per unit, memory, effective support). If strict matching conflicts with ownership semantics, report the comparison as capacity-confounded.
- No tuning of model size or thresholds after seeing final results.

## 6. Unit Update and Validity

Track:
- D_G and D_T at unit granularity, and at object granularity as the coarse comparison;
- for each recomputed unit: complete-state recompute (scratch or warm start, measured separately);
- for each preserved unit: **parameter preservation**, **output preservation** and **radiometric validity**, as three separate properties.

Bitwise preservation never implies physical validity under a changed scene.

Oracle dirty sets are scientific upper bounds. Predeclare their noise floor, criterion, coverage and split; report their cost separately. Dense dirty closure is a valid negative result.

## 7. Metrics

- **Quality:**
  - full-image metrics plus HDR-appropriate error (log-space and relative error, not PSNR alone);
  - per-unit, boundary/interior and intra-part seam discontinuity;
  - high-frequency/glossy regions;
  - signed change tracking on stationary receivers (edit batches).
- **Accounting:**
  - total and per-unit parameters and memory;
  - convergence;
  - load and inference time;
  - supervision generation, optimisation, checkpoint traffic and oracle cost, each separately;
  - wall-clock to matched quality.

  Never claim a speedup from parameter or unit counts.
- **Visual:**
  - ownership map with seams;
  - seam crops;
  - error maps;
  - a changed mover, a stationary changed receiver and an unaffected region (edit batches).

## 8. Prior Art and Novelty

Priority historical design references (decision 8):
- Temporal Radiance Caching (2007);
- Instant Caching (2009);
- An Efficient GPU-Based Approach for Interactive GI (2009);
- Voxel Cone Tracing (2011).

These, together with hierarchical/selective GI (Drettakis & Sillion 1997 — not the starting point, decision 9; Luksch et al. 2019; Enlighten) and neural per-object composition (NeLT 2023; Superposed Deformable Feature Fields 2024), are prior art. Keep distinct:
- surface representation;
- radiance caching;
- learned parameter ownership;
- transport invalidation;
- sampling/scheduling.

A defensible contribution must concern the validity and reconstruction of learned owned transport, correct nonlocal dependency closure, and measured quality/cost, not "per-region neural models" or "a hierarchy" alone.

## 9. Interpretation and Stop Conditions

- Stop and report if:
  - the baseline is not credible;
  - target semantics cannot be preserved;
  - ownership is impossible without changing scope;
  - surface correspondence fails;
  - composition fundamentally fails;
  - controls cannot be maintained.
- A stop ends the route, not the direction.
- Attribute before interpreting:
  - baseline limitation;
  - candidate limitation;
  - ownership isolation;
  - optimisation/capacity;
  - partition;
  - supervision/correspondence;
  - compositing.
- Do not add rescue heuristics; do not declare the hypothesis dead or successful from one batch.

## 10. Present Next Question

> On a scene where the original substrate is credible, can recomputing the complete state of only the transport-invalid owned units recover the physical GI change of a geometry edit, while preserved units stay valid, and is the dirty-unit set meaningfully smaller than the whole asset?

This awaits the user's Batch-2 decision.
