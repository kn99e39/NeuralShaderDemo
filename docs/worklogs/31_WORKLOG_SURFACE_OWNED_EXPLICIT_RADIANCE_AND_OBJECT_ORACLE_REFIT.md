# Worklog — Surface-Owned Explicit Radiance Units and Whole-Object Oracle Refit on the Teaset (2026-10-12 – 2026-10-13)

## SESSION QUESTION

This batch was started from the user's "Representation Ownership Architecture Study" prompt.
Its question: can a learned transport representation whose state is radiometric, independently
reconstructible and local reproduce high-quality glossy GI? And can geometry-dependent GI then
be reconstructed by updating only the affected state?

Mid-batch, on 2026-10-12, the user rewrote the living documents (`6738be1`). The selection unit
became the **whole object**: a dirty object's entire model is refit, and intra-object masks are
not the main experiment. With the user's confirmation, the interrupted batch was finished and
its selection step reported at **whole-object granularity**. A unit-level oracle subset was run
only as a labelled diagnostic.

**Answer (one substrate: explicit per-part surface tables; teaset T0 → T3; envmap regime; RTX 5080):**

- **O0-like (static, per-object ownership): yes.** Per-part explicit radiance tables have no
  learned shared decoder and are supervised by whole-scene Monte-Carlo targets. They reproduce
  T0 at 8DNA's quality: interaction-ROI display MAE 0.033 vs 8DNA 0.034, R_aff_stat 0.039 vs
  0.054.
- **All-object = all-local warm refit (C): recovers T3.**
  - Recovery rule 1.16 / gain 0.92; seed 1: 1.21 / 0.92.
  - Stationary-receiver tracking gain 1.01. R_aff falls from 0.095 (frozen) to 0.040.
  - The predeclared gate G_local additionally required R_delta ≤ 0.5. C reaches 0.509 and
    **fails it by 0.009**. Scratch B reaches only 0.516, so the bar sits at the representation's
    static floor; the formal gate is still recorded as failed.
- **Whole-object oracle selection is degenerate on this scene.**
  - At the predeclared 2% rule every object is transport-dirty: the distant tin holds 2.7% of the
    change energy, 2.8% in the replicate. So E = C, and E vs C is a null comparison.
  - The predeclared 5% sensitivity arm (E5) freezes the tin. The tin stays bit-identical and E5
    matches C (R_aff 0.040 vs 0.040), but saves only ≈3% of latency: the tin is 8% of the surface.
- **Geometry-dirty-only refit (DG, the moved milk pot) fails, as the object-level docs
  predicted.** The stationary receivers stay stale: R_aff_stat 0.127 = frozen; R_aff 2.2× C.
- **Cost:** on this compact scene with this implementation, no selective arm was cheaper in
  wall-clock. Supervision samples scale with the selected surface (DG 10%), but fixed per-step
  tracer overhead and GPU memory pressure in the subset arms erased it (section COST).

## RELATION TO EARLIER WORK AND LIVING DOCUMENTS

- Worklogs 21–30 are unchanged.
- Worklog 30's negative result concerns triplane-cell masking inside one 8DNA asset with shared
  flows frozen. This batch tests a different ownership structure (no learned shared decoder) and
  does not reinterpret worklog 30.
- The living documents changed during the batch:
  - `8a64244` … `7d19c35`: gate F0+F1, before the batch;
  - `f88b244` … `6738be1`: object-level intent, during the batch.
  The protocol records both. Its selection reporting follows `6738be1`.
- The user's preference for the next O0/O1 batch (compare this explicit substrate with
  per-object RNA) is recorded, not executed. The RNA ownership work started on 2026-10-12 is
  worklog 32.

## PRIOR-ART AUDIT (phase 1)

- **Method:** workflow `wf_9eab5e1a-2d3`, five deep readers (paper + code), five adversarial
  verifiers (all rated their audit "high"; 0–2 refuted checks each), one completeness critic.
- **Verified table:** recorded in `experiments/surface_owned_radiance/protocol/sor_v1.json`
  (`prior_art_audit`).
- **Shared decoder owns directional/glossy structure:**
  - Vertex Features for Neural GI (Su et al., SIGGRAPH Asia 2025; MIT code);
  - Neural Radiosity (2021);
  - OctaOctree Neural Radiosity (Ren et al.; MIT code; shared disparity network + MLP).
  These risk the 8DNA failure mode.
- **Own radiance directly:**
  - Frequency-Aware Spatio-Angular Gaussians (Shah, Lucas, Drettakis, EGSR 2026; code unreleased);
  - Gaussian Light Transport (Attimont, Subr, Soler, SIGGRAPH Asia 2026; code unreleased);
  - N-D Gaussians (SIGGRAPH 2024; Taichi code).
- **No released, runnable implementation met both local ownership and local support.**
- **Correction found by the critic and verified:** the teaset BSDF is **Beckmann** (Mitsuba's
  default for `roughconductor`), not GGX; α 0.05 / 0.1. No project document had said GGX; only the
  audit context did.

## THE CANDIDATE (composed of established mechanisms)

- **Unit:**
  - one anchor per (part, voxel of size h = 0.02, dominant-normal bin) in the part's canonical
    frame, so units move rigidly with their part;
  - 20 439 units: tray 10 848, tea pot 5 580, milk pot 2 271, tin 1 740.
- **Owned state:** one 32×32 octahedral RGB table per unit over the full sphere (front and back
  sides), holding outgoing radiance of paths with ≥ 2 asset vertices.
- **Query:** partition-of-unity Wendland weights over ≤ 16 anchors of the same part (radius 1.5 h,
  normal gate). The output is linear in the entries, and an entry is radiance.
- **Shared computation:** fixed only (search, interpolation, frames). There is no learned shared
  parameter, so the "all-local refit with global computation frozen" arm coincides with
  all-object refit.
- **Supervision:** online 1-spp Monte-Carlo targets in the current whole scene from worklog 24's
  validated `decompose()`, i.e. cross-object transport included. Direct light (v ≤ 1) is
  physical (the worklog-24 class images), as 8DNA computes it analytically.
- **Reused mechanisms:**
  - surface-attached units (Vertex Features / surfels);
  - explicit directional radiance per unit (octahedral maps, PlenOctree / SD-tree style);
  - path-traced cache targets (NRC-style baking);
  - analytic direct light (8DNA);
  - worklog 30's CRN oracle and region evaluation.
- **New hypotheses (not novelty claims):**
  - the ownership contract: no learned decoder, each unit owns its outgoing ≥ 2-vertex radiance;
  - whole-object update semantics;
  - computational locality of supervision.

## IMPLEMENTATION FACT

- **Code:** `experiments/surface_owned_radiance/` (README lists every file). Commits:
  `2fe3a10` (all code, protocol, tests, evidence chain) and `d726b5a`-era helpers. Every evidence
  stage ran at `2fe3a10` from a clean tree.
- **Protocol `protocol/sor_v1.json`:** predeclared, with four amendments before any T3 fit:
  1. batch 524 288 with doubled steps (1 M batches spilled GPU memory);
  2. arm E5 added, with disclosure that a 2 M-query smoke oracle had shown the tin near the 2%
     threshold;
  3. test thresholds;
  4. pilot-based resolution choice.
- **Capacity pilot (T0 only, before the protocol):** h 0.02 / 32²; still improving at 400 steps.
- **Tests (`tests/test_sor.py`, all pass at `2fe3a10`):**
  - octahedral round trip;
  - stencil: partition of unity, same part only, normal gate, 100% coverage;
  - object isolation: changing one object's tables changes no query on another object; linearity;
  - combined lookup;
  - synthetic fixture with an analytic radiance replacing the tracer: fit, then an edit on one
    object and a whole-object refit; others bit-identical, edited object recovers 0.188 → 0.016;
  - GPU vs numpy query path equivalence;
  - seed / data separation.
- **Incidents:**
  - A session ended mid-pilot.
  - A first 1 M-batch pilot was stopped: CPU-bound query sampling, rewritten on the GPU.
  - Torch allocator caching starved DrJit in an E smoke run (fixed with `empty_cache` before
    the evidence run).
  - None touched evidence.

## MEASUREMENT (envmap regime; display MAE vs the corrected references; final step 1600)

### Static (O0-like), gate G0: pass

| | interaction ROI | R_aff_stat | R_unaff | asset |
|---|---|---|---|---|
| explicit tables, T0 composed (direct physical + learned ≥ 2-vertex) | 0.033 | 0.039 | 0.024 | 0.027 |
| released 8DNA T0 | 0.034 | 0.054 | 0.025 | 0.030 |
| reference T0 → T3 change | 0.066 | 0.112 | — | — |

- **Held-out relative MAE of the ≥ 2-vertex radiance** (500 k random queries, 32 spp): 0.52.
  This is dominated by the targets' own MC noise and by angular resolution off the camera
  directions.
- **T0 pretraining:** 1.67 G samples, 32 min (trace 1504 s, optimisation 396 s), peak 7.7 GiB.

### Oracle (CRN, 16.8 M queries, 81 s; replicate seed agrees)

- **Affected queries:** 5.3% of accepted queries; 18 976 of 20 439 units carry some change.
- **Change energy by distance to the mover:** < 0.05: 0.39; 0.05–0.15: 0.27; 0.15–0.30: 0.18;
  ≥ 0.30: 0.16.
- **Object shares:** tray 0.587, milk pot 0.248, tea pot 0.139, tin **0.027**.
  Replicate: 0.588 / 0.247 / 0.137 / 0.028.
- **Relative change** (mean |ΔL| / mean L): tray 0.34, milk pot 0.85, tea pot 0.26, tin 0.10.
- **Dirty objects:** 0.02 rule → all four. 0.01 → all four. 0.05 → tray, milk pot, tea pot.
- **Unit M95:** 7 969 units (39%, 43% of the area). Replicate Jaccard 0.88.

### Arms (seed 0; seed 1 for C and E)

| arm | refit objects | R_aff | R_aff_stat | R_mover | R_unaff | rule ratio / gain | stationary tracking gain / R_delta | latency (s) |
|---|---|---|---|---|---|---|---|---|
| A frozen | — | 0.095 | 0.127 | 0.053 | **0.024** | 2.50 / 0.02 | −0.03 / 1.04 | 0 |
| B all-object scratch | all | 0.040 | 0.047 | 0.031 | 0.027 | 1.13 / 0.92 | 1.01 / 0.52 | 948 |
| C all-object warm (= all-local) | all | 0.040 / 0.039 | 0.047 / 0.046 | 0.031 / 0.030 | 0.027 / 0.027 | 1.16 / 0.92; 1.21 / 0.92 | 1.01 / 0.51; 0.99 / 0.50 | 955 / 950 |
| E oracle dirty (0.02) | all (= C) | 0.039 / 0.040 | 0.047 / 0.048 | 0.029 / 0.031 | 0.027 / 0.026 | 1.22 / 0.90; 1.19 / 0.95 | 1.02 / 0.49; 0.98 / 0.51 | 954 / 947 |
| E5 oracle dirty (0.05, sensitivity) | tray, milk pot, tea pot | 0.040 | 0.049 | 0.029 | 0.026 | 1.24 / 0.92 | 0.99 / 0.51 | 923 |
| DG geometry-dirty only | milk pot | **0.087** | **0.127** | 0.034 | 0.024 | 2.50 / 0.02 | −0.03 / 1.05 | 1227 |
| U unit-level M95 (diagnostic) | 39% of units | 0.040 | 0.046 | 0.032 | 0.027 | 1.15 / 0.93 | 0.99 / 0.50 | 5064 |

- **Gate G_local: fails narrowly** (C R_delta 0.509 > 0.5; the rule passes). Per protocol, the
  E screening is therefore descriptive only.
- **E5 vs C:**
  - quality ratio 1.01;
  - the tin's 1 740 units bit-identical;
  - all 12 894 pixels that read only frozen units are bit-identical to the frozen render;
  - latency speedup at Q* = 1.10 × C: 1.03.
- **DG:** the 18 168 frozen units and the 110 323 pixels reading only them are bit-identical.
  Its stationary receivers are as stale as the frozen model.

### Cost (RTX 5080; latency = T3 handed over → snapshot, excluding evaluation; oracle excluded)

| arm | supervision samples | trace s | optimisation s | peak torch GiB |
|---|---|---|---|---|
| C | 834 M | 754 | 198 | 8.2 |
| E5 (92% support) | 767 M | 739 | 180 | 10.5 |
| DG (10.5% support) | 87 M | 586 | 623 | 10.5 |
| U (51.5% support) | 429 M | 636 | 4422 | 10.5 |

- The subset arms' supervision scales with their support, but wall-clock does not:
  - each step's tracer call carries a fixed overhead (~0.3 s), which dominates small batches;
  - the subset arms ran at ~10.5 GiB versus ~8.2 GiB, which plausibly pushed DrJit + torch into
    the WDDM memory spill seen in worklog 30. The optimisation phase was 3× (DG) and 22× (U)
    slower than C's.
- These are implementation costs of this prototype, not a measured property of selection, and
  were not optimised after the fact.

## OBSERVATION

- **Signed T0 → T3 change maps (`03`):** every recovering arm reproduces the main reflection and
  occlusion bands on the tea pot and tray. All of them also add blotchy Monte-Carlo speckle over
  stationary surfaces. Hence refitting a whole object raises its unaffected-region error:
  R_unaff 0.024 frozen → 0.027 after refit. Freezing the tin (E5) keeps it at its T0 quality.
- **DG:** removes the milk pot's own error. The bands on the tea pot and tray stay at their frozen
  values.
- **Quality vs latency (`05`):** every all-object and object-subset arm converges along the same
  curve. U is shifted right by its slow optimisation.

## INTERPRETATION

1. **Ownership without a learned shared decoder works on this case.**
   - Explicit per-part outgoing-radiance tables trained on whole-scene targets compose to 8DNA-level
     T0 quality.
   - All-local (all-object) refit recovers the T3 change, with tracking gain ≈ 1 on the
     stationary receiver.
   - This is the outcome worklog 30's 8DNA arms D/E could not reach, because there the shared
     flows owned the transport.
   - Scope: one asset, one regime, an explicit (non-neural) substrate whose memory cost
     (63 M entries, 0.25 GB) is far above 8DNA's.
2. **Object-level selection is radiometrically dense here.**
   - In the compact teaset, every object holds measurable change. Even the tin, ≈ 0.35 from the
     mover, holds 2.7% of the change energy and 10% relative change.
   - The predeclared conservative rule marks all four objects dirty, so whole-object selective
     recomputation cannot save anything on this scene.
   - At a looser threshold the only clean object is small: bitwise preservation holds, but the
     saving is negligible.
   - A scene with genuinely transport-valid, sizeable objects is needed to test O1 economics. The
     object-level docs anticipated this.
3. **D_G ≠ D_T is confirmed:** mover-only refit leaves the stationary receivers fully stale.
4. **Whole-object refit has a preservation cost inside dirty objects.** Their unaffected parts are
   re-estimated from fresh noisy samples and degrade unless the refit budget matches the
   pretraining. A finer ownership granularity would avoid this, which motivates the user's
   worklog-32 direction.
5. **The cost question is unanswered by this prototype:**
   - supervision volume follows the selected support;
   - this implementation's per-step overheads and memory behaviour did not let wall-clock follow.

## ANSWERS TO THE BATCH'S FINAL-REPORT QUESTIONS

1. **Representation selected:** explicit per-part surface units owning octahedral tables of
   outgoing ≥ 2-vertex radiance. Selected because the audit found no runnable representation in
   which local state owns glossy radiance; this one makes ownership exact and testable.
2. **Reused:** surface-attached units, explicit directional storage, path-traced cache targets,
   analytic or physical direct light, worklog-24 estimator, worklog-30 oracle and evaluation.
3. **New hypotheses:** the ownership / update contract (no learned decoder; whole-object or unit
   refit from current-geometry supervision).
4. **What owns high-frequency transport:** the units' tables, up to 32² angular resolution (about
   6° per hemisphere texel).
5. **What remains shared:** only fixed computation.
6. **Can local state recover configuration changes:** yes (C, U, E5), with tracking gain ≈ 1. The
   strict R_delta gate missed by 0.009.
7. **Are local updates cheaper:** not in wall-clock with this implementation; supervision volume
   is proportional to the selected surface.
8. **Unresolved:**
   - neural capacity per unit versus memory;
   - a scene with sizeable transport-valid objects;
   - an efficient subset-update implementation;
   - noise-preserving refit inside dirty objects.
9. **Further selective-invalidation research?** The ownership semantics are supported on this
   substrate. Economics need a scene where D_T is sparse and an implementation whose cost follows
   the selected support. The direction is a user decision (worklog 32 started the RNA-based
   ownership line).

## ARCHITECTURE DECISION

- **Supported:** explicit per-object (and per-surface-unit) radiance ownership with no learned
  shared decoder reproduces T0 GI at 8DNA-level quality. All-object = all-local refit recovers
  the canonical T3 change (rule pass, tracking gain ≈ 1; the strict R_delta gate missed by 0.009).
- **Not supported on this scene:** a cost advantage from whole-object oracle selection, because
  the transport-dirty set is all objects.
- **Not established:** a cost advantage in any arm, because of implementation overheads.
- No hierarchy or dependency predictor was built. O2 is not authorized.

Artifacts:
- `results/surface_owned_radiance/v1_2fe3a10/`: units, held-out, arms, oracle, replicate,
  `metrics.json`, chain logs;
- `results/surface_owned_radiance/pilot/`;
- `results/evaluation/31/` (panels `01`–`06`, `07_key_numbers.json`, manifest).
