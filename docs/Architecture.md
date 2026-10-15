# Architecture — Surface-Owned Neural Transport Units

**Updated:** 2026-10-15 (worklog 34)  
**Status:** Batch 1 of ≥ 5 reported. Ownership contract demonstrated exactly on an RNA-based candidate; static-fidelity verdict NOT adjudicated (baseline-credibility stop condition); radiometric recovery after a real edit NOT YET TESTED  
**Branch:** Selective_Recompute  
**Historical architectures:** docs/history/ARCHITECTURE_LIFECYCLE_SEPARATION_2026-10-07.md, docs/history/ARCHITECTURE_INTRA_OBJECT_FEATURE_REFIT_2026-10-08.md, docs/history/ARCHITECTURE_OBJECT_LEVEL_SELECTION_2026-10-12.md

## 1. The Architecture Question

After a geometry edit, which **transport units** hold invalid learned transport, and can the **complete owned neural state of only those units** be recomputed while every valid unit is reused unchanged?

- **Selection unit:** a Surface Transport Unit U_i = (S_i, Θ_i). It can be finer than a scene object (user decision 2); a whole object is the coarsest unit.
- **Recompute unit:** all of Θ_i (decision 4). This is never a subset of cells, channels or layers inside a jointly owned model; that was worklog 30's tested and negative hypothesis.
- **Coarse comparison:** object-level full recomputation, where every unit of an object is recomputed together (decision 5).
- **Baseline substrate:** RNA, adobe-research `66b5b098` (decision 1). It is a substrate, not a commitment.

## 2. Unit Contract (mandatory for every candidate)

**Surface ownership:**
- every evaluated surface sample resolves to one owner, or to a declared explicit composition of owners;
- report coverage, identity, boundary behaviour, overlap semantics, mapping stability and coordinate conventions;
- the owner function must be identical across training, rendering and later updates.

**Learned-parameter ownership:**
- each unit has an identifiable trainable set Θ_i;
- updating U_i changes no Θ_j (j ≠ i);
- every parameter in Θ_i may change.

**Shared components.** A shared component is allowed only with defined semantics, either:
- frozen forever (a fixed function), or
- declared as a coupling.

If a shared trainable decoder must change to recover one unit, that is a representation coupling to report, never to hide with optimiser masks.

**Transport semantics:**
- a unit's targets are full-scene GI, including light from other objects, so ownership does not mean physical locality;
- keep RNA's semantics: outgoing radiance under the dataset light distribution, with direct visibility selecting the lit or shadowed output;
- declare any change to these semantics as a confound.

**Report separately:**
1. parameter preservation;
2. output preservation;
3. radiometric validity.

## 3. Update Model

~~~text
T0 geometry G0 -> train all units U_i^0 on full-scene transport
edit G0 -> G1
   D_G: units/objects whose geometry changed
   D_T: units whose learned transport is invalid under G1 (includes stationary receivers)
   for i in D_T:  Θ_i^1 = FULL_RECOMPUTE(Θ_i^0 or init, supervision_i(G1))   # all of Θ_i eligible
   for j not in D_T: Θ_j^1 = Θ_j^0  (bit-identical)
   render: compose all units on G1
~~~

**D_G ≠ D_T** (worklogs 21–25, confirmed in worklog 31). Whether D_T is sparse at unit granularity is an open empirical question. At object granularity on the compact teaset it was all objects (worklog 31).

## 4. Batch 1 Candidate (worklog 32): Independent Complete RNA Units

**Representation.** K complete upstream `NeuralSurfaceTriPlane` models. Each unit has:
- its own triplane, framed on the unit's AABB (+2% margin);
- its own MLP.

Nothing trainable is shared.

**Partitions (geometry only, predeclared):**
- P1: whole asset;
- P4: one unit per part;
- P16: k-means per part (tray 9, tea pot 4, milk pot 2, tin 1).

The owner of a point is its exact part (point-to-triangle distance at contact zones), then the nearest dense sample's unit.

**Dispatch:** the owner id is appended as a 4th position channel, and each pixel is evaluated by its owner only (hard selection).

**Training:** upstream Lightning module, loss, optimiser and schedule, unchanged.

**Capacity:**
- matched: A's texel size per unit and MLP width 512/√K. Matched P16 still holds 1.89× A's parameters because unit AABBs overlap.
- P16w: full-size units, 16× A (diagnostic only).

Code is in `experiments/rna_surface_ownership/`; the protocol is in `protocol/rso_v1.json`.

**Evidence** (details in worklog 32; C2 = rough nickel α 0.3; C1 = canonical near-mirror):

| property | result |
|---|---|
| static credibility of original RNA (predeclared ≥ 25 dB) | **FAIL** — C1 17.97 dB, C2 24.92 dB (stop condition; fidelity verdict not adjudicated) |
| P16 vs A, C2, descriptive | −0.19 dB, display MAE 1.02×, boundary 1.05×, HDR logL1 **1.15×**, seam discontinuity **1.16×** (C1: 1.11×, 1.39×) |
| P16w (16× params) vs A, C2 | logL1 1.12×, seams 1.14× — the cost is **not** a capacity shortfall |
| P1 (local frame only) vs A | equal (C2) / better (C1) — local normalisation is harmless |
| REINIT (unit re-initialised + fully retrained, 15 frozen) | parameters of 15 units and **100% of their pixels bit-identical**; u* recovers to 0.95× / 1.00× of its prior error |
| composition | correct; hard selection leaves **visible shading steps at intra-part seams** |

**Inherent limits of this candidate:**
- seam steps under hard selection;
- no cross-unit sharing of angular/material learning (hypothesised cause of the HDR cost);
- capacity overhead from overlapping per-unit grids.

**Unknowns:**
- radiometric recovery after a physical edit;
- D_T sparsity at unit granularity;
- any cost advantage (REINIT training cost is 0.65× of a full P16 run with this implementation, because all pixels still pass through all units).

## 4b. Batch 2 (worklog 34): Boundary Attribution and P16-PU Composition

**Seam attribution** (C2; 4 dev views vs 8× references, existing checkpoints):
- **Independent learned functions (H2), dominant.** Seam-pair error correlation is 0.24 vs 0.71 inside units; A shows no drop. A neighbour evaluated across the cut is about as accurate as the owner, yet disagrees with it by as much as each errs.
- **Hard-selection composition (H3) exposes the mismatch.**
- **Supervision starvation is contradicted.** The band has full light coverage, and error is uncorrelated with per-unit sample counts.
- **A small near-seam error elevation** (≈ 13% of the excess) has an unresolved mechanism.

**P16-PU (composition contract, no learned-state change).**
- **Composition:** output = (1 − w) f_own + w f_nbr, within h = 7.2 mm of same-part seams; outside the band it is exactly f_own.
- **Ownership:** parameter ownership unchanged. A unit's output influence is S_i ∪ B_i (its neighbours' band).
- **Held-out results** (36 C2 views vs 1024-spp references):
  - removes 86% of P16's excess seam discontinuity;
  - lowers seam and band error (−9% / −6%);
  - interior, logL1 and high-frequency error unchanged;
  - +0.025 dB image-wide.
- **REINIT regression:** changed outputs equal S_12 ∪ B_12 exactly (3.2% in neighbour bands); recovery 0.993×.
- **C1 stress:** the jump is reduced without an accuracy gain.
- **What it does not fix:** P16's remaining fidelity loss is interior and per-unit (−0.20 dB, logL1 1.18× vs A). Composition cannot fix it.

## 5. Candidate Space for Later Batches (not authorised)

Each is a different realisation of the same contract, to be compared one factor at a time:
- **B1 (Batch 1):** independent complete units with hard selection.
- **Frozen shared decoder + owned per-unit features.** The decoder is trained once and frozen forever, so it is a fixed function; units own features (and optionally a small head). The semantics must state what happens if the decoder becomes inadequate.
- **Overlapping units with explicit partition-of-unity blending.** Implemented as P16-PU in worklog 34; output influence is accounted as S_i ∪ B_i.
- **Non-RNA substrates** (decision 1 allows replacing RNA later), e.g. worklog 31's explicit surface radiance units or other neural surface representations.

## 6. Historical Design References (decisions 8–9)

The priority references are 2007–2013 interactive/dynamic GI:
- Temporal Radiance Caching (TVCG 2007);
- Instant Caching (CGF 2009);
- An Efficient GPU-Based Approach for Interactive GI (SIGGRAPH 2009);
- Voxel Cone Tracing (CGF/PG 2011).

They inform caching, invalidation and scheduling design. None defines **learned-parameter ownership**, which is this architecture's question. Keep distinct:
- surface representation;
- radiance caching;
- learned parameter ownership;
- transport invalidation;
- sampling/scheduling.

The 1997 line-space hierarchy is prior art, not the starting point. See docs/RESEARCH_CENTRIC_TOPIC.md §5.

## 7. Prior Evidence (preserved, scoped)

- Worklogs 21–27: stale nonlocal transport under a part edit (problem evidence) and recompute-cost context.
- Worklogs 28–29: compact relational-state candidates negative on held-out T3 (separate, on hold).
- Worklog 30: intra-object triplane-cell masking negative. This is a different hypothesis, not a test of owned units.
- Worklog 31: explicit per-part surface radiance units (no learned decoder):
  - compose to 8DNA-level T0 quality;
  - all-local refit recovers T3 (tracking gain ≈ 1);
  - object-level D_T = all four objects, so no saving at object granularity;
  - whole-object refit degrades unaffected regions inside dirty objects (MC noise), which motivates finer ownership;
  - no wall-clock saving in that implementation.
- Worklog 32: section 4.

## 8. Implementation Frontier and Non-Goals

**Next decision (user):**
1. How to meet the static-credibility precondition (a predeclared scene/material where RNA clears the bar, or a prospectively revised criterion). Worklog 33 found no integration defect:
   - near-mirror C1 is an RNA representation limit, which only a different substrate would lift (decision 1);
   - the C2 regime is viable if scored against converged targets.

   Against converged targets, Batch-1 ownership costs on C2 are slightly larger (P16 −0.67 dB, logL1 1.28×; 4 views).
2. Then either:
   - radiometric validity of unit recompute under a physical T0 → T3 edit (recommended in worklog 32); or
   - a representation variant from §5.

**Not before feasibility (decision 10):**
- dependency tracking;
- dirty-unit detection;
- hierarchy;
- efficiency/scaling claims;
- automatic advance to the next batch.

## 9. Completion Condition (per batch)

> Can a learned transport representation be divided into independently owned surface units whose complete state can be reconstructed while all other units are preserved, without unacceptable static-GI fidelity loss — and, in later batches, can recomputing only the transport-invalid units recover current-scene GI?

A negative result must be attributed to one of the following before it is interpreted:
- baseline limitation;
- candidate limitation;
- ownership isolation;
- optimisation/capacity;
- partition;
- supervision/correspondence;
- compositing.
