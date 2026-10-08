# Worklog — Oracle Selective-Refit Feasibility on the 8DNA Teaset Triplane (gate F0+F1) (2026-10-08)

## SESSION QUESTION

Even with oracle knowledge of which learned transport state the teaset T0 → T3 edit
invalidated, can a high-quality pretrained neural GI representation be selectively refit
while preserving unaffected transport and substantially reducing quality-matched update cost?

This is gate F0+F1 of `docs/RESEARCH_ROADMAP.md` (commit `7d19c35`). No hierarchy, dependency
detector, new representation or new renderer was built.

**Answer (one substrate, one edit, envmap regime, RTX 5080): no.**

- **F0 (substrate):** 8DNA's released teaset asset was selected. It has a credible T0, a
  strong indirect T3 signal and a T3 scratch ceiling that recovers. Its only position-indexed
  learned state is the triplane.
- **F1 (oracle selective refit):** the batch stops at the predeclared gate `G_D_meaningful`
  in both seeds. With the shared networks frozen, updating **all** triplane cells (D) removes
  only 20% of the stale error that global warm start (C) removes. On the stationary
  receiver, D/E follow 17–23% of the physical change (gain), against 56–67% for C and scratch B.
- The oracle-selective arm E therefore matches D (within 4%), but D itself is not a meaningful
  reference.
- The remaining screening criteria were measured anyway, and both fail:
  - **Preservation:** unaffected-region error +7.5% versus a +5% limit, with 1 changed
    frozen-stencil pixel.
  - **Cost:** E reaches the matched-quality target at 0.51–0.53× D's speed, i.e. about 2× slower.
- **Failure attribution (this substrate):**
  1. Primarily **state ownership by the shared networks**. A post-hoc swap shows C's
     interaction-ROI recovery is carried by its shared weights, not its triplane.
  2. Then **parameter coupling through triplane projections**. 95% of valid training samples
     touch an M95 cell, and 97% of unaffected pixels touch an updated cell.

  This is a substrate result, not evidence against every selective-refit architecture.

## RELATION TO EARLIER WORKLOGS AND LIVING DOCUMENTS

- Worklogs 21–29 are unchanged. Reused without modification:
  - worklog 21's states, envmap scene, ROIs and frozen renders;
  - worklog 22's corrected references and `refit_block` rule;
  - worklog 27's scratch T3 rebuild (snapshots, renders, timing).
- The living documents were rewritten for the Selective_Recompute branch during this session
  (`8a64244` … `7d19c35`, user commits). The draft protocol was reconciled with them before any
  evidence run. Two changes followed:
  - an explicit criterion-based justification for using 8DNA despite their caution against
    picking RNA/8DNA for their historical role;
  - a fixed-budget scratch arm run in the same loop as C/D/E, with worklog 27's long rebuild
    kept as the capacity ceiling.

## SUBSTRATE SELECTION (F0)

Two candidates were audited; RNA was excluded on existing evidence: its G1 margin is 4% and
worklog 27's rebuild had no rule-satisfying checkpoint. Full criterion table:
`experiments/selective_refit_oracle/protocol/sro_v1.json` → `substrate_selection`.

| criterion | Neural Radiosity (`51a2b8c`, code audit only) | 8DNA teaset (`4a2157c`, released ckpt) |
|---|---|---|
| T0 quality vs a nontrivial change | unknown: never run here; α 0.05/0.1 nickel puts the reflected image in the shared MLP's direction input | yes: 29.5 dB; same-surface T0 error 0.054 vs change 0.112 (R_aff_stat) |
| enumerable localized units | yes (dense multi-resolution grids; coarse levels effectively global) | yes: triplane 3 × 8 × 64 × 64 = 98 304 parameters, 12 288 cells, exact bilinear support |
| selective update / shared frozen | yes / yes | yes / yes (571 017 shared parameters) |
| same-scene reference | after scene conversion | yes (worklog-22 corrected references) |
| reproducible here | unvalidated (pinned torch 1.13/cu117, Mitsuba 3.1.1; no SM 12.0) | yes; scratch T3 ceiling recovers twice (WL22 1.07/0.70, WL27 1.149/0.629) |

- Neural Radiosity was not selected because criteria 1 and 6 need a separate
  substrate-validation batch. This is not a negative result about it.
- 8DNA was chosen on these criteria, not for its historical role:
  - In the teaset the learned asset is the whole scene geometry.
  - Its learned state is exactly the ≥ 2-vertex intra-asset transport that the edit
    invalidates; direct scattering is analytic.
- Residual caveat: the triplane's axis-projection coupling is a property of this substrate,
  not of localized GI state in general.

## IMPLEMENTATION FACT

### Code (`experiments/selective_refit_oracle/`, README lists every file)

- **Protocol `sro_v1.json`.** Committed before any evidence. It defines the substrate audit,
  oracle, arms, training contract, regions, gates and stop rules. Five pre-evidence amendments
  plus one memory-spill amendment are recorded in its `amendments`.
- **State units (`sro_common.py`).** One unit = one triplane cell with its 8 channels. Support
  is upstream `fetch_2d`'s exact stencil (4 cells per plane, 3 planes).
- **Oracle (`sro_oracle.py`).**
  - Traces one full upstream reload of 8DNA training rays (268 M rays; N = H = 128,
    SPP = 256) in T0 and T3 with common random numbers.
  - A ray is *affected* if any supervision field differs: validity, occlusion, depth, xi, xo,
    wo or throughput.
  - Per-cell integer-weight accumulation makes the sums deterministic:
    A = A3 (affected T3 support) + A0 (affected T0 support).
  - M95 = the smallest top-A set covering 95% of A.
  - S = equal-size proximity mask: cells nearest the projected milk pot (T0 and T3).
- **Arms (`sro_train.py`).** Upstream `training_step` loss, Adam 5e-4, norm clipping 5, a
  reload and shuffle every 8192 steps, 32 768-step budget, snapshots at
  0 / 128 / … / 32 768, and the same data stream for every arm of a seed (digests identical).
  - **B:** scratch, all parameters.
  - **C:** released checkpoint, all parameters.
  - **D:** released checkpoint, triplane only.
  - **E:** released checkpoint, M95 cells only. Implemented as D's batch restricted to the
    samples touching a selected cell, with the loss normalised by the full valid count.
  - **S:** like E, with the spatial mask.
  - **E_mask:** a 256-step cost run of the masked full batch.
- **Regions (`sro_regions.py`).** Built from geometry and references only, before training
  (512² T3 frame).
  - R_aff_stat: stable stationary pixels with 7×7 display change > 3× reference noise
    (12 879 px).
  - R_unaff: stable stationary pixels with change < 1× noise (88 610 px).
  - R_mover: the milk pot at T3 (9 786 px).
  - R_aff = R_aff_stat ∪ R_mover. R_buffer: 4 208 px; R_unstable: 4 806 px.
- **Renders (`sro_render.py`).** Exactly the historical refit render: upstream mode at T3,
  512 spp, seed 0. Every arm's step-0 render equals worklog 21's released upstream T3 render
  bit for bit (max diff 0.0, 7 runs).
- **Metrics (`sro_metrics.py`).** Region errors, refit rule, change tracking (gain, E_delta,
  R_delta), frozen-stencil output identity, adaptation latency and gates.
- **Post-hoc analysis (`sro_posthoc.py`).** Ownership swap from existing checkpoints, per-step
  cost and per-group parameter change. Analysis only; it changes no verdict.

### Tests (clean tree; `results/selective_refit_oracle/v1_bc085ea/tests/`)

| test | result |
|---|---|
| state units: stencil × weights reproduce upstream `Triplane.forward` | max diff < 1e-6; parameter layout correct |
| masking: masked full-batch gradient vs the same batch with non-touching loss terms zeroed | rel. L2 diff **0.0** (non-touching samples contribute nothing) |
| restricted vs masked gradient | 1.4e-4 rel. L2 (float32 reduction order of smaller GEMMs); unselected gradients 0; frozen tensors and unselected cells bit-identical after 20 Adam steps |
| upstream loss equivalence | 0.0 diff to `train.py`'s expression |
| data equivalence: `sro_paths.generate` vs upstream `PathSamplingDataset.reload` | bit-identical (small config) |
| CRN: same scene twice | 0 affected rays; T0 vs T3 unaffected fields bit-identical; 0 rays within tolerance |
| mask determinism | primary oracle and its repeat bit-identical |
| timing accounting, checkpoint round trip (training load, upstream `load_asset`, released), data separation | pass |
| synthetic fixture: floor + receiver sphere + moving sphere, small 8DNA | pass (contracts only): frozen tensors/cells bit-identical, untouched held-out predictions bit-identical, touched loss decreases, affected rays median 0.05 from mover vs 0.30 unaffected |

The synthetic pass establishes no architecture result. It already showed the coupling
pattern: 7.5% of cells touched 99% of valid samples.

### Commits, environment, incidents

| commit | content / evidence produced |
|---|---|
| `8745838` | protocol, oracle, regions, arms, metrics, tests |
| `bc085ea` | chain driver; **primary oracle** |
| `b6f6420` | export script (presentation); oracle repeat, oracle seed 30031, tests, synthetic, regions |
| `d726b5a` | memory-spill fix (below); **every arm, render and `metrics.json`** |
| `5f0e8ce` | post-hoc analysis and reviewer exports |

All stages recorded clean trees.

**Environment:** RTX 5080 (16 303 MiB, driver 596.49), Windows 11, Python 3.11.9, torch
2.8.0+cu128, Mitsuba 3.5.1 / DrJit 0.4.6 (LoopRecord off, VCallRecord on), 8DNA `4a2157c`
unmodified, released `teaset.ckpt` sha256 `6feb5e1b…`. Nothing was pushed.

**Incidents** (neither touches a reported number):

1. **Clean-tree refusal.** I added `sro_exports.py` while the chain ran. The clean-tree guard
   refused the second oracle stage. The export script was committed and the chain resumed.
2. **GPU memory spill.** The first E_mask cost run showed torch reserving 28 GB on the 16 GB
   card, with a 189 s reload instead of ~14 s.
   - Cause: `sro_train` lacked upstream `on_train_epoch_start`'s `torch.cuda.empty_cache()`
     calls and kept the validation generator resident.
   - The chain was stopped before any arm finished, fixed at `d726b5a` (no math change) and
     re-run.
   - The aborted outputs are in `v1_bc085ea/superseded_memory_spill/` and are not evidence.
   - After the fix: reload 13.4–14.5 s, resample 1.8–2.0 s, max reserved 15.8–16.3 GB.

## MEASUREMENT

Display MAE vs the T3 reference, envmap regime, final snapshot (step 32 768); seed 0 / seed 1.

### Gates G0 / signal / ceiling

- **G0: pass.** Released T0 same-surface error is 0.054 in R_aff_stat and 0.034 in the
  interaction ROI, against reference changes of 0.112 and 0.066. Full-frame PSNR 29.5 dB.
- **G_signal: pass.** Reference repeat noise is 0.009 / 0.006 and neural seed noise is
  0.016 / 0.014 (R_aff_stat / interaction). The changes exceed 3× both.
- **G_B_capacity: pass.**
  - The worklog-27 scratch rebuild's final checkpoint gives interaction error 0.0442. Against
    its own pipeline's T0 (0.0385) that is 1.148, gain 0.62.
  - Against the released T0 (this batch's warm-arm denominator) it is 1.297; the denominators
    differ.
  - Its R_aff is 0.040, the best of all arms.

### Oracle (seed 30030, 105 s total: traces 34 s, compare + accumulate 23 s, decomposition 47 s)

- **Affected samples:** 8.9% of the 86.0 M valid T3 samples (and of T0).
- **Common random numbers:** 0 rays differ below tolerance.
- **Occupied cells:**
  - 3 034 of 12 288 triplane cells hold any supervision.
  - 3 010 of them (99.2%) hold some affected weight.
- **Affected weight by first-hit part (T3 side):** tray 48%, milk pot 39%, teapot 10%,
  tin 3%.
- **Reach beyond the mover:** 24% of the affected weight lies ≥ 0.15 from the milk pot, and
  13% ≥ 0.30.

| mask | cells | of all 12 288 | of 3 034 occupied | A coverage | share of all T3 support | collateral (unaffected share of its support) |
|---|---|---|---|---|---|---|
| M90 | 892 | 7.3% | 29% | 0.90 | 0.51 | 0.84 |
| **M95 (E)** | **1 195** | **9.7%** | **39%** | **0.95** (throughput-weighted 0.94) | **0.61** | **0.86** |
| M99 | 1 747 | 14.2% | 58% | 0.99 | 0.78 | 0.89 |
| S (equal size) | 1 195 | 9.7% | 39% | 0.84 | 0.40 | 0.82 |

- **Seed stability:** M95 from seed 30031 has 1 196 cells, Jaccard 0.994.
- **M95 vs S:** Jaccard 0.52.

### Quality at the fixed budget

| arm | trainable params | R_aff | R_aff_stat | R_mover | R_unaff | interaction ROI ratio / gain (rule) | tracking gain / R_delta on R_aff_stat |
|---|---|---|---|---|---|---|---|
| A frozen, attached | 0 | 0.085 | 0.110 | 0.052 | 0.025 | 2.23 / 0.02 (no) | −0.03 / 1.04 |
| step 0 (released, T3 frame) | — | 0.115 | 0.110 | 0.121 | 0.025 | 2.25 / 0.02 | −0.03 / 1.04 |
| B scratch, matched budget | 669 321 | 0.050 | 0.058 | 0.040 | 0.034 | 1.63 / 0.65 (no) | 0.67 / 0.61 |
| B scratch WL27, epoch 29 (ceiling) | 669 321 | 0.040 | 0.048 | 0.031 | 0.026 | 1.30 / 0.62 (own-T0 rule: 1.15, yes) | — |
| **C global warm** | 669 321 | **0.046 / 0.044** | 0.056 / 0.053 | 0.034 / 0.033 | 0.026 / 0.026 | 1.28 / 0.66 (no); 1.22 / 0.62 (yes) | 0.63 / 0.61; 0.56 / 0.60 |
| **D local state, shared frozen** | 98 304 | **0.077 / 0.077** | 0.095 / 0.094 | 0.054 / 0.054 | 0.028 / 0.029 | 2.09 / 0.22; 2.10 / 0.20 | 0.22 / 0.90; 0.23 / 0.89 |
| **E oracle selective (M95)** | 9 560 | **0.080 / 0.080** | 0.099 / 0.099 | 0.055 / 0.055 | 0.027 / 0.027 | 2.11 / 0.18; 2.11 / 0.16 | 0.17 / 0.93; 0.17 / 0.93 |
| S spatial equal-size | 9 560 | 0.078 | 0.096 | 0.055 | 0.027 | 2.15 / 0.17 | 0.20 / 0.90 |

- **Held-out validation (T3 path samples, loss):** C −0.326, D −0.318, E −0.316, S −0.315,
  B −0.230.
- **Changed state:**
  - C and D changed all 3 016 occupied-and-trained cells.
  - E changed 1 182 of 1 195 selected cells; S changed 943.
  - Unselected cells and every shared tensor stayed bit-identical in D, E and S.

### Screening (predeclared; both seeds)

| criterion | seed 0 | seed 1 | verdict |
|---|---|---|---|
| G_D_meaningful: D stale reduction vs A (R_aff) > 3 × neural seed noise **and** ≥ 0.5 × C's | 0.0077 vs C 0.0389 (0.20×) | 0.0084 vs 0.0407 (0.21×) | **fail** (C's own 0.039 is also below 3 × 0.015 = 0.045 per-pixel noise; see interpretation) |
| S1 quality: E ≤ 1.10 × D | 1.039 | 1.041 | not evaluated (gated by G_D) |
| S2 preservation: E R_unaff ≤ 1.05 × step 0, and no frozen-stencil pixel changes | 1.075; 1 px (max 3.8e-4) | 1.077; 1 px | **fail** |
| S3 frozen subset: \|M95\| ≤ 50% of eligible cells | 9.7% | 9.7% | pass nominally (39% of occupied cells) |
| S4 latency: t_D(Q\*) / t_E(Q\*) ≥ 2, with Q\* = 1.10 × D_final | D 93 s (step 4 096), E 175 s (step 8 192) → **0.53** | 89 s / 176 s → **0.51** | **fail** |

- Q\* (0.085 / 0.084) is essentially the frozen attached model's own R_aff (0.085).
- **Frozen-stencil pixels:** only 2 703 pixels (3.0% of R_unaff) have a 12-cell stencil
  containing no updated cell. The rest of the unaffected region shares at least one updated
  cell. The single changed frozen-stencil pixel (≤ 4e-4 linear) is unexplained. A plausible
  source, not tested, is the 5×5 sub-pixel lattice missing a jittered sample, or a secondary
  asset query.

### Cost (seed 0 unless stated; adaptation latency excludes validation and the oracle)

| quantity | value |
|---|---|
| fixed floor to the first usable snapshot (step 128) | 20.2–21.4 s, every arm: path generation 13.4–14.5 s, shuffle 1.8–2.0 s, optimisation 2.3–2.8 s |
| ms per step | C 22.7 / 21.0, B 21.2, D 19.2 / 17.9, E 19.9 / 19.5, S 19.8, E_mask 18.4 |
| restricted share of valid samples (E / S) | **94.6%** / 76.6% |
| latency to Q\* = 1.10 × D_final | C 24 s (step 256), D 93 s, S 96 s, B 103 s, **E 175 s**; WL27 scratch: epoch 0, 239 s |
| latency to 1.10 × C_final (0.051) | C 404 s, B 757 s, WL27 scratch 1 076 s; **D, E, S never** |
| oracle construction (excluded from E above) | 105 s |
| peak memory | torch allocated 15.7–15.9 GiB, reserved ≤ 16.3 GiB, whole-GPU 15.8–15.9 GiB, every arm; host peak 22.6–24.1 GB |
| updated parameters | C/B 669 321, D 98 304, E/S 9 560 |

### Post-hoc ownership swap (existing checkpoints; not a protocol quantity)

| model (rendered at T3) | interaction ROI error / ratio / gain | R_aff_stat | R_mover |
|---|---|---|---|
| C_s0 final | 0.044 / 1.28 / 0.66 | 0.056 | 0.034 |
| C's shared tensors + released triplane | 0.047 / 1.39 / **0.63** | 0.079 | 0.094 |
| released shared tensors + C's triplane | 0.076 / 2.24 / 0.11 | 0.095 | 0.076 |
| D_s0 final (triplane trained alone) | 0.071 / 2.09 / 0.22 | 0.095 | 0.054 |

C's relative L2 parameter change: triplane 0.22, albedo 0.22, flows 0.12–0.15, cubemaps
0.09–0.13.

## OBSERVATION

- **Crops** (`02`, `08`):
  - The frozen model keeps the dark smear at the milk pot's T0 base on the tray.
  - D, E and S remove most of the smear but leave a darker, smudged band under the moved pot
    and on the tea pot's side facing it.
  - C and both B variants are visibly closer to the reference there.
- **Signed T0 → T3 change on stable pixels** (`03`):
  - The reference shows the moved reflection and occlusion bands on the tea pot and tray.
  - C and B reproduce much of that structure; their maps also contain broader static
    differences.
  - D, E and S reproduce a weak version of the main bands only.
- **Quality vs latency** (`06`):
  - The warm arms start at the frozen level.
  - C drops fastest and keeps improving.
  - D, E and S flatten near the frozen attached model's R_aff by about 100–200 s.
  - The scratch arms start far worse and reach 1.10 × C's final R_aff only after 12.6 min
    (matched B) or 18 min (WL27).
  - Unaffected-region error stays near 0.026–0.029 for every warm arm and is far higher for
    scratch early on.
- **Oracle mask** (`04`, `05`):
  - The affected weight forms bands along the triplane projections of the milk pot.
  - M95 covers the occupied region around and through the mover's footprint on all three
    planes.
  - In the image, most stable pixels have a stencil partly inside M95, including surfaces far
    from the mover.

## INTERPRETATION

1. **Selective refit fails on this substrate first because the stale transport is not owned by
   the localized state.**
   - With every shared tensor frozen, even unrestricted training of all triplane cells (D)
     recovers 20% of what global warm start recovers, and follows about 22% of the physical
     change on the stationary receiver.
   - The post-hoc swap is consistent with this: C's shared tensors alone, on the released
     triplane, carry nearly all of C's interaction-ROI tracking (gain 0.63 vs 0.66). C's
     triplane alone carries none (0.11).
   - In 8DNA the conditional exit distribution for the tea pot's directions toward the milk pot
     is decoded by shared flows and direction cubemaps. A 24-dim per-point code cannot express
     the new distribution once those are fixed.
   - Per `docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md` §7 ("D fails while C succeeds"), this is
     a **conditional substrate-ownership issue**, not a statement about the oracle mask.
2. **The localized state is not radiometrically or computationally local either.**
   - The triplane stores a cell per axis-projected line.
   - 61% of all T3 supervision touches M95.
   - 86% of M95's support is unaffected samples.
   - 97% of the unaffected image region has an updated cell in its stencil.
   - Restricting computation to the touched samples therefore removes only 5% of the work. E
     costs the same per step as D, and converges more slowly toward the same plateau. Hence no
     cost advantage, and a measurable rise in unaffected-region error (+7.5%).
3. **The oracle itself behaves as intended.**
   - It is exact under common random numbers, deterministic and seed-stable (Jaccard 0.994).
   - It shows genuine nonlocal invalidation: 24% of the affected weight lies ≥ 0.15 from the
     mover, and 99% of occupied cells receive some affected weight.
   - Under this substrate's projections, a 95% cover is 39% of the occupied cells. Nothing here
     suggests the mask, rather than the substrate, is the limiting factor: S (proximity) and E
     (oracle) end at nearly the same quality, both at D's plateau.
4. **The `G_D_meaningful` noise clause is conservative** (interpretation, not a verdict change).
   - It compares a region-mean error difference with a per-pixel seed-repeat MAE (0.015).
     Even C's large reduction (0.039) does not clear 3×.
   - The decisive clause is the other one: D reaches 0.20× of C's reduction against the
     required 0.5×. This holds in both seeds, and is consistent with the tracking gains
     (0.22 vs 0.63) and the visual crops.
5. **Global warm start is the strong alternative here.**
   - At the same 32 768-step budget, C beats matched scratch on both affected (0.046 vs 0.050)
     and unaffected error (0.026 vs 0.034).
   - It reaches the D/E plateau in 24 s and 1.1× its own final quality in about 6.7 min.
   - That is still offline. It does not establish that neural refit can be interactive.
6. **Scope.** One asset, one rigid edit, one regime, one substrate whose localized state is a
   coarse axis-projected triplane. Not shown:
   - that selective refit fails for state with 3D-local support (dense or sparse grids,
     surface features);
   - that it fails for representations whose shared decoder does not own configuration
     transport;
   - that it fails at finer granularity, other materials or other edits.

## FINAL REPORT (batch questions)

1. **Substrate selected:** the released 8DNA teaset asset.
   - It is the only audited candidate with established T0 quality, a radiometrically verified
     indirect change and a scratch T3 ceiling on this hardware.
   - Neural Radiosity was audited at code level and deferred (unvalidated runtime and T0
     quality).
2. **High-quality static T0?** Yes, for this test: same-surface T0 error is about half the
   physical change (0.054 vs 0.112).
3. **Strong, interpretable signal?** Yes. It exceeds 3× reference and neural noise, and is
   attributed to indirect occlusion plus mover-reflected radiance (worklog 24).
4. **Affected state identified and localized?**
   - Identified: yes. The CRN oracle is exact per sample and deterministic.
   - Localized: only partly. 99% of occupied cells carry some affected weight, and 24% of the
     weight lies ≥ 0.15 from the mover.
5. **Oracle invalidation set:** M95 = 1 195 cells (9 560 parameters).
   - That is 9.7% of eligible cells and 39% of occupied cells.
   - It supports 61% of all T3 samples.
6. **Optimise only those units with the rest frozen?** Yes, mechanically. Unselected cells and
   all shared tensors stayed bit-identical (E, S, D).
7. **Affected-region transport recovered?** No.
   - E: R_aff 0.080, tracking gain 0.17, R_delta 0.93.
   - Frozen attached: 0.085 / −0.03 / 1.04.
   - C: 0.046 / 0.63 / 0.61.
8. **Unaffected quality preserved?** Not by the predeclared criterion: +7.5% and 1 changed
   frozen-stencil pixel. 97% of unaffected pixels share an updated cell.
9. **Against full reconstruction and global warm start:**
   - E stays far above both: R_aff 0.080 vs C 0.046, matched B 0.050, WL27 B 0.040.
   - It sits at D's plateau (0.077).
10. **Quality-matched cost:**
    - To Q\* (≈ the frozen attached quality): E 175 s, D 93 s, C 24 s.
    - E never reaches C's or B's quality.
    - Oracle construction (105 s) is excluded from these figures.
11. **Multi-bounce expansion?**
    - The affected weight is nonlocal: 24% beyond 0.15 and 13% beyond 0.30 from the mover,
      over 99% of occupied cells.
    - Under triplane projections, a 95% cover touches 95% of the training samples.
    - Invalidation in the cost sense approaches the whole representation.
12. **Selective neural-state reconstruction viable on this representation?** No (F1 negative).
    Causes, in order:
    1. the shared networks own the configuration-dependent transport (parameter coupling to
       shared weights, D ≪ C);
    2. projection coupling of the localized state (no unaffected preservation, no compute
       reduction);
    3. no cost advantage.
13. **Does this justify investing in a transport-dependency hierarchy?**
    - No, not on this substrate. F2 is not authorized by this result.
    - Whether a substrate with 3D-local state and a non-owning shared decoder behaves
      differently is open. It is a separate substrate decision for the user.

## UNRESOLVED QUESTION

- Would a substrate with spatially local 3D state behave differently? Candidates: dense or
  sparse grids (e.g. Neural Radiosity) or surface-attached features. Its F0 audit is
  unfinished (runtime port, scene conversion, T0 quality, T3 ceiling).
- How much of D's failure is directional?
  - The tea pot's change lies in narrow near-mirror lobes, which 8DNA decodes through shared
    flows and a shared wi cubemap. Treating the wi/xo cubemaps as additional "local" state
    would test that.
  - That would be a different state-unit definition and a new protocol. It was not run.
- C's recovery is near the historical threshold (1.22–1.28) at this budget, and it keeps
  improving. Longer warm-start budgets were not measured.
- The one changed frozen-stencil pixel per seed was not traced.

## ARCHITECTURE DECISION

**F1 NEGATIVE on the 8DNA teaset triplane substrate.**

- Oracle-guided selective refit of localized learned state does not recover the canonical
  T3 transport.
- It does not preserve unaffected output under the predeclared criterion.
- It is slower than updating all localized state, which itself recovers only a fifth of what
  global warm start recovers.
- The failure belongs to the **substrate's state ownership and parameter coupling**: the
  shared decoder owns the configuration-dependent transport, and triplane projections couple
  every cell to distant surfaces. It does not belong to the oracle mask.
- It is not evidence against selective refit for representations with 3D-local, non-shared
  transport state.
- Per the roadmap, the batch stops here. F2 is not authorized. Whether to audit a second
  substrate is a user decision.

Artifacts:
- `results/selective_refit_oracle/v1_bc085ea/` (oracle, regions, runs, renders, `metrics.json`,
  `posthoc/`, tests, chain logs);
- `results/evaluation/30/` (panels `01`–`08`, `09_key_numbers.json`, `manifest.json` with
  sources and SHA-256).
