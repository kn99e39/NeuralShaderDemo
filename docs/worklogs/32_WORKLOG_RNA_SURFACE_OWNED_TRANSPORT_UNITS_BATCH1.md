# Worklog — RNA Surface-Owned Transport Units, Batch 1 (2026-10-13 – 2026-10-14)

## SESSION QUESTION

Batch 1 of a minimum five-batch architecture investigation (user prompts "RNA-Based Representation
Ownership — Batch 1" and the follow-up "Research Implementation Batch 1 — RNA-Based Surface-Owned
Neural Transport Representation", 2026-10-13):

> Can an RNA-style high-quality neural transport representation be divided into independently
> owned Surface Transport Units, with each unit supporting complete neural-state reconstruction
> while preserving the other units, without unacceptable static-GI fidelity loss?

**Short answer: conditional.**
- **Ownership: yes, exactly.**
  - Parameters: every unit's parameters are its own.
  - Reconstruction: re-initialising one unit and retraining it recovers its prior quality.
  - Preservation: the other 15 units are bit-identical, and so is every pixel they render.
- **Static-GI fidelity: not adjudicable under the predeclared protocol.** Original RNA (A) failed
  the static-credibility gate (val PSNR ≥ 25 dB) in both configurations:
  - C1 (canonical, near-mirror nickel): 17.97 dB;
  - C2 (rough nickel α = 0.3): **24.92 dB, 0.08 dB short**.
  The follow-up's first stop condition ("original RNA fails to establish a credible static radiance
  baseline") therefore applies. Descriptively, matched-capacity P16 numerically meets every fidelity
  screening value against A, but it shows two consistent costs:
  - a 1.15× HDR log error (C2) and 1.11× (C1);
  - visible discontinuities at ownership seams (+16% / +39% excess seam discontinuity).
  Full-size units (P16w, 16× A's parameters) do not remove either cost.

This batch implements no hierarchy, dirty detection or Batch 2.

## RELATION TO EARLIER WORK AND LIVING DOCUMENTS

- **Worklog 30** (intra-object triplane-cell masking in one 8DNA model) is preserved unchanged.
  Its negative result concerns a different hypothesis: a feature subset inside a jointly owned
  model. Here every unit owns a complete RNA model and is recomputed whole.
- **Worklog 31** tested explicit (non-neural) surface radiance units and whole-object oracle refit.
  Its outcome is now recorded in the living documents, which had not been updated after it.
- **Living documents rewritten for the user's ten decisions of 2026-10-13:**
  - RNA is the initial baseline;
  - ownership finer than objects is allowed;
  - Surface-Owned Transport Units are the active hypothesis;
  - an invalidated unit's complete state may be recomputed;
  - object-level recompute is the coarse comparison;
  - at least 5 architecture batches come before a broad viability judgment;
  - a single negative result does not falsify the hypothesis;
  - 2007–2013 dynamic GI is the priority historical reference;
  - the 1997 line-space hierarchy is not the starting point;
  - efficiency and scaling come after feasibility.

  Files touched: `docs/RESEARCH_CENTRIC_TOPIC.md`, `docs/Architecture.md`, `docs/RESEARCH_ROADMAP.md`,
  `docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md`, and the top section of `AGENTS.md`. The previous
  object-level-only architecture is archived at
  `docs/history/ARCHITECTURE_OBJECT_LEVEL_SELECTION_2026-10-12.md`.

## BASELINE AUDIT (RNA upstream `66b5b098`, unmodified)

| component | where | tied to the whole asset? | adapted per unit? |
|---|---|---|---|
| triplane features 3×8×512², Gaussian blur σ 8 → 1 over 20 k iterations | `rna.models.NeuralSurfaceTriPlane.triplane_grid` | spans the dataset AABB | yes: one per unit, framed on the unit AABB (+2% margin) |
| MLP 17 → 512 ×4 → 6 (lit / shadowed RGB) | `rgb_network` | one decoder for every surface point | yes: one per unit (no shared trainable decoder) |
| position normalisation | dataset H5 `aabb_min/max` | asset AABB | owner id appended as a 4th position channel; units renormalise to their own AABB |
| loss / optimiser | `NeuralSurfaceModule`: log-L2, visibility weight 10, Adam 1e-3, StepLR(50, 0.5), 250 epochs, one view per step | — | reused unchanged (Lightning module subclassed, only `self.model` replaced) |
| ground truth | worklog-22 `rna_bridge.generate_h5` (Mitsuba), full-scene path tracing incl. cross-part GI | scene | unchanged (C2 only patches the material α) |
| renderer dispatch | per-pixel forward of the model | — | `OwnedTriplaneSet.forward` routes each pixel to its owner (selection, never a sum) |
| checkpoint | `state_dict`; eval-time blur `last_sigma` is not saved (defaults to `min_sigma`) | — | same; per-unit sub-dicts `units.<k>.*` |

The upstream source was not modified. The integration has two known quirks, both handled outside
upstream:
- `PlaneFeatureGrid` squeezes a single-sample input. A unit owning one pixel evaluates a
  duplicate and keeps one output (`089bbff`).
- Upstream writes `yz_grid.exr` into the working directory, so tests run in their output directory
  (`375bf89`).

## THE CANDIDATE: UNIT CONTRACT AS IMPLEMENTED

U_i = (S_i, Θ_i):

- **S_i (surface domain).**
  - Partitions: P1 (whole asset), P4 (one unit per part), P16 (k-means per part; tray 9, tea pot 4,
    milk pot 2, tin 1).
  - The k-means runs on 1 M area-uniform surface samples (seed 0); see `rso_partition.py`.
  - Owner of any surface point: its exact part, by point-to-triangle distance in contact zones;
    within that part, the nearest dense sample's unit.
  - Coverage: every masked pixel has exactly one owner (no overlap).
  - Mapping stability: the same function serves training, evaluation and REINIT, so the mapping is
    deterministic given the stored partition. Identical partition files (sha256) were used in C1
    and C2.
  - Known approximation: silhouette pixels in the RNA H5 carry hit-averaged positions up to 0.33
    off-surface. They are owned by the nearest part, so ownership is ambiguous only at those
    silhouette pixels.
- **Θ_i (learned state).** A complete upstream `NeuralSurfaceTriPlane` per unit: its own triplane
  and its own MLP. **Nothing trainable is shared.** The only shared items are fixed buffers (unit
  AABBs, dataset AABB) and fixed code.
- **Composition.** Hard selection by owner id: no blending, no sum, no double counting.
- **Update boundary.** A unit update sets `requires_grad` on that unit's module only. The REINIT
  test re-initialises unit u* with the upstream initialiser and trains it under the same 250-epoch
  contract.
- **Transport semantics.** Unchanged from RNA:
  - outgoing radiance of the full scene (all cross-object GI) at each surface point, under the
    dataset's light distribution;
  - direct visibility selects the lit or shadowed output.

  Ownership is storage/update ownership, not physical locality.

Capacity (declared in `protocol/rso_v1.json` before any training):
- **Matched:** each unit keeps A's texel size (resolution scaled to the unit's largest extent,
  clamped to [64, 512]), and MLP width is 512/√K, so the summed MLP parameters equal A's.
- **P16w:** every unit at A's full size; capacity diagnostic only.

## IMPLEMENTATION FACT

| item | value |
|---|---|
| code | `experiments/rna_surface_ownership/`: `rso_partition.py`, `rso_model.py`, `rso_train.py`, `rso_eval.py`, `rso_unit_star.py`, `rso_make_config.py`, `rso_data_c2.py`, `rso_server_chain.sh`, `rso_server_c2.sh`; this session added `rso_diag.py`, `rso_curves.py`, `rso_exports.py` |
| protocol | `experiments/rna_surface_ownership/protocol/rso_v1.json` (written before any training; no amendments) |
| commits | 5ea531c … 858df4d (implementation, chain), c0b70a8 / 375bf89 / ca3cde3 / 5a84731 (C2 and server), faee583 (fix below), this session's commit (diagnostics, exports, docs) |
| data C1 | worklog-22 H5 `teaset_T0_{train,val}.h5` (200 / 40 views, 512², 512 spp) |
| data C2 | same geometry/cameras/lights/seeds, all parts roughconductor Ni α = 0.3 (Beckmann), 128 spp; val generated on the 5080, train on LabServer63; val sha256 `5f2506db…` identical on both machines |
| seeds | 0 for every arm; partition k-means seed 0 |
| environment | WSL Ubuntu-22.04, torch 2.8.0+cu128, Lightning 2.5.5; C1 on RTX 5080, C2 on RTX 3080 Ti (LabServer63) |
| run dirs | `results/rna_surface_ownership/v1_858df4d/` (C1), `results/rna_surface_ownership/v1_858df4d_c2/` (C2, copied from the server without H5 / Lightning checkpoints), `results/rna_surface_ownership/diag_wl32/` |
| tests | `tests/test_rso.py`: partition (coverage, part consistency), routing / dispatch (owner output = unit alone, perturbing one unit leaves other pixels identical), single-unit equivalence with upstream, isolation + checkpoint round trip — **4/4 PASS in both chain runs**; this session added composition reproducibility (repeat, reload and shuffled pixel order bit-identical) — **5/5 PASS** (`results/rna_surface_ownership/diag_wl32/tests/tests.json`) |

Execution notes (affect timing only, not quality):
- **Placement deviated from the protocol's plan** (C1 on the server) at the user's direction:
  C1 ran on the 5080, and C2 generation and training ran on the server.
- **C1 A shared the 5080** with C2 data generation for about 25 min (13:42–14:07).
- **P1 and P16w** failed at their first run with `specs.json` not serialisable: an OmegaConf
  `ListConfig` on the K = 1 / unmatched path, before any training. Fixed in `faee583` (values
  unchanged) and resumed. Those two arms ran from `faee583`; the other arms from 858df4d / 5a84731.
- **C1 overnight interruption.** C1 was stopped overnight at the user's request (no checkpoint
  existed yet), and its P1 restarted on 2026-10-14 at 10:18.
- **C1 P16w epoch-time anomaly.** It ran at 107 s/epoch versus 36.6 s on the slower server GPU.
  This was not diagnosed (possibly contention while the desktop was in use), so treat that
  timing as unreliable.
- **Cross-configuration timings are not comparable** (different GPUs).

## MEASUREMENT

All numbers are on the validation set (40 views), using the final epoch-250 weights (no selection
on the validation metric).
- **PSNR:** RNA's own tonemap x/(1+x), γ 2.2, pooled over all masked pixels. This is the protocol
  metric, written by `rso_eval`.
- **dMAE:** display MAE of clamp(x^(1/2.2)).
- **Boundary:** pixels next to a different P16 owner. This is the protocol's seam metric, and it
  includes cross-part geometric edges.
- **Diagnostic columns:** from `rso_diag.py`, whose definitions were fixed in its docstring before
  any of them were computed.
  - logL1 = mean |log(1+p) − log(1+g)|.
  - hf = the 10% of pixels with the largest GT gradient per view.
  - seam disc = over 4-neighbour pairs straddling an *intra-part* ownership cut,
    mean | |Δpred| − |ΔGT| |, using the same pairs for every arm. A, P1 and P4 have no such cut,
    so they are controls.

### Static-credibility gate (predeclared: A val PSNR ≥ 25 dB)

| configuration | A val PSNR (protocol) | verdict | Lightning best epoch val PSNR (validation-selected, NOT the gate metric) |
|---|---|---|---|
| C1 canonical (α 0.05 / 0.1) | 17.97 dB | **FAIL** (protocol had predicted failure from worklog 22's 18.87 dB) | 19.40 (epoch 98) |
| C2 rough (α 0.3) | 24.92 dB | **FAIL by 0.08 dB** | 25.29 (epoch 144); final-epoch Lightning value 25.00 |

The Lightning value is per-step validation, and the best epoch is selected on the validation set
itself. The protocol excludes both by design, so the threshold was not reinterpreted. Per protocol,
ownership fidelity is screened only "on a configuration whose A passes static credibility", so the
fidelity verdict is **not adjudicated**. The tables below are **descriptive, regime-labelled**.

### C2 rough (RTX 3080 Ti) — descriptive

| arm | params | PSNR | Δ vs A | dMAE (×A) | boundary dMAE (×A) | logL1 (×A) | relMSE | hf dMAE (×A) | seam disc (×A) | train s | s/epoch | peak GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 7.35 M | 24.92 | — | 0.0411 | 0.0661 | 0.0177 | 0.111 | 0.0770 | 0.0486 | 2004 | 7.2 | 3.24 |
| P1 | 7.35 M | 24.91 | −0.02 | 1.00 | 1.00 | 0.99 | 0.125 | 1.00 | 0.99 | 2801 | 7.4 | 3.30 |
| P4 | 10.11 M | 24.86 | −0.07 | 0.99 | 1.02 | 1.07 | 0.118 | 1.03 | 1.04 | 4411 | 13.4 | 2.34 |
| **P16** | 13.91 M | **24.73** | **−0.19** | **1.02** | **1.05** | **1.15** | 0.136 | 1.06 | **1.16** | 10054 | 34.3 | 2.01 |
| P16w | 117.67 M | 24.88 | −0.04 | 0.99 | 1.02 | 1.12 | 0.096 | 1.02 | 1.14 | 13111 | 36.6 | 5.97 |
| REINIT | 13.91 M | 24.72 | −0.20 | 1.02 | 1.05 | 1.15 | 0.134 | 1.05 | 1.16 | 6495 | 21.9 | 1.25 |

Fidelity screening values, descriptive only:
- P16 ≥ A − 0.5 dB: −0.19, met;
- dMAE ≤ 1.10× A: 1.022, met;
- boundary ratio ≤ 1.15: 1.045, met.

### C1 canonical (RTX 5080) — descriptive, non-credible baseline

| arm | params | PSNR | Δ vs A | dMAE (×A) | boundary (×A) | logL1 (×A) | relMSE | hf dMAE (×A) | seam disc (×A) | train s | s/epoch | peak GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 7.35 M | 17.97 | — | 0.0886 | 0.1216 | 0.0358 | 11.1 | 0.1862 | 0.0636 | 2271* | 4.4 | 3.24 |
| P1 | 7.35 M | 18.22 | +0.25 | 0.91 | 0.96 | 0.97 | 16.1 | 0.99 | 0.97 | 1827 | 4.4 | 3.30 |
| P4 | 10.11 M | 17.66 | −0.31 | 1.00 | 0.99 | 1.10 | 22.6 | 1.03 | 1.01 | 2569 | 6.1 | 2.34 |
| **P16** | 13.91 M | **17.89** | −0.08 | 0.99 | 0.99 | **1.11** | 24.9 | 1.05 | **1.39** | 5299 | 16.7 | 2.01 |
| P16w | 117.67 M | 18.21 | +0.24 | 0.83 | 0.91 | 1.08 | 5.2 | 1.05 | 1.44 | 24835† | 107.1† | 5.97 |
| REINIT | 13.91 M | 17.90 | −0.07 | 0.98 | 0.99 | 1.11 | 31.9 | 1.05 | 1.38 | 3424 | 10.9 | 1.26 |

\* shared GPU for ~25 min. † anomalous, see execution notes.

relMSE = mean((p − g)²/(g² + 0.01)) is dominated by a few near-black GT pixels with bright
predictions. This is especially true in C1 (near-mirror highlights), where it ranks arms
inconsistently with every other metric. It is reported as declared, but logL1 is the HDR metric
interpreted here.

### Per-part and per-unit (display MAE, P16 vs A)

- **C2 per part.** P16/A ratios:
  - tray: 0.0385 → 0.0383 (1.00);
  - milk pot: 1.04;
  - tin: **1.10** (a single P16 unit, so this is not a seam effect);
  - tea pot: 1.05.
- **C2 per P16 unit.** Ratios to A range 0.96–1.10.
- **C1 per P16 unit.** Ratios range 0.72–1.15.
- **Unit surface area vs difficulty.** Spearman correlation of unit area with the P16/A ratio is
  0.19 (C2) and 0.56 (C1); for P16w, −0.16 and 0.23. There is no consistent relation.

### Independent unit reconstruction (REINIT; u* = unit 12, tea pot, most training pixels)

| property | C1 | C2 |
|---|---|---|
| 1. complete owned state re-initialised and retrained (250 epochs, upstream contract) | yes | yes |
| 2. parameter preservation: 15 frozen units bit-identical (`train.json` digests) | **yes** | **yes** |
| 3. output preservation: val pixels owned by frozen units bit-identical to P16 | **2 380 895 / 2 380 895** | **2 379 729 / 2 379 729** |
| u* pixels changed | 233 743 / 233 743 | 233 593 / 233 593 |
| 4. u* display MAE after retraining vs its P16 value (criterion ≤ 1.10×) | 0.0931 vs 0.0981 (0.95×) **PASS** | 0.04198 vs 0.04209 (1.00×) **PASS** |
| 5. composition / boundary coherence after the update: seam disc REINIT vs P16 | 0.0878 vs 0.0884 | 0.0565 vs 0.0566 |
| 6. unintended change in unaffected domains | none (item 3) | none (item 3) |
| radiometric validity of the preserved units | **not tested**: the intervention is synthetic (static targets); no geometry changed | same |
| training cost vs full P16 | 3424 s vs 5299 s (0.65×) | 6495 s vs 10054 s (0.65×) |

REINIT's per-epoch cost is ~65% of full P16, not ~1/16. Every training step still evaluates
every pixel of the view through all units (frozen ones included), and data handling dominates.
This is an implementation property, not a measured property of ownership. No speedup is claimed.

### Model loading and inference (RTX 5080, `rso_diag`)

| | A | P1 | P4 | P16 | P16w |
|---|---|---|---|---|---|
| load (state.pt → eval-ready) | 0.20 s | 0.14 s | 0.20 s | 0.33 s | 2.29 s |
| inference per 512² view | 25 ms | 15 ms | 16 ms | 41 ms | 45 ms |

P16 runs one forward per unit per view: 16 small kernel batches.

### Convergence (`05_val_psnr_curves.png`)

- All arms are converged by epoch ~150–200 in both configurations.
- Partitioned arms converge more smoothly, but P16 plateaus lowest (C2 last-epoch Lightning
  value 24.79 vs A 25.00).
- REINIT starts high (the frozen 15 units already carry the scene) and recovers u* within ~50
  epochs.

## OBSERVATION (reviewer exports, `results/evaluation/32/`)

- **`01_C{1,2}_views.png` (GT / A / P16 / P16w and error maps).** At full-image scale P16 is
  hard to tell from A. C1's near-mirror highlights are blurred for every arm (baseline limitation).
- **`02_ownership_map_P16.png`.** The 16 units on four validation views, with intra-part seams
  drawn as black lines and cross-part edges as white lines. The tea-pot seams run across the
  highlighted body.
- **`03_C{1,2}_seam_crops.png`.** 96² crops at the window with the most intra-part seam pixels
  (chosen from geometry only). P16 and P16w show **visible shading steps along the ownership cuts**
  on the tea pot (clearest in view 20). A and P4 (no cut there) are continuous.
- **`04_C{1,2}_reinit_output_preservation.png`.** |REINIT − P16| is non-zero only inside u*; the
  "any bit changed" mask coincides with u*'s pixels.

## INTERPRETATION AND ATTRIBUTION

1. **Ownership isolation succeeds, as a representation contract, and is exact.** Independent
   complete RNA units give bitwise parameter and output preservation and full-state
   reconstruction of a unit to its prior quality. The trainable state has no hidden coupling: no
   shared trainable decoder exists, and frozen units (`requires_grad` off) receive no gradient,
   so Adam leaves them untouched (verified by parameter digests).
2. **The baseline, not the candidate, stops the fidelity verdict.**
   - C1: RNA cannot fit the near-mirror canonical teaset (known since worklog 22).
   - C2 (chosen in advance to relieve exactly that) got within 0.08 dB of the bar.
   This is an **RNA baseline limitation on this asset**, not an ownership failure.
3. **Local frames are harmless.** P1 (one unit on its own AABB) equals A in C2 and is slightly
   better in C1.
4. **Fully independent units have a fidelity cost.** It is not explained by capacity. In C2:
   - P16's HDR log error is 1.15× A and the seam discontinuity 1.16×;
   - the predeclared display metrics barely move (1.02×).
   - Giving every unit A's full size (P16w, 16× parameters) leaves logL1 at 1.12× and seams at
     1.14×.

   The candidate explanation is the loss of the **shared angular/material learning** that A's
   single MLP gets from all surfaces. With hard selection, each unit's decoder learns its view-
   and light-dependence only from its own pixels and is extrapolated independently up to the seam.
   The single-unit tin (1.10× per part) supports this, but it was not isolated by an
   experiment, so the explanation is a hypothesis.
5. **Seam discontinuity is inherent to hard-selection composition of independently trained
   units.** It shows up visually, and more in the glossy C1 (+39%) than in C2 (+16%). The protocol's
   boundary metric (pixel error at boundaries, 1.05×) understates it, because the error is a
   relative step between neighbours, not a large per-pixel error.
6. **Capacity accounting.** Matched P16 holds 1.89× A's parameters (13.9 M vs 7.35 M), because
   per-unit AABBs overlap in space while keeping A's texel size. P4 holds 1.38×. P16 units hold
   0.44–1.35 M parameters each, dominated by the triplane (the MLP is width 128). The comparison is
   therefore **capacity-confounded upward**: the candidate had more parameters than A, not fewer.
7. **Partition size vs difficulty.** No consistent relation between unit area and relative error
   (Spearman −0.16 … 0.56 across configurations and arms).

| attribution class | finding |
|---|---|
| RNA baseline limitation | **yes**: gate failed in both configurations (stop condition) |
| candidate representation limitation | likely: +12–15% HDR log error and seam steps persist with 16× capacity |
| ownership isolation failure | **no**: exact parameter and output preservation |
| optimisation / capacity failure | not the main factor (P16w); per-unit decoder width may still matter (tin) |
| surface partition failure | no: coverage complete, owners stable, identical across runs; silhouette ambiguity only |
| supervision / correspondence | no change from RNA; full-scene targets kept |
| rendering / compositing failure | no failure; hard selection produces the seam steps (item 5) |

## FINAL REPORT

### Architecture
- **Representation.** K complete, independently trainable RNA surface models. Each owns a surface
  domain from a fixed geometric partition, its own triplane framed on its domain AABB, and its own
  MLP.
- **Learned-state ownership boundary.** One `NeuralSurfaceTriPlane` module per unit. Shared state
  is non-trainable only (AABB buffers, code).
- **Differences from RNA.** The partition, the owner channel, per-unit framing, and hard-selection
  dispatch. Loss, optimiser, schedule, targets and visibility treatment are unchanged.
- **Correspondence/composition contract.** One owner per surface sample, the same owner function in
  training and evaluation, and selection with no blending.

### Evidence
- **Static baseline:** fails the predeclared credibility gate (17.97 / 24.92 dB).
- **Candidate static fidelity (C2, descriptive):** −0.19 dB, 1.02× dMAE, 1.05× boundary,
  1.15× logL1, 1.16× seam discontinuity.
- **Unit reconstruction (both configurations):**
  - u* recovers to 0.95× / 1.00× of its P16 error;
  - 15 units' parameters are bit-identical;
  - 100% of frozen-unit pixels are bit-identical.
- **Tests:** 4/4 pass in both chain runs; 5/5 with the added composition-reproducibility test.
- **Diagnostics:** runtime and memory tables above, with timing caveats.
- **Exports:** `results/evaluation/32/` (manifest with sha256).

### Attribution
- **Succeeded:**
  - exact ownership, isolation and complete unit reconstruction;
  - composition correctness;
  - capacity accounting.
- **Failed:** the baseline-credibility precondition (stop condition).
- **Unconfirmed:**
  - static fidelity under a credible baseline;
  - radiometric recovery after a real geometry edit;
  - whether transport-dirty units are sparse at this granularity;
  - any cost advantage.
- **Inherent to this candidate:** seam steps under hard selection; loss of cross-unit sharing of
  angular learning.
- **Possibly implementation- or capacity-dependent:**
  - per-unit decoder width (tin);
  - REINIT's training cost (all-pixel forward);
  - the C1 P16w timing.

### Next Architecture Decision (recommendation; not implemented)

Batch 2 should not start until the user decides how to meet the static-credibility precondition,
since Batch 1 stopped on it. Options, each to be predeclared:
- (a) a scene/material configuration where original RNA clears the existing 25 dB bar;
- (b) a prospectively revised credibility criterion, documented as a new protocol and not applied
  retroactively to Batch 1.

On a credible baseline, the evidence points to two candidate questions:
1. **Recommended: radiometric validity under a physically grounded edit with the current
   independent-unit candidate.** On T0 → T3: refit the oracle transport-dirty P16 units versus
   all units versus whole objects (coarse comparison).
   - Why: ownership semantics are now exact, so the decisive unknowns are whether a recomputed
     unit recovers real GI change and whether D_T is sparse at unit granularity.
   - Worklog 31 found D_T = all objects at object granularity.
2. **Alternative: a representation variant that restores shared angular learning without
   trainable sharing.** For example: a decoder pretrained once, then frozen forever, plus per-unit
   owned features. Or overlapping units with explicit partition-of-unity blending to remove seam
   steps. Either addresses the 1.12–1.15× HDR and seam costs, but adds a declared frozen shared
   component or an explicit owner composition.

Do not implement either without a new decision.

## STILL OPEN

- The user's decision on the baseline-credibility route and on Batch 2.
- Cause of the C1 P16w epoch-time anomaly (timing only).
- Lightning checkpoints and H5 files for C2 remain on LabServer63
  (`~/nsd_w32/results/rna_surface_ownership/v1_858df4d_c2/`); `state.pt` files were copied locally.
