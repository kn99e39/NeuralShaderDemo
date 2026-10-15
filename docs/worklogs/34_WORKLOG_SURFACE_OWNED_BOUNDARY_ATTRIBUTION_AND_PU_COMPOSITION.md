# Worklog — Architecture Batch 2: Surface-Owned Boundary Attribution and Partition-of-Unity Composition (2026-10-15)

## SESSION QUESTION

User prompt "Architecture Batch 2 — Surface-Owned Neural Transport Boundary Fidelity" (2026-10-15):

> Is the remaining boundary degradation caused by insufficient local supervision, independent
> learned-function inconsistency, or the composition architecture itself — and can one
> evidence-supported representation change improve radiometric continuity without invalidating
> independent neural-state reconstruction?

**Short answer.**
- **Cause of the seams:** mainly independent learned-function inconsistency. At ownership cuts the
  two units' errors are decorrelated, and hard-selection composition exposes this as a jump.
  Supervision starvation is not supported.
- **One composition change helps at the boundary.** P16-PU (partition-of-unity composition in a
  7.2 mm band, all learned parameters unchanged) removes 86% of P16's excess seam discontinuity on
  36 held-out C2 views against converged references. It also lowers the physical error at seam and
  band pixels, and leaves interior and high-frequency error unchanged.
- **Independent reconstruction is preserved**, but each unit's output influence grows by its
  neighbours' band (3.2% of the influenced pixels for the tested unit).
- **It does not recover P16's overall fidelity loss.** That loss lies mostly inside units, not at
  boundaries.

## RELATION TO EARLIER WORK

- **Worklog 32:** Batch 1 established exact unit ownership and reconstruction (A, P1, P4, P16,
  P16w, REINIT).
- **Worklog 33:** the RNA baseline audit.

  Neither result is reinterpreted, and their code, configurations, checkpoints and results are
  unchanged. New code lives in `experiments/rna_surface_ownership_b2/`; outputs in
  `results/rna_surface_ownership_b2/`.
- **Protocol:** `experiments/rna_surface_ownership_b2/protocol/b2_v1.json`.
  - Committed in `3321227`, before any diagnostic ran.
  - Amendment 1 (`250134f`), written after the dev-view diagnostics and before any held-out view
    or candidate code:
    - the diagnostic verdicts;
    - the H2 dominance operationalisation (its "jump threshold" had been left unspecified);
    - the candidate and its band width.
- **Commits:** 3321227, 250134f, 1eb5dd8 (composer and tests), and this session's final commit.

## REFERENCE PROTOCOL (predeclared)

- **Primary configuration:** C2 (rough Ni α 0.3, worklog-32 data). C1 is a stress test only.
- **Dev views:** C2 val views 0, 10, 20 and 30, with the worklog-33 8× spp references. All
  diagnostics and design decisions used only these.
- **Held-out views:** the other 36 C2 val views, each re-rendered at 1024 spp (seed 950000 + view,
  `b2_reference.py`, unchanged bridge; ≈ 90 s/view). They were selected as "all remaining views"
  and used only for the final comparison.
- **Reference noise:** the dataset GT is 29.4 dB from the 8× reference (worklog 33). An 8×
  reference carries 1/8 of that variance.
- **Secondary:** all 40 views against the stored dataset GT (noisy). The historical 25 dB gate is
  neither applied nor changed.

## DIAGNOSTICS (dev views; existing A / P4 / P16 / P16w checkpoints; no training)

Definitions are fixed in the protocol and implemented in `b2_diag.py` and `b2_support.py`. All
radiance comparisons use the RNA tonemap.

**A — surface correspondence.**
- **Pair selection:** 4-neighbour pixel pairs with full coverage (α ≥ 0.999).
- **Classes:** seam (same part, different P16 unit), interior (same unit), cross-part, and
  geometric edge. Seam and interior pairs must pass a continuity test: 3D gap ≤ 3× the view's
  median interior gap, and normal angle ≤ 20°.
- **Excluded from the seam class:** pairs whose two pixels have different visibility labels
  (physical shadow edges, 21 pairs), and geometric edges (31 060 pairs).
- **Pairs kept on dev views:** 2 742 clean seam pairs and 394 484 interior pairs. The interior
  pairs are reweighted to the seam pairs' distribution of reference radiance variation |Δref|.

**D — composition attribution:** jump excess J = | |Δpred| − |Δref| |, the extra discontinuity
beyond the physical one.

| C2 dev | A | P4 | P16 | P16w |
|---|---|---|---|---|
| seam J | 0.0137 | 0.0144 | **0.0268** | 0.0257 |
| interior J (matched) | 0.0143 | 0.0147 | 0.0148 | 0.0192 |
| seam error correlation ρ(e_p, e_q) | 0.641 | 0.629 | **0.238** | 0.143 |
| interior ρ (matched) | 0.649 | 0.659 | 0.710 | 0.432 |
| seam \|e\| | 0.0247 | 0.0247 | 0.0293 | 0.0260 |

- **A shows no seam effect.** Seam and interior values are the same, so A is a valid control.
- **P16 has 1.81× the seam jump of its own interior.** Its error correlation collapses across the
  cut while staying high inside units.

**C — independent-function attribution.**
- **Cross-evaluation.** At seam pixels, the neighbour unit evaluated across the cut is nearly as
  accurate as the owner (median error 0.0242 vs 0.0210). 98.6% of band points lie inside the
  neighbour's AABB, so this is not wild extrapolation. Yet the two units disagree by as much as
  their errors (median 0.0251).

  Both are plausible fits of the same radiance with independent errors.
- **Decomposition.** Using E|e_p − e_q| ∝ σ√(2(1−ρ)), P16's seam/interior J ratio splits into a
  magnitude factor of 1.08 (H1-type) and a decorrelation factor of 1.62 (H2-type): ≈ 13% / 87% of
  the log ratio.

**B — training support** (measured on the local C1 train H5; C2 used the same cameras, light
directions and seeds).
- **Samples per unit:** 0.32–1.33 M. Of these, 1.4–4.0 × 10⁴ lie within 7.2 mm of a same-part seam
  (2.2–5.4%).
- **Direction coverage:** light directions cover 100% of 128 hemisphere bins everywhere,
  including the band. View-direction coverage is 82–96%.
- **Correlation with error (C2):** the per-unit P16/A error ratio is uncorrelated with samples
  (Spearman +0.06), samples per area (−0.02) and band samples (+0.20).
- **C1 stress:** a weak correlation with samples per area (−0.53).

**Distance profile** (P16/A error ratio vs 3D distance to the nearest same-part seam, `04`):

| distance (footprints ≈ 3.6 mm) | 0–1 | 1–2 | 2–4 | 4–8 | 8–16 | 16–32 | 32+ |
|---|---|---|---|---|---|---|---|
| C2 P16 / A | 1.22 | 1.11 | 1.10 | 1.08 | 1.07 | 1.05 | 1.01 |
| C2 P16w / A | 1.06 | 1.00 | 1.00 | 0.96 | 0.96 | 0.97 | 0.95 |

**Fixed-parameter composition test** (band half-width 2 footprints):
- P16 seam J falls from 0.0268 to 0.0144 (A 0.0137), and band error from 0.0311 to 0.0293.
- C1: J falls from 0.0672 to 0.0309 (A 0.0280), but band error rises from 0.0806 to 0.0817.

## ATTRIBUTION (final report A)

| hypothesis | classification | measured fact | interpretation |
|---|---|---|---|
| H1 reduced training support per unit | **supervision starvation: contradicted** (C2). Per the predeclared rule H1 is "supported, weak": only rule (b) holds, P16/A near/far 1.116 ≥ 1.10 on C2 dev; C1 1.077 | full light coverage in the band; error uncorrelated with sample counts and coverage; P16w (same samples, 16× capacity) flattens the near-seam rise | a small boundary-local error elevation exists (≈ 13% of the seam excess); its mechanism (e.g. triplane features supervised from one side, decoder extrapolation) is **unresolved** |
| H2 independent learned functions | **supported**; the dominant mechanism | seam error correlation 0.24 vs interior 0.71 (A: no drop); cross-evaluated neighbours disagree as much as they err | independently trained decoders reach different, equally plausible fits; their errors are independent across the cut |
| H3 composition (hard selection) | **supported** as the exposure mechanism, and removable | fixed-parameter PU removes 95% (dev) / 86% (held-out) of the seam excess and lowers seam/band error (C2) | hard selection does not create the inconsistency but exposes it as a step; an explicit continuity contract hides the pixel-scale step without retraining |
| H4 other (local frame normalisation, routing) | **contradicted** | P1 ≈ A (worklog 32); routing and dispatch tests pass | — |

## THE CANDIDATE (final report B): P16-PU

- **Hypothesis.** The seams are hard-selection exposure of decorrelated per-unit errors. An
  explicit, geometry-defined composition across ownership boundaries removes the excess
  discontinuity without changing any learned parameter.
- **Why it was chosen:** the predeclared mapping (H3 supported → explicit overlap composition)
  and the dominance analysis (H3 removes 95% of the excess as a lever).
- **Contract** (`b2_compose.py`):
  - output(x) = (1 − w) f_own(x) + w f_nbr(x), with nbr = the nearest other unit of the same part
    (dense partition samples);
  - w = clip(0.5 − (d_nbr − d_own)/(2h), 0, 0.5), with h = 7.2 mm = 2 × the median dev-view pixel
    footprint (fixed before held-out evaluation, not tuned);
  - outside the band the output is exactly the owner's.
- **What changed relative to P16:** composition only. Partition, every learned parameter
  (worklog-32 P16 `state.pt`), training, capacity and data are all unchanged.
- **Parameter ownership:** unchanged. Θ_i is unit i's triplane + MLP. There is no shared
  trainable or immutable learned state. The composition data is fixed geometry (1 M partition
  samples, ≈ 20 MB) plus h.
- **Output influence:** unit i influences S_i ∪ B_i, where B_i is the band inside its same-part
  neighbours. Outputs outside S_i ∪ B_i are guaranteed unchanged by an update of unit i.
- **Invalidation consequence.** If unit i becomes transport-dirty, the band B_i inside valid
  neighbours also reads stale f_i until i is recomputed. Neighbours need no parameter
  recomputation, but their band outputs belong to i's output-invalidation set.
- **Trade-offs:**
  - a slightly wider output footprint per unit;
  - inference overhead in this prototype: 1.08 s/view vs P16's ≈ 0.04–0.09 s, from CPU KD-tree
    queries. This is an implementation cost that per-sample precomputation could remove; it was
    not optimised;
  - band weights quantised by the 2.4 mm partition-sample spacing.

## EVIDENCE (final report C)

**Synthetic contracts** (`tests/test_b2.py`, 5/5 PASS):
- weights;
- coverage and determinism;
- outside-band output bit-identical to hard, and the hard part equal to `OwnedTriplaneSet.forward`;
- known constant functions: the hard step of 0.60 becomes a maximum step of 0.014 between adjacent
  samples, and equal constants are unchanged;
- unit re-initialisation + retraining:
  - the other units are bit-identical;
  - the changed outputs are exactly S_1 ∪ B_1;
  - outputs elsewhere are bit-identical;
  - checkpoint round trip identical.

The worklog-32 `test_rso.py` contracts are unaffected (no shared code changed).

**C2 held-out, 36 views vs 1024-spp references** (`eval/C2/eval.json`). Primary arms: A, P16 and
P16-PU. P16w and P16w-PU are shown in the bottom rows as secondary evidence.

| arm | PSNR | logL1 | hf \|e\| | seam J | interior J | seam \|e\| | interior \|e\| | band \|e\| | off-band \|e\| |
|---|---|---|---|---|---|---|---|---|---|
| A | 26.637 | 0.0145 | 0.0501 | 0.0155 | 0.0150 | 0.0255 | 0.0258 | 0.0287 | 0.0282 |
| P16 | 26.411 | 0.0171 | 0.0535 | 0.0288 | 0.0154 | 0.0312 | 0.0267 | 0.0335 | 0.0288 |
| **P16-PU** | **26.436** | 0.0171 | 0.0533 | **0.0173** | 0.0154 | **0.0284** | 0.0267 | **0.0315** | 0.0288 |
| P16w | 26.457 | 0.0169 | 0.0516 | 0.0288 | 0.0206 | 0.0292 | 0.0261 | 0.0318 | 0.0281 |
| P16w-PU | 26.486 | 0.0168 | 0.0514 | 0.0202 | 0.0205 | 0.0260 | 0.0260 | 0.0293 | 0.0281 |

- **Seam excess over A:** P16-PU removes 86% of it. Seam J is 0.0173 vs A 0.0155 (1.12×; dev
  1.05×).
- **Physical error:** seam-pixel error falls 9% and band error 6%.
- **Unchanged:** interior, off-band, logL1 and high-frequency error.
- **Whole image:** +0.025 dB. The band is a small fraction of pixels.
- **Per part** (\|e\|, tray / milk pot / tin / tea pot):
  - A: 0.0260 / 0.0341 / 0.0320 / 0.0317;
  - P16-PU: 0.0257 / 0.0355 / 0.0356 / 0.0333.

  P16's remaining gap to A is interior and per-unit, e.g. the single-unit tin at 1.11×.
- **Secondary set** (all 40 views vs dataset GT; noisy): P16 24.729 → P16-PU 24.745 (A 24.923);
  seam J 0.0342 → 0.0285 (A 0.0275).
- **C1 stress** (dev views vs 4096-spp references):
  - seam J 0.0672 → 0.0319 (A 0.0280);
  - seam \|e\| 0.0808 → 0.0827 and band \|e\| 0.0790 → 0.0799 (+1.1%);
  - whole image unchanged (18.373 → 18.374).

  On the near-mirror stress case PU removes the jump **without** improving accuracy.

**Unit reconstruction regression** (u* = unit 12, worklog-32 rule; REINIT parameters composed with
P16-PU vs P16-PU; 36 held-out views):
- **Changed pixels:** 218 126 changed, exactly the predicted S_12 ∪ B_12 (0 outside). Of these,
  211 053 are owned by unit 12 and **7 073 (3.2%) lie in neighbours' bands**.
- **Outside the influence set:** every output is bit-identical.
- **Error on the influence set:** 0.03149 vs P16-PU's 0.03170 (recovery 0.993×).
- **Parameter groups:** all 15 of unit 12's tensors (3 triplanes + 6 MLP weight/bias pairs) changed; the other 15 units are bit-identical
  (worklog-32 digests). No composition parameter exists.

**Cost:**
- no additional trainable parameters (P16: 13.9 M);
- composition data ≈ 20 MB;
- inference 1.08 s/view (prototype; A 0.049 s);
- no training.

**Exports** (`results/evaluation/34/`):
- `01` reference / A / P16 / P16-PU / P16w and error maps;
- `02` geometry-chosen seam crops;
- `03` ownership, seams and band;
- `04` distance profile;
- `05` output-influence map;
- `06` key numbers;
- manifest.

## INTERPRETATION (final report D)

- **Succeeded:**
  - causal attribution of the seams;
  - a composition contract that removes most of the excess discontinuity on held-out C2 data and
    lowers physical error at the boundary;
  - independent reconstruction intact, with an exactly accounted output-influence set.
- **Merely visual:**
  - On C1 the jump is reduced without an accuracy gain.
  - In the crops (`02`, view 20) the sharp step becomes a narrow ramp. The **broad tone mismatch
    between neighbouring units remains visible**, because it is an interior per-unit error, not a
    boundary error.
- **Physical radiance fidelity:**
  - Improved at seams and in the band (C2).
  - Essentially unchanged image-wide: P16-PU stays −0.20 dB and logL1 1.18× behind A on held-out
    data.
- **Independent reconstruction:** preserved.
- **Effective update influence:** expanded by the neighbour band (B_i), reported rather than
  hidden.
- **Structural limitation:** independently trained decoders give each unit its own error field.
  Composition can only hide their mismatch at the cut; it cannot restore the shared angular and
  material learning that A's single decoder gets from all surfaces. This is the remaining fidelity
  cost of fully independent units (worklog 32), now localised to unit interiors.

## NEXT DECISION (final report E; not implemented)

Recommended single investigation for Batch 3: **an immutable shared decoder with unit-owned
features**, holding the P16 partition and P16-PU composition fixed.
- A decoder is trained once and frozen forever; each unit owns its triplane features.
- This targets the measured remaining cost: interior, per-unit fidelity from independent decoders
  (H2).
- It must report the ownership distinction explicitly: a frozen shared function is not a
  self-contained per-unit model.
- It must test whether a unit can be fully reconstructed from its features alone, including after
  a physical edit that the frozen decoder never saw.

The physical T0 → T3 unit-level recompute (worklog 32's recommendation) remains a separate
question, after representation feasibility.

## STILL OPEN

- The mechanism behind the small near-seam error elevation (H1 residual).
- Prototype inference cost of PU (KD queries).
- Single seed and a single band width (derived, not swept).
- C2 dev/held-out only; no other scene.
- Batch 3 is not started.
