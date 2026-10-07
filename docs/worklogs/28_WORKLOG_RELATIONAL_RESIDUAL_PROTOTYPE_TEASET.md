# Worklog — First Bounded Prototype: Frozen RNA + Sparse Current Relational State + Shared Residual (2026-10-07)

## SESSION QUESTION

On the canonical teaset failure, can the frozen persistent RNA appearance stay
unchanged while a compact, sparse **current** relational state and a shared
residual operator recover the held-out T3 transport change?

If not: did it fail because current nonlocal relation information is
unnecessary, because the chosen sparse state lacks essential transport
information, or because the persistent/dynamic lifecycle abstraction itself is
inadequate?

This is the first bounded architecture experiment that `docs/Architecture.md`
and `docs/RESEARCH_ROADMAP.md` authorise after worklogs 26/27. It is not the
end-to-end method. One architecture hypothesis was tested; nothing was
extended after the result.

**Answer, bounded to this case (stable interaction-ROI queries, common light, RTX 5080):**

- **The relational candidate does not recover T3.** All three seeds fail the
  historical rule (ratio 2.5 / 3.1 / 2.5; gain 0.63 / 0.49 / 0.52). Its error
  in the T0 → T3 *change* (display 0.086 / 0.105 / 0.086) is no smaller than
  the frozen model's (0.096) against a reference change of 0.094.
- **Current nonlocal information is not shown to be unnecessary.** The
  local-only control cannot represent the change at all (gain −0.001 to
  −0.004), as worklogs 24/25 predicted.
- **The predeclared oracle diagnostic attributes the failure to the state, not
  to the lifecycle split or the shared operator.** The same decoder, with the
  same training contract, three training configurations and frozen RNA, is fed
  a reference-derived incident-transport state. It then tracks the unseen T3
  change almost exactly (change error 0.013–0.016, gain 0.95–1.00) and
  satisfies the rule in 2 of 3 seeds.
- **This is CASE C of the batch.** This first sparse geometric relation state
  is insufficient. The higher-level lifecycle split is not falsified.

## RELATION TO EARLIER WORKLOGS

Worklogs 21–27 are unchanged. This batch reuses, without modification: the
worklog-21/22 states and correspondence; the common-light protocol; the
interaction ROI and its T0 pairing; the worklog-22 evaluation functions
(`teaset_frozen_eval.evaluate`, `cross_backbone_eval.rule`, `refit_block`); the
historical frozen T0 RNA and its renders; the worklog-24 reference
decomposition (oracle only); and the worklog-27 full T3 rebuild render
(comparison D).

The remote `main` gained three documentation commits during the batch
(`d8d71eb`, `e3a1d63`, `36ed9c5`). They authorise this bounded prototype and
add AGENTS.md's living-document synchronisation rule. They were merged as
`b1152e6` before the review and the documentation updates. No evidence run
depends on them.

## IMPLEMENTATION FACT

### Sidecar, frozen baseline intact

New folder `experiments/relational_residual_prototype/` (README lists every
file). Nothing in `external/relightable-neural-assets` (upstream `66b5b09`) or
in the historical experiment code was changed.

- **Frozen RNA.**
  - Checkpoint: the worklog-22 T0 validation-best,
    `epoch=187-val_psnr=18.87dB.ckpt`, sha256 `81063b28…`.
  - Loaded read-only, exactly as `rna_infer.py` loads it.
  - The checkpoint hash and a digest of every parameter were checked before
    and after: unchanged.
  - The per-sample base prediction was recomputed with `rna_infer`'s
    canonical-mode loop. Averaged per pixel, it reproduces the historical
    frozen renders at every stable ROI pixel: max relative difference 0 for
    T0, T0_B, T1, T2 and T3, and 5.8e-7 for T1b.
  - The candidate image is the historical frozen render plus a residual on
    the stable ROI pixels. All other pixels are untouched historical values.
- **Persistent features** are RNA's own triplane lookup at a canonical point,
  captured with a forward hook on `model.triplane_grid`. A direct call to the
  triplane module, with the sigma RNA's forward passes, gives identical
  features.

### Queries and split

- **Queries.** The sub-pixel samples (16 per pixel) of the established
  interaction ROI.
- **Stable pixel.**
  - T1b/T2/T3: all 16 samples hit the tea pot with the same canonical
    position, normal and view as at T0.
  - T1: the whole asset moved, pixels are paired to T0 as in worklog 22, and
    only the part condition applies.
  - Stable pixels per state: T0 4158, T1 4403, T1b 4146, T2 4158, T3 4134 (of
    4170 / 4412). This gives Q = 66 144 queries at T3.
- **Split.**
  - Training: T0, T1b, T2.
  - Held out: T1 and T3. A loader raises `SplitError` for any non-training
    state; this is tested.
  - Selection uses 8×8-pixel validation blocks of the training states
    (10 158 training / 2 304 validation pixels).
  - T3 never entered training, selection or any choice.

### Current relational state (the one hypothesis)

- **Probes.** K = 32 fixed hemisphere directions per query: the upper half of
  the project's existing `teaset_parts._fibonacci_dirs(64)`.
  - Local frame: the buffer normal plus a deterministic Duff et al. basis.
  - Rays are cast into the **current** geometry-only scene (four parts, no
    emitter).
- **Per-probe descriptor (20 channels).**
  - Hit, and two distance encodings.
  - Remote normal and facing cosine, in the query frame.
  - RNA triplane feature at the remote hit's canonical pull-back.
  - Remote direct visibility and cosine toward the light centre, via the
    existing `rna_bridge.visibility`.
  - The fixed probe direction.
  - Excluded: state IDs, transforms, mover distance, world positions and
    references. A name guard and the tests enforce this.
- **Operator.** A shared per-probe MLP (20→64→64), mean+max pooling, and a
  linear projection to g ∈ ℝ³². The decoder (51→128→128→3) adds the 19
  persistent/local inputs: query triplane feature, view and mean light in the
  query frame, current direct visibility and irradiance, and log of the
  frozen sample prediction. 33 187 trainable parameters.
- **Local-only control.** The probe encoder is replaced by a local MLP
  (19→96→64→32) with the same decoder: 33 763 parameters (+1.7%). It has no
  argument through which probe data could enter; this is tested.
- **Training (both branches).**
  - Loss: mean_c ((ΔL_pred − ΔL_GT)/(GT + 0.1))², with ΔL_GT = GT − frozen
    pixel.
  - Adam 1e-3, 4000 steps, 512 pixels per batch, validation every 50 steps,
    best-validation selection.
  - Seeds 0/1/2. Seed 0 is primary, and a claim needs all three.
- **Oracle diagnostic** (predeclared, run only because the candidate failed;
  never a method). The worklog-24 path-class radiances of each ROI pixel
  (9 classes × RGB, log1p) replace g. They go through an encoder shaped like
  the local-only one, with the same decoder, inputs and contract: 34 531
  parameters.

### Tests

| suite | result |
|---|---|
| `test_rrp_common.py` (deterministic probe set = existing utility, frames, local/world round trip, split, stable logic, hit index, leakage name guard) | 8/8 pass |
| `test_rrp_probes.py`: **synthetic fixture** (fixed plane, moving sphere): persistent query frame identical, relation state changes, occluder hits pull back onto the canonical sphere (≤ 1.4e-7), deterministic; **teaset**: 4134/4170 stable T0/T3 pixels, stable-query canonical position, normal, view and part identical, probe state differs T0/T3, T3 milk-pot hits pull back onto the T0 milk pot (100%), stationary-part hits are their own canonical points | 17/17 pass |
| `test_rrp_model_wsl.py` (local-only has no probe path, capacity matched, permutation-stable aggregation, probes used, state serialisation, hold-outs refused, frozen RNA hash / no grad / triplane hook independent of directions) | 7/7 pass |
| smoke runs (20-step untrained model, dirty tree allowed, never evidence) | 3 runs; they found a DrJit vector-construction error and an over-heavy remote-feature timing path, both fixed before the evidence run |

The full project regression was not run. The module is an isolated sidecar
that changes no shared production or evaluation code, and the historical
frozen outputs reproduce bit for bit (above).

### Commits and environment

| commit | content |
|---|---|
| `237bb45` | protocol `rrp_v1.json`, probes, features, branches, training, evaluation, tests. **Evidence run `v1_237bb45`** (probes, features, training): clean tree |
| `6d0ebe5`, `da318f5` | export fixes and ROI-tight review panels (presentation only); final exports written at `da318f5` |
| `1e4a0a5` | post-hoc attribution diagnostics (analysis of existing outputs) |
| `3502222` | predeclared oracle diagnostic; oracle trained at this commit |

Environment: RTX 5080 (driver 596.49), Windows 11, Mitsuba 3.5.1 / Python
3.11.9 for probes and evaluation; WSL Ubuntu-22.04, torch 2.8.0+cu128 for
RNA features and training. Outputs are in
`results/relational_residual_prototype/v1_237bb45/`; reviewer exports are in
`results/evaluation/28/`. Nothing was pushed.

## MEASUREMENT

### A–E on held-out T3 (interaction ROI; worklog-22 definitions)

Branch ratios and gains use the branch's own T0 (same-pipeline rule, primary).
Rows D use the frozen T0, as in their historical evaluation.

| | T3 abs error | ratio | gain | recovers | T0 abs error |
|---|---|---|---|---|---|
| A frozen RNA | 0.124 | 1.381 | 0.002 | no | 0.090 |
| B local-only (s0 / s1 / s2) | 0.126 / 0.119 / 0.121 | 1.34 / 1.40 / 1.45 | −0.001 / −0.004 / −0.004 | no ×3 | 0.094 / 0.085 / 0.083 |
| **C relational (s0 / s1 / s2)** | **0.087 / 0.108 / 0.083** | **2.53 / 3.08 / 2.52** | **0.63 / 0.49 / 0.52** | **no ×3** | 0.034 / 0.035 / 0.033 |
| D WL27 full T3 rebuild (val-best) | 0.119 | 1.32 | 0.43 | no | — |
| D WL22 T3 refit | 0.109 | 1.22 | 0.56 | yes | — |
| oracle diagnostic (s0 / s1 / s2) | 0.019 / 0.020 / 0.016 | 1.17 / 1.23 / 1.29 | 0.95 / 0.96 / 1.00 | yes, yes, no | 0.016 / 0.016 / 0.012 |

The secondary rule with the frozen T0 as denominator gives relational ratios
0.97 / 1.20 / 0.92 and gains 0.50 / 0.34 / 0.42. Only seed 0 meets it, at
exactly gain 0.501. This is reported, not the verdict.

### Attribution: static error vs change error (post hoc, `diagnostics.json`)

On T3 queries identical to T0's, e(T3) = [N(T0) − G(T0)] + [dN − dG].
Display-space means over stable pixels:

| | static (T0) error | change error ∣dN − dG∣ | ref. change ∣dG∣ | change error / ∣dG∣ |
|---|---|---|---|---|
| frozen RNA | 0.090 | 0.096 | 0.094 | 1.03 |
| local-only (3 seeds) | 0.083–0.094 | 0.104 | 0.094 | 1.11 |
| relational (3 seeds) | 0.032–0.034 | 0.086 / 0.105 / 0.086 | 0.094 | 0.92 / 1.12 / 0.91 |
| oracle (3 seeds) | 0.012–0.016 | 0.013–0.016 | 0.094 | 0.14–0.17 |

Relational change error / ∣dG∣ by split (frozen ≈ 1.03 everywhere):

| state | training pixels | spatial validation pixels |
|---|---|---|
| T1b (training state) | 0.30 / 0.30 / 0.31 | 0.65 / 0.69 / 0.64 |
| T2 (training state) | 0.47 / 0.49 / 0.49 | 0.94 / 1.01 / 1.07 |
| T3 (held out) | 0.92 / 1.13 / 0.93 | 0.95 / 1.07 / 0.84 |

### The state actually changes (stable queries vs T0)

| | probes changed | hit status changed | remote feature changed (both hit) | persistent query inputs max ∣Δ∣ (triplane, view, light) |
|---|---|---|---|---|
| T1b | 22.5% | 2.5% | 25.0% | 0.06 (light-sample direction mean; triplane and view identical) |
| T2 | 19.8% | 2.0% | 28.4% | 0.06 |
| T3 | 26.6% | 4.2% | 37.1% | 0.06 |
| T0_B (noise control) | 0% | 0% | 0% | 0.06 |

The light-direction mean differs between buffers only through their own
area-light samples; T0_B shows the same 0.06. All 66 144 T3 queries have at
least one changed probe. Probes hitting the moved part per query average 3.1
(T0), 3.4 (T1b), 5.2 (T2) and 7.3 (T3). **74% of T3 queries exceed the
maximum their own query saw in any training state**, by 2.8 probes on average.

### Held-out T1 (relation-preserving control)

| | T1 error (own T0) | rise | gain | correction magnitude T0 / T1 (display) |
|---|---|---|---|---|
| frozen RNA | 0.092 (0.090) | +1.5% | 0.78 | — |
| local-only s0 | 0.103 (0.093) | +10.4% | 0.45 | 0.108 / 0.115 |
| relational s0 / s1 / s2 | 0.050 / 0.048 / 0.052 (0.034–0.035) | +46% / +35% / +58% | 0.51 / 0.63 / 0.57 | s0: 0.085 / 0.090 |
| oracle s0 | 0.019 (0.017) | +13% | 0.97 | 0.086 / 0.088 |

### State size and cost (T3; structure O(Q·K), fixed K)

| | value |
|---|---|
| Q queries / K / QK probes | 66 144 / 32 / 2 116 608 |
| probe hit fraction | 0.554 (tray 608 972, milk pot 483 311, biscuit tin 22 014, tea pot 57 606) |
| dynamic state | 32 float per query = 8.47 MB; raw descriptors 169 MB (float32) |
| probe rays + remote shadow rays (Mitsuba, GPU) | 5.8 ms |
| remote persistent-feature lookup (triplane, GPU-resident) | 3.5 ms (117 ms through RNA's full forward, 154 ms with host copies) |
| per-probe encoder + aggregation | 6.4 ms |
| **state update total** | **15.7 ms** |
| residual decoder (inference) | 0.26 ms |
| local-only branch total | 0.64 ms |
| training (4000 steps) | relational 12.9–13.0 s, local-only 4.7–5.3 s, peak 1.2–1.7 GiB |

The stages run in two processes (Mitsuba on Windows, torch in WSL) and are
summed. Frozen RNA rendering and its G-buffer are not included.

## OBSERVATION

- **Change maps (`07_T3_change_maps_roi.png`).**
  - The reference T0 → T3 change on the tea pot is a set of sharp interleaved
    bright and dark bands: the moved mirror image of the milk pot, and tray
    reflection newly blocked by it.
  - Frozen RNA and local-only barely change.
  - All three relational seeds put a broad, smooth brightening over the part
    of the ROI facing the milk pot. Its location matches where the relation
    changed, but it does not reproduce the bands, and it brightens areas
    whose reference darkens.
  - The oracle reproduces the band structure almost exactly.
- **Static residual (`08_T1_vs_T0_residuals_roi.png`).**
  - At T0 (training) and T1 (held out, relation preserved), the relational
    residual reproduces the target residual GT − frozen. That residual is
    dominated by RNA's static error on this near-mirror material.
  - Its T1 map is essentially its T0 map, slightly noisier. There is no
    spurious cross-part correction where the relation is unchanged.
  - The higher T1 "rise" (+35–58%) matches the in-sample vs held-out-pixel
    gap (static error 0.026–0.029 on training pixels vs 0.053–0.058 on
    validation pixels at T0). T1's pixels are new views of the surface.
- **Full frames (`01`, `02`).** Corrections are confined to the stable ROI by
  construction, so a seam is visible at the ROI boundary. No other region is
  touched. The sparkle of RNA's static reconstruction remains everywhere
  else.

## INTERPRETATION

- **Failure-attribution route (batch section 15), in the order the prompt
  gives:**
  1. *Descriptors barely change*: **no.** 27% of probes and 37% of remote
     features change at T3, and every query is affected.
  2. *Descriptors change, but local-only and relational perform alike*:
     **no.** Relational fits the in-sample changes (change error 0.30–0.49 of
     ∣dG∣ on training pixels) where local-only cannot (≈1.1).
  3. *Training states fit, held-out T3 fails*: **yes**, the supported reading.
     The relational advantage shrinks on spatially held-out pixels and is gone
     at T3. T3 is also outside the range of relation states seen in training
     for 74% of queries.
  4. *Both fail*: no.
  5. *Dominated by RNA's static ceiling*: **partly.** The ratio verdict is
     dominated by the static term: the relational branch fits RNA's static
     error at T0 (0.090 → 0.034), so its own-T0 denominator becomes small.
     The transport-change term evaluated separately also fails, so the
     static ceiling does not explain the failure away.
- **Oracle.** The shared operator, with the persistent RNA path frozen, three
  training configurations and an identical training contract, carries an
  accurate correction to the unseen T3 configuration when its state contains
  the transport information in radiometric form. **The operator class and the
  lifecycle split are therefore not the bottleneck in this case. The
  bottleneck is what the sparse geometric state provides.** Caveat: the
  oracle state is a path-traced decomposition whose classes sum to the
  reference radiance, so it nearly contains the answer. Its success shows the
  operator is capable when the information is present; it does not show how
  hard a runtime state with that information would be to build.
- **What the probe state is missing (hypotheses, not measured here).**
  - It describes *which* directions are blocked or re-routed and *which
    persistent surface* is hit. It does not describe the **radiance arriving
    along each probe**: the remote triplane feature is RNA's position-indexed
    feature, not the radiance the remote point sends toward the query.
  - For near-mirror nickel, the change is concentrated in narrow specular
    paths that 32 uniform probes sample coarsely.
  - The operator would have to learn geometry → incident radiance from two
    weaker relation changes and extrapolate to a stronger one. It did not.
- **Not claimed.**
  - That sparse relational states cannot work in general.
  - That the result transfers to other materials, assets or relations.
  - Any claim from the oracle about runtime feasibility.
  - Any O(N·k) scene-scale claim. Only the implemented update *structure*
    is O(Q·K) with fixed K.

## FINAL REPORT (batch order)

1. **Frozen historical baseline preserved exactly?** Yes. The checkpoint hash
   and parameter digest are unchanged. The base prediction reproduces the
   historical frozen renders at every stable ROI pixel (max relative
   difference ≤ 5.8e-7, 0 in 5 of 6 buffers). Historical code, worklogs and
   results are untouched.
2. **Did the sparse relational state change T0 → T3 on the stable subset?**
   Yes. 26.6% of probes, 37.1% of remote features and 100% of queries
   change, while persistent query inputs are identical.
3. **Did the local-only residual recover T3?** No. T3 gain is ≈0 in all three
   seeds and the ratio is 1.34–1.45, the frozen behaviour.
4. **Did the relational residual recover T3?** No. Ratio 2.5–3.1 and gain
   0.49–0.63 against its own T0. Its change error equals the frozen model's.
5. **Held-out T1?** Neither branch invents a cross-part correction there.
   - Relational: T1 error 0.048–0.052, below frozen 0.092, but a rise of
     +35–58% over its own in-sample T0, so the rule's control check is not
     met. Gain 0.51–0.63 vs frozen 0.78.
   - Local-only: rise +6–16%, gain 0.45–0.49.
6. **Historical T3 recovery rule satisfied?** Not by any candidate or control
   seed. The oracle diagnostic satisfies it in 2 of 3 seeds; it is not a
   method.
7. **Does the correction follow the moved transport feature?** Only coarsely.
   It is placed where the relation changed but does not reproduce the moved
   mirror-image and occlusion bands. The oracle does.
8. **State size and cost?** 32 floats per query (8.5 MB at Q = 66 144),
   2.1 M probes. State update 15.7 ms and decoder 0.26 ms on the RTX 5080
   (two-process sum; frozen RNA rendering excluded).
9. **What failed?** Generalisation of the sparse *geometric* relation state
   to an unseen, stronger relation. It fits training configurations and
   degrades on held-out pixels and on T3.
10. **Architecture statements.**
    - *Supported*: local current state is insufficient (local-only cannot
      move at T3). A shared residual operator over frozen persistent RNA can
      carry a T3 correction when the current state carries incident-transport
      information (oracle, diagnostic only).
    - *Weakened*: that sparse geometric relation probes plus persistent remote
      features are a sufficient current state.
    - *Not falsified*: the persistent / dynamic-state / shared-operator
      lifecycle split.

## UNRESOLVED QUESTION

- Can a **runtime-constructible** current state carry radiometric incident
  transport (radiance arriving along each probe from current neighbours),
  rather than geometry only, at a cost comparable to the 15.7 ms measured
  here? This is the next architecture question. It is a second hypothesis and
  is not attempted in this batch.
- How much of the failure is training-relation coverage? Two changed training
  configurations, both weaker than T3, versus the state's content. Separating
  them needs more training configurations; not done.
- Probe angular resolution for near-mirror transport (K = 32 uniform) versus
  the missing radiance channel: not separated.
- Generality across materials, assets and relation types: untouched.

## ARCHITECTURE DECISION

**CASE C.** This first sparse relation state (K = 32 fixed geometric probes
with persistent remote features) is **insufficient** for the canonical T3
change. The lifecycle split (frozen persistent appearance + current
configuration-dependent state + shared operator) is **kept**: the oracle shows
the operator path works when the state carries incident-transport
information, and nothing here contradicts the split. The living documents
record that the next representation question is a current state that carries
**radiometric incident transport**, not only geometric relations. No second
architecture was implemented in this batch.
