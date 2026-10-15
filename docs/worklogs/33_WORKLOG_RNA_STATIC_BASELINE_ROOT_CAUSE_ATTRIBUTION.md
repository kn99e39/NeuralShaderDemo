# Worklog — RNA Static Baseline Quality: Root-Cause Attribution (2026-10-14 – 2026-10-15)

## SESSION QUESTION

User prompt "RNA Static Baseline Quality — Root-Cause Attribution Batch" (2026-10-14). This is a
diagnostic batch, not architecture Batch 2.

> Is the existing RNA baseline a valid reference for evaluating Surface-Owned Neural Transport, and
> if not, which specific implementation, data, optimisation or representation factor prevents it
> from being one?

**Short answer.**
- **The integration is valid.** The project's RNA pipeline reproduces upstream behaviour:
  - evaluation is numerically identical to upstream Lightning;
  - the dataset contract matches upstream;
  - the generator differences (no clamping, unlimited bounces) are measured and are too small to
    matter.

  No integration defect was found.
- **Why the teaset failed the 25 dB gate differs between the two configurations:**
  - **C1 (near-mirror Ni, α 0.05/0.1):** a representation / inductive-bias limitation of RNA on
    this material under a delta light. Model error is 89% of the total; reference noise is 11%.
    It is not capacity, spatial resolution, camera/light distribution, estimator semantics, or
    data density at 100–200 views.
  - **C2 (rough Ni, α 0.3):** mainly a measurement factor. The predeclared gate scores against
    128-spp targets whose noise floor is 29.4 dB. Against a converged reference, the same final
    weights score 26.7 dB on the four reference views (25.15 against the dataset GT). The pooled
    miss on the full val set was 0.08 dB.
- **So the existing RNA baseline is not a valid static reference on C1.** On C2 it is a valid
  reference by contract, but the Worklog-32 credibility measurement was dominated by target noise.
  Batch 1's relative comparisons remain interpretable. Against converged targets, ownership looks
  slightly more costly than Worklog 32 reported (C2 P16: −0.67 dB / logL1 1.28× vs −0.49 / 1.21×,
  on 4 views).

## RELATION TO EARLIER WORK

- **Worklog 32** stopped Batch 1 on the predeclared static-credibility gate: original RNA (A), val
  PSNR ≥ 25 dB, pooled, final weights. C1 scored 17.97 dB and C2 24.92 dB. Worklog 32 is not
  changed, and the threshold is not changed.
- **Preserved unchanged:** worklogs 17–32, every checkpoint, every dataset, and the upstream
  source (`git diff` empty at `66b5b098`). Everything new lives under
  `experiments/rna_baseline_quality/` and `results/rna_baseline_quality/`.
- **Protocol:** `experiments/rna_baseline_quality/protocol/rbq_v1.json`.
  - Committed in `120e321`, before any C1 control-C, D1, B2 or D2 result existed.
  - Amended in `d1acb10` before D2 training: it records that B2b was not triggered and
    predeclares D2.
- **Commits:** 120e321, 45fcb09, d1acb10, and this session's final commit.

## 1. HISTORICAL EVIDENCE AUDIT

Every RNA static number in the project, on a common footing (audit by a sub-agent, spot-checked).
Four metric contracts exist:
- **LVAL:** upstream Lightning `val_psnr`, the mean of per-view PSNR. The best-epoch value is in
  the checkpoint filename.
- **POOL:** worklog 32 `rso_eval`, one MSE pooled over all masked pixels of all views, final
  weights.
- **LINPEAK:** linear-HDR PSNR with peak = max(GT, pred). Used by the Rain held-out gate of
  worklog 9 and by worklog 17.
- **DMAE:** display MAE against a 32 768-spp Mitsuba reference. Used by worklogs 22–23.

| run (worklog) | asset / material | data generator | metric | value |
|---|---|---|---|---|
| Rain HQ (9, 15) | cloth, dielectric fabric (texture roughness; scarf metallic 0.4) | official Cycles generator, 256 spp | LVAL best / final | 30.83 / 30.68 |
| Rain held-out lights (9, 17) | same, 8 point lights, intensity 1000 (off-distribution) | official | LINPEAK | 30.9–37.7 (**≈ 20.5 under the LVAL tonemap**; the peak normalisation adds ~20–25 dB) |
| Einar (9) | skin / cloth | official, 256 spp | LVAL best = final | 24.79 |
| Teaset T0 (22) | Ni α 0.05/0.1 | Mitsuba bridge, 512 spp | LVAL best | 18.87 |
| Teaset T0 area-trained (23) | same, area training light | bridge | LVAL best | 19.41 |
| Teaset T3 (22 / 27) | same | bridge | LVAL best | 18.11 / 18.02 |
| Worklog 32 C1 A | same | bridge (worklog-22 H5) | POOL final / LVAL final / LVAL best | 17.97 / 18.53 / 19.40 |
| Worklog 32 C2 A | Ni α 0.3 | bridge, 128 spp | POOL final / LVAL final / LVAL best | 24.92 / 25.00 / 25.29 |

- **The "high Rain quality" comparisons mix contracts.** The 30.9–37.7 dB figures are LINPEAK. On
  the common LVAL tonemap, Rain HQ is 30.7–30.8 dB on its training-distribution validation, and
  about 20.5 dB under held-out point lights.
- **Factors that differ between Rain and the teaset:**
  - scene and geometry, including the AABB, which is ≈ 3× larger for the teaset, so triplane
    texels are coarser;
  - material: diffuse-dominant fabric vs near-mirror conductor;
  - generator: Cycles vs the Mitsuba bridge;
  - per-sample clamping;
  - bounce caps;
  - pixel filter;
  - spp;
  - pixel coverage;
  - training host.

  No single historical run isolates any of them. C2 changes only the material (plus spp and host)
  and moves A from 18.53 to 25.00 LVAL-final.
- **The 25 dB value** in `rso_v1.json` has no derivation beyond that protocol. It was a screening
  value and is kept as such.

## 2. UPSTREAM CONTRACT AUDIT (RNA `66b5b098` vs `rna_bridge.py`, `rso_*`)

| contract | class | evidence |
|---|---|---|
| H5 schema (keys, shapes, float16, world-space positions, attributes) | 1 equivalent | `rna/datasets.py:54-82` vs `rna_bridge.py:259-283`; val H5 inspected |
| `color` = full path-traced radiance, one delta directional light, E = 5, black world, coverage-premultiplied | 1 | `blender_render_dataset.py:209-271` vs `rna_bridge.py:66-121,226` |
| direction conventions (camera_dir towards viewer, normal flipped to camera side, light_dir towards light, world frame) | 1 | measured n·v < 0 on 0.6% of pixels (silhouette averaging) |
| up axis / camera distribution (+y vs +z, same hemisphere law) | 2 controlled | the network sees only world vectors and AABB-normalised positions |
| AABB normalisation | 1 | same formula; H5 AABB matches the masked-position extent |
| visibility (`diffuse_direct`): N·L > 0 and unoccluded; bridge uses the pixel centre, upstream one jittered sample | 2 | 0.1% of lit labels have hit-averaged n·l ≤ 0 |
| training lights per pixel / val one per image, uniform height ×0.999 | 1 | `generator.py:129-160` vs `rna_bridge.py:183-187,270-273` |
| loss, optimiser, schedule, batch = one view, 250 epochs | 1 (config identical to Rain HQ apart from name/paths) | `diff` of the yml files |
| material regime (near-mirror conductor) | 3 potentially quality-affecting | far outside the shipped upstream assets |
| **estimator semantics:** upstream clamps per sample (direct 20 by the generator, indirect 10, glossy/diffuse bounces 4, blur_glossy 1.0, Blackman-Harris 1.5 px, all read from the generator `.blend` files with the custom bpy, `blend_settings.json`); the bridge is unbiased and unclamped | 3, **tested in B2a below: not explanatory** | `blender_render_dataset.py:230-240`; `rbq_blend_settings.py` |
| checkpoint selection: upstream saves top-k by val; worklog 32 evaluates final weights | 2 documented | — |
| validation aggregation: upstream mean of per-view PSNR vs worklog 32 pooled | 2 documented, **quantified in control A** | `interfaces.py:262-269` vs `rso_eval.py:113-127` |

No class-4 item remains open. The two Cycles-side unknowns were settled by reading the `.blend`
settings: bounce caps, filter and indirect clamp. Partial-coverage AOV normalisation affects 2.2%
of pixels (silhouettes) and was not pursued.

## 3. CONTROLS AND RESULTS

Regimes: all PSNRs use the RNA tonemap (x/(1+x))^(1/2.2) over masked pixels. "Pooled" = one MSE
over all pixels and views; "view-mean" = upstream Lightning. Weights are final (epoch 250) unless
labelled otherwise.

### A — evaluation-path equivalence (`rbq_eval_paths.py`)

The same weights were run through upstream `NeuralSurfaceDataModule` + `trainer.validate` (U) and
through `rso_eval`'s loader (O). The blur index is pinned to the end of its schedule, because a
fresh trainer would otherwise reset it to σ = 8.

| weights | Lightning logged | U view-mean | U pooled | O pooled | max \|pred U − O\| | GT diff |
|---|---|---|---|---|---|---|
| C1 A best (epoch 98) | 19.389 | 19.389 | 18.611 | 18.611 | 0.137 | 0 |
| C1 A final | 18.529 | 18.529 | 17.969 | 17.969 | 0.178 | 0 |
| C2 A final | **24.9986** | 24.999 | 24.922 | 24.922 | 0.010 | 0 |
| Rain final | 30.679 | 30.679 | **30.008** | 30.008 | 0.003 | 0 |
| Einar final | 24.785 | 24.785 | **20.775** | 20.775 | 0.002 | 0 |

- **Per-view PSNR agreement:** U and O agree to within 0.0002 dB on every view. Masked pixel
  counts are identical. The residual per-pixel difference comes from float rounding of the AABB
  normalisation.
- **The C1 best checkpoint** gives 19.389 here vs 19.40 in its filename. At epoch 98 (step 19 800)
  the σ schedule had not yet reached its minimum.
- **The Lightning-vs-offline gap is pure aggregation** (Jensen's inequality): 0.56 dB on C1, 0.08
  dB on C2.
- **C2 A misses 25 dB under upstream's own metric too** (24.9986). It passes only with the
  validation-selected best checkpoint (25.29), which is selection on the evaluation set.
- **The gate is asset-dependent even on the official pipeline.** Rain clears it under the gate
  metric (30.0). Einar fails it (20.8), and its training log is NaN in 566 of 1000 logged steps,
  so it is not a clean control.

### B — native vs integrated

- **No genuinely comparable native control exists.** The teaset has never been rendered by the
  official Cycles generator, and its Mitsuba conductor material cannot be reproduced exactly in
  Cycles. Rain (a different scene and material) is reported as a **confounded** known-good
  control only.
- **The integration's only moving part is the data producer.** The training code is identical,
  and control A shows the evaluation is identical. B1 and B2a therefore test the producer's
  semantic differences directly.
- **B1 (`rbq_blend_settings.py`):** reads the effective upstream estimator settings, listed in
  section 2.
- **B2a (`rbq_clamped.py`):** re-renders val views 0/10/20/30 with an upstream-like estimator. It
  is a copy of the bridge tracer, with `rna_bridge.py` itself unchanged:
  - per-sample direct clamp 20 and indirect clamp 10;
  - at most 4 bounces;
  - same cameras, lights, spp and seeds.

  Not emulated: glossy blur and the pixel filter.

| pooled, views 0/10/20/30 | C1 | C2 |
|---|---|---|
| clamped targets vs converged reference | 25.90 | 29.46 |
| unclamped dataset GT vs converged reference | 27.36 | 29.41 |
| clamped vs unclamped GT | 27.97 | 43.19 |
| A vs converged reference | 18.28 | 26.70 |
| A vs clamped targets | 18.16 | 25.16 |

The upstream-like estimator moves C1 targets far less than A's error, and leaves C2 unchanged. By
the predeclared rule, B2b (retraining on clamped data) was **not run**: PSNR(clamped, ref) 25.90 >
PSNR(A, ref) 18.28.

### C — reference noise and convergence (`rbq_reference.py`, `rbq_analyze.py`)

- **Method.** Val views 0/10/20/30 were replayed from `generate_h5`'s RNG. View 0, re-rendered with
  its original seed, reproduces the stored H5 exactly (max |Δ| = 0 after float16 rounding). Each
  view was rendered again at the dataset spp with an independent seed, and once at 8× spp.
- **Noise share** = MSE(GT, ref) / MSE(A, GT).

| pooled, 4 views | C1 (512 spp, ref 4096) | C2 (128 spp, ref 1024) |
|---|---|---|
| GT vs converged ref (noise ceiling) | 27.36 | 29.41 |
| A vs GT | 17.81 | 25.15 |
| A vs converged ref | 18.28 | **26.70** |
| noise share of A's error | **0.11** | **0.375** |
| share of A's squared error in shadowed pixels (pixel fraction 0.45) | 0.31 | 0.58 |
| share in the brightest 1% (reference luminance) | 0.046 (A PSNR there 11.2) | 0.008 |
| share in the top 1% error pixels | 0.25 | 0.28 |

- **C1:** model error dominates. Lit and specular pixels carry 69% of it, and A predicts glints and
  glow where the reference is dark (`02_teaset_crops.png`).
- **C2:** 37.5% of the gate error is target noise. Shadowed (indirect-only) pixels dominate the
  remainder.

### D — training and representation

**D0 (existing curves).** Upstream val loss vs train loss:

| run | val / train loss |
|---|---|
| C1 A | 9× |
| C2 A | 7× |
| Rain | 2.3× |

- C1's val loss rises after epoch ≈ 60–100 while its train loss keeps falling (overfitting).
- **Worklog 32's P16w (16× parameters)** lowers train loss 5–10× with no val change on either
  configuration. Its per-unit triplanes are several times finer, so neither capacity nor spatial
  resolution is limiting.

**D1 (`rbq_factorial.py`).** Camera {train, val} × light {per-pixel, per-image}; 4 views per cell;
fresh renders at the dataset spp; A final.

| cell | C1 pooled (loss) | C2 pooled (loss) |
|---|---|---|
| TS stored training views (seen samples) | **19.53 (0.010)** | n/a (C2 train H5 on the server) |
| TP train cameras, new per-pixel lights | 18.05 (0.087) | 25.50 (0.008) |
| TI train cameras, per-image light | 18.76 (0.081) | 24.55 (0.011) |
| VP val cameras, per-pixel lights | 18.30 (0.091) | 25.13 (0.010) |
| VI stored val | 17.81 (0.082) | 25.15 (0.007) |

- **Any fresh sample scores like validation.** Loss is 8–9× that of the stored training samples
  (C1). Seen cameras and the light-distribution change each make < 1 dB difference; TI is lower
  in C2 only through its larger shadowed fraction.
- **The train/val gap is fitting of the specific training samples**, including their noise and
  fireflies. It is not a camera or light distribution shift.

**D2 (supervision density, predeclared in the amendment).** Arm A was trained on the first 100 of
the 200 C1 training views, with optimisation matched per step:
- 500 epochs;
- StepLR every 10 000 steps;
- the same iteration-based blur schedule;
- 50 000 steps in total.

| | A_100 | A_200 (worklog 32) |
|---|---|---|
| pooled final val PSNR | 17.66 | 17.97 |
| view-mean final | 18.32 | 18.53 |
| train / val loss | 0.0061 / 0.126 | 0.0117 / 0.102 |

- **Rule:** a drop of ≥ 1.0 dB would mark data density as material. The measured drop is −0.31 dB,
  so the result **does not support a data-density limit at 100–200 views**.
- Halving the views does increase memorisation (lower train loss, higher val loss). Single seed;
  no claim about > 200 views.

### Worklog-32 arms against converged targets (`rbq_batch1_vs_ref.py`, views 0/10/20/30)

| C2 | vs dataset GT: Δ PSNR vs A / logL1 ratio | vs converged ref: Δ PSNR / logL1 ratio |
|---|---|---|
| P1 | −0.14 / 1.01 | −0.15 / 1.01 |
| P4 | −0.22 / 1.08 | −0.31 / 1.11 |
| P16 | −0.49 / 1.21 | **−0.67 / 1.28** |
| P16w | −0.02 / 1.11 | −0.09 / 1.16 |
| REINIT | −0.50 / 1.20 | −0.67 / 1.26 |

- **C2:** target noise adds a common error floor that compresses the ownership cost. Against
  converged targets the P16 cost is somewhat larger.
- **C1:** all arms sit in the representation-limited regime (P16 +0.10, P16w +0.47 vs ref).

## 4. OBSERVATION (`results/evaluation/33/`)

- **`01_rain_view*`:** RNA error is thin and confined to silhouettes and folds. This is the
  known-good regime.
- **`01_C1_view*` / `02_teaset_crops`:**
  - C1 RNA renders a brownish glow over the whole tray.
  - It invents glints and streaks in regions the converged reference shows black (shadowed
    receiver crop).
  - It blurs the near-mirror highlight.
  - |GT − ref| is small by comparison.
- **`01_C2_view*`:**
  - Errors are low-amplitude and spread over shadowed receivers.
  - |GT − ref| (target noise) is of the same order as |RNA − ref|.
- **`03_train_val_curves`:** the teaset train/val gap and the P16w memorisation (train-only gain).
- **`04_factorial`:** flat across camera/light cells; only the seen training samples (C1 TS) stand
  out.

## 5. ROOT CAUSES CONFIRMED OR EXCLUDED

| candidate | verdict | evidence |
|---|---|---|
| evaluation/integration defect | **excluded** | A: U = O to 2e-4 dB; Lightning reproduced; H5 replay exact; contract table |
| aggregation (pooled vs view-mean) | contract difference, quantified | 0.56 dB (C1), 0.08 dB (C2); not a defect |
| checkpoint selection | contract difference | best-by-val is +0.3 to +0.9 dB but uses the evaluation set |
| reference noise | **dominant for C2's gate miss**, minor for C1 | noise share 0.375 vs 0.11; C2 A vs converged 26.70 |
| estimator semantics (upstream clamps, bounce caps) | **excluded as explanation** | B2a: target change < model error (C1), none (C2) |
| camera / light distribution shift | excluded | D1 cells within ~1 dB |
| capacity / spatial resolution | excluded | P16w: train-only gain |
| optimisation (convergence) | excluded as limit | converged by epoch ~150; C1 overfits after ~60–100 |
| supervision density (100 → 200 views) | not supported | D2: −0.31 dB < 1.0 dB rule |
| **representation / inductive bias on near-mirror conductor under a delta light** | **remaining explanation for C1 (not directly tested)** | model error 89%; glow and invented glints; out-of-sample error flat across every controlled factor |

## 6. LIMITATIONS AND CONFOUNDS

- **Control C, B2a and the converged-reference comparison** use 4 validation views per
  configuration, not all 40.
- **D2:** a single seed and a single reduction step.
- **B2a** does not emulate glossy blur or the Blackman-Harris filter.
- **No native control on the same scene:** Rain is confounded with scene, material and generator,
  and Einar has NaN training steps.
- **C2 timings and data** come from a different GPU (worklog 32); quality numbers are unaffected.
- **"Representation limit" is an attribution by exclusion.** It was not shown positively, for
  example by a model with a different angular encoding. More than 200 views and other seeds
  remain untested.

## 7. FINAL REPORT

1. **Root causes.**
   - Confirmed:
     - the C1 failure is a representation / inductive-bias limit of RNA on near-mirror conductors
       (by exclusion);
     - the C2 failure is dominated by target noise in the gate measurement, plus the aggregation
       and final-weights contract.
   - Excluded: integration and evaluation defects, estimator semantics, distribution shift,
     capacity, resolution, convergence, and data density at 100–200 views.
2. **Source-level differences:** the table in section 2. Material regime and estimator clamping
   are the only class-3 items; clamping was tested and is not explanatory.
3. **Baseline validity.**
   - The integrated RNA pipeline is a faithful RNA baseline.
   - The teaset C1 configuration is not a credible static quality reference for RNA.
   - C2 is a valid reference whose gate measurement was noise-limited.
4. **Comparisons:** sections 3–4, and `results/evaluation/33/`.
5. **Remaining limitations:** section 6.
6. **Is Batch 1's quality evidence still interpretable?** Yes, as relative evidence on identical
   data and metric, since the integration is validated.
   - C2 relative costs hold, and are slightly larger against converged targets (P16 −0.67 dB,
     logL1 1.28×).
   - C1 relative numbers sit inside a representation-limited regime and should not carry
     ownership conclusions.
   - Ownership semantics (bit-identity, reconstruction) are unaffected.
7. **Can a credible RNA static baseline be established now?**
   - Not on C1 with RNA as is.
   - On C2-like rough conductors, the evidence indicates yes, under a prospectively declared
     protocol that scores against converged validation targets: A scored 26.7 dB on the 4
     converged views.
   - This is a user decision. It must be a new protocol, not a re-scoring of worklog 32's gate.
8. **Recommendation for the next architecture investigation.**
   - Declare the static reference prospectively before Batch 2, for example:
     - the C2 material regime;
     - converged (≥ 8× spp) validation targets;
     - pooled, final weights;
     - the threshold fixed in advance.
   - Treat near-mirror C1 as an RNA representation limit to be addressed only by a different
     substrate (user decision 1 allows replacing RNA). It is not a reason to reject surface
     ownership.
   - Batch 2 itself (the physical T0 → T3 unit-level recompute recommended in worklog 32) is not
     started.

## STILL OPEN

- The user's choice of static-reference protocol and of Batch 2.
- Whether a different angular representation lifts the C1 ceiling (would confirm the
  representation attribution positively).
- Artifacts:
  - `results/rna_baseline_quality/`: controls, analysis JSONs, D2 run;
  - `results/evaluation/33/`: panels, key numbers, manifest.
