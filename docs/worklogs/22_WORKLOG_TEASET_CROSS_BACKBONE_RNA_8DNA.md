# Worklog — Teaset Cross-Backbone RNA × 8DNA and Same-State Refits (2026-09-30 – 2026-10-01)

## Session question

On the same teaset geometry-relation protocol, do RNA and 8DNA show the same
frozen-transport validity failure, different failure behaviour, or different
robustness? And for each failing architecture, does a same-state refit on T3
recover the physical transport?

**Primary classification (teaset, common-light regime, one asset):**
**SAME FROZEN-TRANSPORT FAILURE IN BOTH BACKBONES.** Frozen 8DNA and frozen RNA
both fail the locked rule at T1b and T3, both pass the relation-preserving T1
control, and both show the same signature: in the interaction ROI their output
barely moves (gain ≈ 0) while the physical image changes. RNA's reading carries
a quality caveat: it cleared the static gate by only 4%.

**Refits (case A/B, locked rule):**

| backbone / regime | refit T3 error ÷ T0 model error | gain | case |
|---|---|---|---|
| 8DNA, worklog-21 envmap (corrected references) | 1.07 | 0.70 | **A** — recovers |
| 8DNA, common light | 1.28 (limit 1.25) | 0.58 | **B** by the rule, borderline |
| RNA, common light | 1.22 (limit 1.25) | 0.56 | **A**, borderline |

## Relation to worklog 21

Worklog 21 stays the accepted 8DNA teaset result and is not edited. Its states,
released checkpoint, correspondence adapter, ROIs and envmap evidence are
reused unchanged. This batch adds a second regime and two refit controls; it
does not redesign the worklog-21 experiment.

### Correction to worklog 21's references (measurement, not verdict)

Worklog 21's path-traced references were rendered with DrJit `LoopRecord` on.
On this GPU that loses 1–5% of the radiance depending on kernel lane count and
does not reproduce across runs; with the flag off, both `path` and `prb` match
Mitsuba's CPU LLVM backend (`probe_loop_record.py`, `probe_llvm_reference.py`).

`w21_reference_correction.py` re-rendered the locked references in wavefront
mode and re-evaluated the **unchanged** historical 8DNA renders:

| | classification | failing states |
|---|---|---|
| worklog 21 as published | CROSS-MODEL GEOMETRY-CONFIGURATION FAILURE OBSERVED | T1b, T2, T3 |
| with corrected references | CROSS-MODEL GEOMETRY-CONFIGURATION FAILURE OBSERVED | T1b, T2, T3 |

- Regression of the evaluator on the historical references: max abs diff **0.0**.
- Historical/corrected image-mean ratio: 0.982–0.984 (1.6–1.8% bias).
- Corrected interaction-ROI rises: T1 −5.9% (gain 0.95), T1b +112%, T2 +60%,
  T3 +124% (gains 0.017–0.025).
- Corrected physical-signal gate: T1 5.63×, T1b 9.59×, T2 5.88×, T3 9.70×.

**Worklog 21's conclusion stands.** The bias was 20–70× smaller than the effect
it measured. Its absolute reference numbers are 1.6–1.8% low.

## IMPLEMENTATION FACT

### Why a separate regime was required

RNA's network is conditioned on a single light direction per shading sample,
its dataset generator deletes the World background, and it has no
environment-map path (`rna/models.py`, `rna/lights.py`,
`training_dataset/blender_render_dataset.py`). The worklog-21 envmap scene
therefore cannot be given to RNA without changing RNA's semantics.

Rather than approximate one renderer with another, both models run in a new
regime, `teaset_cross_backbone_common_light`: the same Mitsuba scene, released
Ni_palik materials, camera and geometry as worklog 21, with the envmap replaced
by one light and a black world. **Both models are scored against one shared
reference**, so their pixels are directly comparable — this is option A of the
batch's preference order, not a native-renderer approximation. Worklog-21
evidence is untouched.

### Lighting revisions (all before any RNA training or common-light verdict)

| rev | light | why it was replaced |
|---|---|---|
| 1–2 | delta directional, to_light `[-0.6647, 0.7071, 0.2418]`, E 5.0 | 8DNA's own seed-to-seed noise in the interaction ROI was 0.0585–0.0595, 60–93% of the physical signal: heavy-tailed specular fireflies of a delta light on near-mirror nickel (p99.9 seed difference 13.9 linear vs 0.20 under the envmap). 4× the samples cut it only 1.33× (`results/8dna_replication/neural_noise_finding.json`). |
| 3 (locked) | square area light, same direction, half-angle 15°, distance 8, irradiance 5.0 at the origin | — |

The user chose the area light; 15° was selected from a GT-and-noise-only size
scan (`light_size_scan` in the protocol). The delta-light outputs are archived
in `results/8dna_replication/superseded_delta_light/` and are not evidence.
The physical-signal gate was strengthened at the same time: a state is
interpreted only if its interaction-ROI dG exceeds 3× the **model's own** T0
seed repeat as well as 3× the reference repeat (`decision_rule.noise_precondition`).
The earlier gate checked only the reference and could not see this failure.

### Data bridge (RNA)

`rna_bridge.py` populates RNA's official H5 schema from that Mitsuba scene, with
every channel defined to the official Cycles semantics (colour, alpha,
position, normal flipped to the camera side, camera_dir, tangent, uv, binary
`diffuse_direct` visibility, per-pixel `light_dir`). RNA's network, loss and
training code are unchanged; only the data producer differs.

Focused tests (`test_rna_bridge.py`, all pass):

| test | result |
|---|---|
| estimator vs Mitsuba, same light | display MAE 0.0176 vs prb's own seed repeat 0.0172; means within 0.3% |
| visibility semantics | 0 shadowed pixels receive direct light |
| H5 schema | official names, shapes, float16, AABB attrs |
| rigid pullback | T0 canonical ≡ current; T3 mover re-hit fraction 1.000 |

### Dataset settings

512² , 200 train / 40 val views, **512 spp**, chunk 128. The spp was chosen
before any RNA training from a GT-only measurement against a 4096-spp reference
(`rna_teaset/spp_cost_quality.json`): display MAE 0.0029 at 512 and 0.0023 at
1024, both 20–40× below this batch's interaction-ROI signal, at half the cost.
The official range is 256–1024 and the project's Rain datasets used 256.

### Correspondence contract (RNA)

Per part *p* with translation *t_p*, the triplane is queried at `x − t_p`
normalised by the **training** H5 AABB; normal, camera_dir, light directions
and visibilities come from the current state. Mode `current` (official
renderer behaviour: current-world hit, current scene AABB) is kept as a
declared coordinate-mismatch diagnostic. Nearest-XYZ correspondence is not used.

### Area-light inference (RNA)

RNA is trained on directional lights (its official training distribution) and
rendered under the same 15° area light as the reference, with its own
`RectangularLight` estimator: per sub-pixel sample, 16 stratified points *y*
on the emitter rectangle (read from the loaded Mitsuba scene), direction
w = (y − x)/|y − x|, weight E = L_r·cos θ_light/(|y − x|²·pdf_area), radiance =
mean of RNA(w, vis)·E/5. 16 sub-pixel × 16 light samples per pixel. A second
light-sample seed renders T0 again for RNA's own noise floor.

One deliberate difference from the official renderer: visibility is the binary
test of each segment x → y, where the official renderer thresholds one Cycles
`diffuse_direct` pass (visible if *any* of the light is). Per-segment is what
the reference computes in the penumbra and is the per-direction meaning of the
training data's `diffuse_direct`.

Validation (`test_rna_bridge.py` test 5): the surface irradiance
Σ E·vis·max(0, n·w) from these samples against Mitsuba's own emitter sampling of
the same rectangle — mean ratio 1.0001 (T0) and 0.9999 (T3), per-pixel
difference 1.4% (Monte Carlo level), same unlit fraction (0.2%).

### Runtime facts

- References render in wavefront mode (see the correction above).
- A checkpoint written by upstream `train.py` is not readable by upstream
  `load_asset` under torch 2.8, which defaults to `weights_only=True` and
  refuses the OmegaConf hyper-parameters and Lightning callback objects the
  released files do not carry. `render_8dna_refits.py` writes a copy holding
  only the two fields `load_asset` reads; the state_dict passes through
  untouched and was verified bit-identical after loading.
- **Where each job ran.** RNA training ran on LabServer63 (RTX 3080 Ti), the
  rest on the RTX 5080. RNA training is pure PyTorch over the H5 datasets, so
  it needs no Mitsuba; 8DNA training generates path samples with Mitsuba
  throughout and had to stay on the 5080, because that server's NVIDIA
  user-space libraries are mismatched (`libnvoptix`/`libnvidia-glcore`
  570.195.03 against kernel driver 595.71.05) and Mitsuba segfaults when
  building an OptiX acceleration structure there. Dataset generation, the
  references and every evaluation render ran on the 5080, so no reference is
  split across machines and each model is scored against its own T0 in one
  place. Server environment: torch 2.8.0+cu128, pytorch_lightning 2.5.5.
- **Pre-evidence self-check (2026-09-30, before any RNA dataset of rev 3
  finished, any RNA training or any RNA inference).** Four defects were found
  and fixed; none touched an output that is evidence:
  1. The first rev-3 feature buffers still used the delta-light contract. Their
     visibility ray was infinite, so under the area light it hit the emitter
     rectangle and **every** visibility was 0; RNA would have been rendered
     entirely in its shadow branch. Replaced by the area-light inference above;
     the old buffers are archived in
     `results/8dna_replication/superseded_20260930_selfcheck/`.
  2. The chain called `ssh`/`scp`, but this machine has no Windows OpenSSH
     client; it would have died at the server launch after ~5.6 h of datasets.
     It now uses Git's OpenSSH client (same `~/.ssh/config`).
  3. An 8DNA T0 retrain killed at epoch 12 had left `last.ckpt`, which the
     chain would have taken as a finished retrain. Archived; both refits train
     from scratch.
  4. The slow static wavefront baselines (batch-1 correction) were scheduled
     before training; they now run last, off the evidence path.
  A partial T0 training dataset (38/200 views) was deleted and regenerated.
- Incidentally relevant to worklog 19: on that server's SM 8.6 GPU the upstream
  torch 2.3.1 pins install and import cleanly, supporting the reading that
  worklog 19's failure was specific to the 5080's SM 12.0 generation.

- **Operational incidents during the run (none changed an evidence output):**
  the teardown of Python processes that load both DrJit and torch crashes
  with an access violation *after* all outputs are written, which made two
  finished steps report failure (fixed: those scripts leave without teardown);
  a server launch through `nohup setsid … &` kept the PowerShell ssh call open
  until the remote job ended (fixed: `setsid -f`); a Tailscale SSH
  re-authentication check stalled the status poll for 12 min until the user
  authenticated (fixed: polls are capped at 90 s and log the link); one chain
  hand-over stopped the T0 validation dataset at 36/40 views because a wait
  condition matched an old FAIL line in the log, and that dataset was
  regenerated from scratch.

### Repository

Exact commits of the evidence outputs (from each record's `environment`, clean
trees):

| output | commit |
|---|---|
| worklog-21 reference correction | `e34d24c` |
| GT design, gate, references; frozen 8DNA common-light renders | `4fcbcac` |
| RNA datasets T0 train/val, T3 train/val | generated by chains started at `320d001` / `f621dcc` / `5cf2921`; the dataset code and protocol are identical across these (their H5 `project_commit` attribute is HEAD at write time: `26f35db`, `dc7dfc9`, `5cf2921`) |
| RNA feature buffers (area light, seeds A and B) | `dc7dfc9` |
| RNA frozen renders | `6b41314` |
| frozen comparison record | `c32454f` |
| 8DNA T0 retrain and T3 refit (training) | chain `5cf2921` |
| 8DNA refit renders, RNA refit render, full evaluation | `5cf2921` |
| evaluation exports (`results/evaluation/22/`) | `5f65d4f` |

Scaffolding commits: `e34d24c`, `a3a0cbb`, `fe0d682`, `a2107e5`, `24016a8`,
`2ed5f4c`, `4fcbcac`, `08fdd4b`, `320d001`, `26f35db`.

## MEASUREMENT

### Physical-signal gate, common-light regime (GT only, lighting rev 3)

Interaction-ROI display MAE vs T0 (seed set A); reference noise = A-vs-B
repeat; 8DNA noise = frozen 8DNA T0 seed repeat (0.0207):

| state | dG | reference noise | dG / reference | dG / 8DNA noise |
|---|---|---|---|---|
| T1 (relation preserved, control) | 0.0446 | 0.0049 | 9.1 | 2.13 |
| T1b | 0.1068 | 0.0047 | 22.7 | 5.15 |
| T2 | 0.0583 | 0.0051 | 11.4 | 2.80 |
| T3 | 0.0939 | 0.0051 | 18.4 | 4.53 |

Reference gate **passes** for T1b, T2, T3. Against 8DNA's own noise, T1b and T3
clear 3×; **T2 does not** and is reported as supporting evidence only. T1 is the
control, where a small change is expected. ROI pixel sets are identical to
worklog 21's. RNA's own noise floor (0.0060) puts all of T1b, T2 and T3 above
3× for RNA (see the frozen-response table).

### RNA training (LabServer63)

Rain HQ configuration, unchanged; T0 datasets of lighting rev 3. 250 epochs in
34 min (17:48–18:22). Validation PSNR plateaued near 18.8 dB from about epoch
70; the validation-best checkpoint (the locked selection rule) is epoch 187,
**18.87 dB**. No setting was changed after seeing it.

### Static gate G1 (T0 interaction error below the T3 physical signal)

| model | T0 interaction error | reference T3 change | pass |
|---|---|---|---|
| 8DNA (attached) | 0.0718 | 0.0939 | yes |
| RNA (canonical) | 0.0900 | 0.0939 | yes, **thin margin (4%)** |

### Frozen response, common-light regime (each model against its own T0)

Interaction ROI, display MAE; gain = ⟨dN, dG⟩/⟨dG, dG⟩ (linear). Noise
floors (T0 seed repeat): reference 0.0049, 8DNA 0.0207, RNA 0.0060.

| state | dG | 8DNA rise | 8DNA gain | 8DNA dN | dG/8DNA noise | RNA rise | RNA gain | RNA dN | dG/RNA noise |
|---|---|---|---|---|---|---|---|---|---|
| T1 (control) | 0.0442 | −1.3% | 0.70 | 0.0483 | 2.1 | +1.5% | 0.78 | 0.0428 | 7.4 |
| T1b | 0.1069 | +105% | 0.00 | 0.0015 | 5.2 | +69% | 0.00 | 0.0062 | 17.8 |
| T2 | 0.0581 | +26% | 0.00 | 0.0009 | 2.8 ✗ | +17% | −0.01 | 0.0059 | 9.7 |
| T3 | 0.0939 | +59% | 0.00 | 0.0018 | 4.5 | +38% | 0.00 | 0.0068 | 15.7 |

Rule outcome (locked rule + noise precondition):

| model / mode | classification | failing | meets criterion below noise floor |
|---|---|---|---|
| 8DNA attached (primary) | FROZEN FAILURE | T1b, T3 | T2 |
| 8DNA fixed | FROZEN FAILURE | T1b, T3 | T2 |
| 8DNA upstream (diagnostic) | failure, control confounded | T1b, T3 | T2 |
| RNA canonical (primary) | FROZEN FAILURE | T1b, T3 | — |
| RNA current (diagnostic) | FROZEN FAILURE | T1b, T3 | — |

RNA T2 is above its noise floor but its rise (+17%) is below the 25% threshold,
so it is not a failing state (its gain is still ≈ 0).

Other ROIs (rise %, gain), T1b / T2 / T3:

| ROI | 8DNA | RNA |
|---|---|---|
| mover | +27/0.21, +17/0.21, +30/0.17 | +28/0.19, +21/0.11, +36/0.10 |
| tray | +15/0.01, +7/0.00, +11/0.00 | +2/0.01, +2/0.00, +3/0.00 |
| far | +5/0.00, +1/0.00, +6/0.00 | +3/0.00, +3/0.00, +6/0.00 |

Record: `results/8dna_replication/cross_backbone/cross_backbone_frozen.json`
(RNA renders `rna_teaset/frozen/`, 8DNA `frozen/teaset_common_light_8dna/`).

### Refits (same architecture, capacity and schedule, trained on one state)

8DNA: upstream `train.py`, 30 epochs, seed 9, on T0 (`T0_retrain`, the
same-pipeline control) and on T3 (`T3_refit`); 3.6–4.0 min/epoch on the 5080.
RNA: Rain HQ configuration on the T3 datasets; 250 epochs in 34 min on
LabServer63; validation-best checkpoint **18.11 dB** (T0 run: 18.87 dB).
Recovery rule: refit error at T3 ≤ 1.25 × the same pipeline's T0 model error on
the same surfaces at T0, AND gain ≥ 0.5 (interaction ROI).

| | T0 model error | refit T3 error | ratio | gain | pipeline control (T0 retrain vs released) | recovers |
|---|---|---|---|---|---|---|
| 8DNA, w21 envmap (corrected refs) | 0.0385 | 0.0413 | 1.07 | 0.70 | 0.0385 vs 0.0341 (+13%), ok | **yes** |
| 8DNA, common light | 0.0806 | 0.1033 | 1.28 | 0.58 | 0.0806 vs 0.0718 (+12%), ok | **no** (ratio) |
| RNA, common light | 0.0900 (frozen T0 model) | 0.1094 | 1.22 | 0.56 | — (no released RNA teaset model) | **yes** |

Record: `results/8dna_replication/cross_backbone/cross_backbone.json`.
Reviewer panels: `results/evaluation/22/` (manifest with sources and SHA-256).

## OBSERVATION

- In both backbones the relation-changing states leave the interaction ROI
  almost unchanged: 8DNA dN 0.0009–0.0018 and RNA dN 0.0059–0.0068, against
  dG 0.058–0.107. The same holds for the tray and far ROIs (gain 0.00–0.01).
- Both backbones partly follow the moved part itself (mover ROI gain
  0.10–0.21) and fully follow the relation-preserving T1 motion (gain
  0.70 / 0.78, error rise −1.3% / +1.5%).
- RNA fails although its visibility input is recomputed from the current
  geometry for every light sample, so the change it does not follow is not the
  direct-shadow term alone.
- RNA's static image under this material shows strong high-frequency sparkle
  on the tray (`01_T0_GT_RNA_8DNA.png`). Its own seed noise is small (0.0060),
  so this is the network's output, not light-sampling noise.
- A same-state refit brings 8DNA back to its T0 quality under the envmap
  (ratio 1.07) but not quite under the common light (1.28); RNA's refit lands
  at 1.22. Both common-light refits have gain 0.56–0.58.

## INTERPRETATION

- **Same failure, not different robustness.** On this asset and in this
  regime, the frozen-transport failure found for 8DNA in worklog 21 is
  reproduced by a second, structurally different backbone — a relightable
  neural asset with an explicit, current-geometry visibility input — with the
  same failing states and the same signature. This makes an 8DNA-specific
  artefact an unlikely explanation for worklog 21's result.
- The shared signature is consistent with both networks having baked the
  canonical configuration's intra-asset transport (interreflection between
  parts) into their frozen parameters. An input that says "this part moved"
  (8DNA's per-part correspondence; RNA's per-part pullback plus current
  visibility) is not enough for either to produce the transport the new
  relation implies.
- **Refits.** Under the envmap, 8DNA is case A: the failure is attributable to
  the frozen state, and the architecture can represent T3. In the common-light
  regime the case boundary is not resolved: 8DNA misses the error criterion by
  0.03 of the ratio (case B by the rule) and RNA passes it by 0.03 (case A),
  each with a gain just above 0.5, from single-seed renders. Read these two as
  "partial recovery near the threshold", not as a capacity difference between
  the backbones.
- Not claimed: generality beyond this teaset, these rigid translations and
  these two regimes; any ranking of the backbones by raw error (their T0
  errors differ, and each is judged against its own T0).

## UNRESOLVED QUESTION

- Is the common-light case split real? The refit renders are single-seed and
  8DNA's seed noise in this regime is large (0.0207). A seed repeat of the refit
  renders, and a second training seed, would show whether 1.28 vs 1.22 is
  distinguishable from 1.25.
- How much does RNA's reading depend on its low static quality here (18.87 dB,
  G1 margin 4%)? A less mirror-like material would test this, but this batch
  excludes changing the material after seeing RNA.
- Which transport component carries the change neither model follows —
  interreflection between parts only, or also near-contact occlusion of the
  light? A per-bounce decomposition of the reference in the interaction ROI
  would answer this without any neural model.
- Breadth: one asset and rigid translations only. The evidence breadth the
  strategy document requires for a project-level statement is not met.
