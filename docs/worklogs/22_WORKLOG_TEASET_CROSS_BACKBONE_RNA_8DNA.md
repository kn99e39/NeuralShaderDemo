# Worklog — Teaset Cross-Backbone RNA × 8DNA and Same-State Refits (2026-09-30)

> **DRAFT — the evidence run is in progress.** Sections marked *(pending)* have
> no measurement yet. Nothing here is a verdict until this line is removed.

## Session question

On the same teaset geometry-relation protocol, do RNA and 8DNA show the same
frozen-transport validity failure, different failure behaviour, or different
robustness? And for each failing architecture, does a same-state refit on T3
recover the physical transport?

**Primary classification:** *(pending)*

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

### Repository

- Experiment commit for this batch: *(pending — recorded by the chain)*
- Scaffolding commits: `e34d24c`, `a3a0cbb`, `fe0d682`, `a2107e5`, `24016a8`.

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
worklog 21's. RNA's noise ratio is measured once RNA exists.

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

### Refits *(pending)*

## OBSERVATION *(pending)*

## INTERPRETATION *(pending)*

## UNRESOLVED QUESTION *(pending)*
