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
by one directional light (to_light `[-0.6647, 0.7071, 0.2418]`, irradiance 5.0,
black world). **Both models are scored against one shared reference**, so their
pixels are directly comparable — this is option A of the batch's preference
order, not a native-renderer approximation. Worklog-21 evidence is untouched.

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
normalised by the **training** H5 AABB; normal, camera_dir, light_dir and the
binary visibility come from the current state. Mode `current` (official
renderer behaviour: current-world hit, current scene AABB) is kept as a
declared coordinate-mismatch diagnostic. Nearest-XYZ correspondence is not used.

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
- Incidentally relevant to worklog 19: on that server's SM 8.6 GPU the upstream
  torch 2.3.1 pins install and import cleanly, supporting the reading that
  worklog 19's failure was specific to the 5080's SM 12.0 generation.

### Repository

- Experiment commit for this batch: *(pending — recorded by the chain)*
- Scaffolding commits: `e34d24c`, `a3a0cbb`, `fe0d682`, `a2107e5`, `24016a8`.

## MEASUREMENT

### Physical-signal gate, common-light regime (GT only)

Interaction-ROI display MAE vs T0 (seed set A), over A-vs-B repeat noise:

| state | change | noise | ratio |
|---|---|---|---|
| T1 (relation preserved) | 0.0658 | 0.0329 | 5.33 |
| T1b | 0.1024 | 0.0302 | 10.04 |
| T2 | 0.0702 | 0.0366 | 5.79 |
| T3 | 0.0989 | 0.0384 | 8.17 |

Gate **passes**. ROI pixel sets are identical to worklog 21's.

### RNA static gate *(pending)*

### Frozen response, both models *(pending)*

### Refits *(pending)*

## OBSERVATION *(pending)*

## INTERPRETATION *(pending)*

## UNRESOLVED QUESTION *(pending)*
