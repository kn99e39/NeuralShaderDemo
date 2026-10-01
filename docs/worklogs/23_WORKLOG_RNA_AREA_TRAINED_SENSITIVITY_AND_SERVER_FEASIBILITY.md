# Worklog — Area-Trained RNA Sensitivity and LabServer63 Feasibility (2026-10-01)

## Session question

Worklog 22's frozen RNA render was visibly noisy (surface-fixed sparkle). Is
the RNA side of the cross-backbone result an artefact of that noise, i.e. of
poor static reconstruction, or does the same stale interaction pattern remain
when the noise source is removed? This is the roadmap's first immediate step
(`docs/RESEARCH_ROADMAP.md` §16.1).

Side question raised by the user: can LabServer63 take Mitsuba-bound work
(dataset generation, 8DNA training) now that its OptiX problem is understood?

## Relation to worklog 22

Worklog 22 is unchanged and stays the primary result. Everything below is a
**post-hoc sensitivity analysis**, declared after the frozen RNA output was
seen, in its own protocol file
(`experiments/8dna_deformation_replication/protocol/teaset_rna_area_trained_sensitivity.json`).
8DNA, the references, ROIs, decision rule and RNA feature buffers are the
worklog-22 ones; only RNA's training light changes.

## IMPLEMENTATION FACT

### Diagnosis of the noise (before any change)

- Not inference noise: frozen RNA's own T0 seed repeat in the interaction ROI is
  0.0060 display MAE (worklog 22).
- The training targets: RNA's official training light is a delta directional
  light per pixel. On the released near-mirror Ni_palik nickel this makes the
  targets firefly-dominated (measurement below). The surface-fixed pattern in
  the RNA render matches a triplane memorising that noise.
- This was visible before RNA was trained: worklog 22's spp study
  (`rna_teaset/spp_cost_quality.json`) recorded a 95% mean on-asset per-pixel
  relative error at 512 spp. I judged the spp choice on display MAE (0.0029),
  which hides heavy tails, and dismissed the relative error as inherent. That
  was a wrong call; the targets' tails should have been checked then.

### Area-trained variant

- **Training light.** RNA's official generator has a `light_radius` knob (the
  Blender sun's angular size; official configs use 0, i.e. delta). The variant
  uses, per pixel, a square area light centred on that pixel's random direction,
  built exactly like the evaluation emitter (distance 8, half-angle 15°,
  Mitsuba `look_at` axes), irradiance 5 at the origin. Visibility
  (`diffuse_direct`) is "any part of the square unoccluded and above the
  surface" (4×4 stratum centres), the meaning of the official Cycles pass
  thresholded > 0 under a soft light. Cameras, light directions, seeds, spp
  (512) and view counts are the worklog-22 ones.
- **Estimator** (`rna_bridge.trace_directional(..., squares=...)`): uniform
  point on the lane's square, weight L_r·A·cos θ_light/r², segment shadow test,
  no MIS. The delta path is unchanged.
- **Inference** (`rna_infer.py --light-model area-trained`): the model has
  learned the area-light response, so it is queried once per sub-pixel sample
  at the light's centre direction with any-visibility taken from the existing
  feature buffers' 16 light samples, times E/5 — the official renderer with a
  sun of angular size. (Worklog 22 integrated a delta-trained model over 16
  light samples instead.)
- **Validation** (`test_rna_bridge.py`, all six tests pass,
  `rna_teaset/bridge_tests/bridge_tests.json`): test 6 renders with every pixel
  at the protocol direction against Mitsuba's C++ `path` on the evaluation
  scene — mean 0.10536 vs 0.10488 (+0.46%); noise 1.6× Mitsuba's own seed
  repeat at equal spp (no MIS); 0 pixels marked not-visible receive direct
  light.

### Where each job ran

| job | machine | time |
|---|---|---|
| area-trained T0 validation set (40 views) | LabServer63 GPU | 6400 s (2.7 min/view) |
| area-trained T0 training set (200 views) | RTX 5080 | 3.2 h (≈0.95 min/view) |
| RNA training (250 epochs) | RTX 5080, WSL, detached | 25 min |
| features, inference, evaluation, exports | RTX 5080 | minutes |

The training set was started on the server, stopped at 6/200 views and moved
to the 5080 once the per-view times were known. The server and the 5080
produce bit-identical views for the same seeds and code (below), so one
dataset can mix the two.

### LabServer63 feasibility (server facts for this project)

- The server's kernel driver and `libcuda` are 595.71.05, but `libnvoptix`,
  `libnvidia-rtcore` and `libnvidia-gpucomp` are 570.195.03; Mitsuba segfaults
  building an OptiX acceleration structure (worklog 22).
- Fix without root or system changes: extract the 595.71.05 user-space
  libraries from NVIDIA's public driver package (`--extract-only`) into
  `~/nvlibs/optix595` (`libnvoptix.so.1`, `libnvidia-rtcore`,
  `libnvidia-gpucomp`, `nvoptix.bin`) and set `LD_LIBRARY_PATH` to it.
  `ninja` was added to the server's 8DNA venv for the upstream extension
  build. Mitsuba `cuda_ad_rgb` then renders.
- Code reaches the server as a git bundle into a worktree (`~/nsd_w22`), so
  each server run is at an exact local commit.
- **Dataset generation: works and is exact.** Two worklog-22 T0 validation
  views regenerated on the server match the 5080's file: every channel
  identical except one normal value (0.0024, one float16 step).
- **8DNA training: does not fit.** A one-epoch smoke test of the upstream
  trainer (T0, seed 9, same wrapper as the 5080 retrain) ran out of GPU memory
  twice at the first epoch's start, also with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The failing allocation is
  2 GiB in `PathSamplingDataset.resample`, a GPU `randperm` over 128⁴ ≈ 2.7×10⁸
  sample indices, on the 3080 Ti's 12 GiB. It runs only with a smaller dataset,
  which changes the training schedule, so 8DNA training stays on the 5080.
- Placement rule agreed with the user: the 5080 is the default for every job,
  including RNA training; the server takes parallel work only while the 5080
  is busy, or on request.

### Operational incidents (none changed an evidence output)

- A background wait loop expanded `~` in a remote path on the local side and
  never saw the server finish; the comparison ran hours later than it could
  have.
- A `pkill -f` pattern matched its own ssh session's shell and ended it.
- The evaluation divided by a zero model noise floor (see measurement); fixed
  in `5311ed8` before the recorded run.

## MEASUREMENT

### Training-target tails (T0 validation sets, 40 views, same cameras and light directions)

`rna_teaset/target_tail_stats.json`, medians (p90):

| | delta light (worklog 22) | area light (this run) |
|---|---|---|
| share of on-asset energy in the top 0.1% pixels | 0.314 (0.654) | **0.067** (0.135) |
| local speckle, mean \|px − 3×3 median\| / median | 0.44 (0.87) | **0.22** (0.33) |
| on-asset maximum | 111 (171) | **13** (15) |

### RNA training

Validation-best checkpoint: epoch 138, 19.41 dB. Not comparable with worklog
22's 18.87 dB: the validation targets differ (area vs delta light).

### Static gate G1 and T0 errors (against the shared reference)

| ROI | worklog 22 RNA | area-trained RNA |
|---|---|---|
| interaction | 0.0900 | 0.1017 |
| tray | 0.1228 | 0.0973 |
| far | 0.0852 | 0.0880 |
| mover | 0.0982 | 0.1072 |

G1 requires the T0 interaction error to be below the reference's T3
interaction change (0.0939): worklog-22 RNA passes (4% margin), the
area-trained RNA **fails** (0.1017).

### Frozen response (interaction ROI unless noted)

| state | dG | WL22 RNA rise / gain / dN | area-trained rise / gain / dN | area-trained mover rise / gain |
|---|---|---|---|---|
| T1 (control) | 0.0442 | +1.5% / 0.78 / 0.0428 | +0.4% / 0.70 / 0.0389 | +3.7% / 0.65 |
| T1b | 0.1069 | +69% / 0.00 / 0.0062 | +44% / 0.000 / 0.0002 | +22.5% / 0.18 |
| T2 | 0.0581 | +17% / −0.01 / 0.0059 | +7.7% / 0.000 / 0.0000 | +14.9% / 0.12 |
| T3 | 0.0939 | +38% / 0.00 / 0.0068 | +15.9% / 0.000 / 0.0004 | +23.8% / 0.12 |

The area-trained model's T0 seed repeat is identical inside the interaction
ROI (391 of 262144 pixels differ, all on penumbra edges): it is queried once
per sample, and only its any-visibility depends on the light-sample seed.
Locked rule outcome: FROZEN FAILURE at T1b (T2, T3 below the 25% rise); the
same in the diagnostic `current` mode. **Because G1 fails, the protocol does
not admit this classification as a verdict.**

Records: `cross_backbone/cross_backbone_frozen_area_trained.json`, renders
`rna_teaset/frozen_area_trained/`. Reviewer panels: `results/evaluation/23/`.

## OBSERVATION

- The area light removes most of the target fireflies (top-0.1% energy share
  0.31 → 0.07, maximum 111 → 13), and the area-trained render is less grainy
  on the parts, but the tray still sparkles (`01_…`, `02_interaction_crops.png`).
- Static quality moves in both directions: tray error falls 21%, interaction
  error rises 13%.
- With either training light, RNA's output in the interaction ROI does not
  follow the relation change. The area-trained model's change there is
  essentially zero (dN ≤ 0.0004 against dG 0.058–0.107), while it follows the
  relation-preserving T1 motion (gain 0.70) and partly the moved part itself
  (mover gain 0.12–0.18).
- In the GT crops the moved milk pot's mirror reflection in the stationary tray
  and tray wall moves with it; neither RNA reproduces that movement.
- Relative rises are smaller for the area-trained model mainly because its T0
  error is larger; its absolute T1b/T3 errors (0.146 / 0.118) are close to the
  worklog-22 model's (0.152 / 0.124).

## INTERPRETATION

- **The tracking failure is not an artefact of the target noise.** Removing
  most of it changes RNA's static error pattern but leaves gain ≈ 0 in every
  relation-changing state. This is the answer to the roadmap's §16.1 question
  as far as RNA can give it: the shared pattern is not explained by RNA's poor
  static reconstruction alone. It stays *supporting* evidence, because neither
  RNA variant reconstructs this material well and the area-trained one fails
  the predeclared gate.
- The remaining sparkle is most plausibly a representation/sampling limit, not
  target noise: near-mirror, view-dependent reflections of other parts in the
  tray are hard for a triplane + MLP fitted from 200 views. Not measured
  directly.
- The worklog-22 primary result (both backbones fail T1b and T3) and its
  caveat (RNA's thin G1 margin) are unchanged.
- Practical: LabServer63 can host Mitsuba rendering and dataset generation
  with exact results, about 3× slower than the 5080; it cannot run the
  upstream 8DNA trainer as configured.

## UNRESOLVED QUESTION

- Would a less mirror-like material, or a second asset, give RNA a static
  quality that clears G1 comfortably? That is the breadth axis of the roadmap;
  this batch was not allowed to change the material.
- Is the sparkle a view-count (200) limit or a triplane-resolution limit? Not
  tested; the roadmap does not need it unless RNA becomes the method substrate
  on this material.
- Reference-only transport decomposition of the interaction ROI (which
  component — mirror inter-reflection of the moved part, near-contact
  occlusion, higher order — carries dG) is still open; the crops suggest
  specular inter-part reflection dominates, which has not been measured.

## Commits

| output | commit |
|---|---|
| bridge area mode, inference, protocol, server run script | `6e98be7` |
| area-trained validation set (server) | generated at `6e98be7` (H5 `project_commit` attribute, HEAD at write time: `ea26891`; generation code identical) |
| area-trained training set (5080) | generated at `dc2b29a` (attribute `39604b0`, the user's docs merge landed during the run; generation code identical) |
| RNA training | WSL detached run at `39604b0`, config `configs/rna_teaset_T0_area_trained.yml` |
| RNA renders | `39604b0` |
| frozen comparison record | `5311ed8` (zero-noise-floor fix) |
| target statistics and `results/evaluation/23/` | `5790ae3` |
| bridge tests incl. test 6 | `6e98be7` |
| server 8DNA smoke test | `4a4220d` |

Scaffolding: `ea26891`, `dc2b29a`, `4a4220d`.
