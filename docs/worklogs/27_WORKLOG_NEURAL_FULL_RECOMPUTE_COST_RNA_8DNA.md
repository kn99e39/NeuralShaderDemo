# Worklog — Neural Full-Recompute Cost of RNA and 8DNA on the Teaset T3 State (2026-10-06 – 2026-10-07)

## SESSION QUESTION

After the accepted teaset T0 -> T3 change (worklogs 21/22), how long does an
existing neural transport baseline take to produce a representation that is
valid for T3 again, when it is rebuilt exactly as worklog 22 refit it? Is
that latency small enough for full neural recomputation to be a credible
response to geometry change, or does a large gap remain?

This is a measurement of the strongest existing alternative to an
incremental method. No incremental method was designed or implemented. RNA
and 8DNA source, configs, schedules, seeds, losses, data sizes, recovery
thresholds and evaluation settings were not changed. Worklogs 21–26 were not
edited.

**Answer, measured on the RTX 5080 (one asset, one change, fixed historical
schedules):**

| track | first checkpoint satisfying the worklog-22 rule | historical selection usable (verdict) | total fixed schedule |
|---|---|---|---|
| 8DNA, worklog-21 envmap (primary) | **38.9 min** (epoch 10; oracle, see below) | 1.76 h (last.ckpt: **recovers**, 1.149 / 0.629) | **1.76 h** |
| 8DNA, common light (secondary, same training) | 63.5 min (epoch 17, isolated) | 1.76 h (does not recover, 1.364 / 0.595) | 1.76 h |
| RNA, common light | **none within the fixed schedule** (0 of 62 checkpoints) | 3.24 h (validation-best: does not recover, 1.322 / 0.433) | **3.24 h** |

Every figure is category C (offline): 10^5–10^6 frames of a 30/60 FPS budget.

## BASELINE SEMANTICS

### What "full recompute/refit" meant historically

Established from the worklog-22 protocols, configs and chain logs before
anything was timed (`protocol/nrc_t3_recompute_v1.json`, block
`historical_refit_semantics`):

| question | 8DNA (T3_refit) | RNA (T3 refit) |
|---|---|---|
| initialised from scratch? | yes: `from_scratch: true`, upstream `train.py`, `seed_everything(9)` | yes: official `scripts/train.py`, seed 0 |
| initialised from the T0 checkpoint / learned state reused? | no | no (RNA's train.py has `TODO: Add init_from checkpoint support`) |
| training data regenerated from T3? | yes, online: `PathSamplingDataset.reload` path-traces new samples from the T3 scene at the start of every epoch | yes, offline: 200 train + 40 val H5 views rendered from T3 by `rna_bridge.py h5` before training |
| path samples regenerated online? | yes, every epoch (`dataset_reload: 1`) | no (fixed H5 targets) |
| stages omitted because an artifact existed | none for the refit itself | none for the refit; the inference feature buffers (`features/T3.npz`) existed, but they are an inference input, not part of the representation |
| stages a new configuration requires | scene load, per-epoch path generation + shuffle, 30 epochs of optimisation, per-epoch validation and checkpointing, conversion to the `load_asset` layout | T3 target rendering (train + val), H5 write and load, 250 epochs, per-epoch validation and checkpointing, validation-best selection |
| checkpoint the protocol selects | `last.ckpt` at the end of the schedule | validation-best `val_psnr` |
| lighting | 8DNA training is lighting-independent: the one T3 checkpoint was evaluated in both regimes | common light only |

### The two concepts, applied

- **A. REPRESENTATION REBUILD** (normal untrained initialisation, standard
  pipeline): this is the historical worklog-22 path for **both** backbones,
  so it is the primary measurement for both.
- **B. FULL REFIT / WARM RECOMPUTE**: **not measured.** Neither method has an
  existing, supported warm-start path. RNA has no `init_from`. 8DNA's
  `--resume` restores the trainer state of the same run, epoch counter
  included, so it is not a fine-tune from the T0 asset. A warm start would be
  new methodology, so no secondary control exists to measure.

### Correction of record: what worklog 22's 8DNA "last.ckpt" held

Found while diagnosing reproducibility, not before the runs. The worklog-22
`refit/T3_refit/last.ckpt` holds **epoch 28** (`global_step` 237 568); the
epoch-29 end-of-schedule weights are only in `final.ckpt`. Both files have
`epoch` fields and steps that show this, and `last.ckpt` is bit-identical to
`epoch=28-step=237568.ckpt`. Worklog 22's refit verdicts (envmap 1.07 / 0.70,
common light 1.28 / 0.58) therefore describe epoch-28 weights. This changes
no verdict. Worklog 22 is not edited. In this batch's rebuild, `last.ckpt`
is epoch 29, verified bit-equal to the epoch-29 snapshot and to the export.

## IMPLEMENTATION FACT

### Instrumentation only

New experiment folder `experiments/neural_recompute_cost/` (README lists
every file). All of it wraps the unchanged historical entry points:

- `nrc_8dna_train_timed.py` runs `train_8dna_state.py` -> upstream
  `train.py`. It wraps `PathSamplingDataset.__init__`/`reload`/`resample`,
  `EightDNA.__init__` and `Trainer.save_checkpoint` with begin/end events and
  appends one Lightning callback (optimisation, validation, epoch
  boundaries). It writes a per-epoch weight snapshot to a separate directory,
  and after training it times the `as_released_layout` export.
- `nrc_rna_h5_timed.py` runs `rna_bridge.generate_h5` with a derived protocol
  whose only change is `rna_dataset_dir`. Each `render_view` call is timed.
- `wsl/nrc_rna_train_timed.py` runs RNA's `scripts/train.py` through
  `runpy`, with the same kind of callback and a weights-only snapshot after
  epoch 0 and every 5th epoch (WSL-native directory, copied out afterwards).
- `nrc_chain.py` runs the stages one at a time on the 5080 and records each
  stage's launch and exit. The WSL stage goes through `wsl_run_detached.sh`.
  The chain itself was started through `windows/launch_detached.ps1`
  (WMI-created, outside any tool's process tree).
- Evaluation scores every snapshot with the unchanged
  `cross_backbone_eval.refit_block` (`nrc_eval_8dna.py`, `nrc_eval_rna.py`).
  Inference cost is measured separately (`nrc_inference_timing.py`).
- The wrappers draw no random numbers and do not touch the model, optimiser,
  data or schedule. Their side effects are `torch.cuda.synchronize()` at
  phase boundaries and the snapshot writes. The snapshots cost 0.14 s in
  total for 8DNA and 1.2 s for RNA.

Deviations from worklog 22, declared in the protocol before the runs:

1. RNA training ran on the RTX 5080 (WSL), not LabServer63, so all timing is
   on one host. This is the worklog-23 route on the 5080.
2. Outputs went to `results/neural_recompute_cost/v1_d2d6432/`. Historical
   result directories were only read.
3. The derived RNA config differs from `configs/rna_teaset_T3.yml` only in
   `name` and the two dataset paths (checked line by line by the chain).

### Defects found in smoke runs and fixed before the evidence run (not evidence)

- Lightning 2.1 runs `ModelCheckpoint` after every other callback and writes
  in `on_train_epoch_end`, so the checkpoint writes fell outside my epoch
  interval. An epoch now runs from one `on_train_epoch_start` to the next.
- Wrapping `mi.load_dict` had no effect, because Mitsuba caches attribute
  lookups per variant. Scene preparation is now reported as a derived
  interval: 8DNA's `PathSamplingDataset` constructor, and RNA's
  `generate_h5` start -> first view.
- RNA checkpoints pickle RNA classes, so their metadata and weight digests
  are read in RNA's WSL venv.
- The identity check's WSL comparison lacked RNA's root on `sys.path`.

### Tests and checks

| check | result |
|---|---|
| `tests/test_nrc_records.py` (schema, phase accounting, double-counting detection, cumulative arithmetic, stage gaps, checkpoint->wall-clock mapping incl. clock offset, first recovery, budget ratios, clean-tree refusal, NaN refusal, serialisation round trip) | 14 tests, all pass |
| `tests/test_historical_rule.py`: the batch's evaluation inputs reproduce worklog 22's three refit verdicts | **exact equality** (envmap 1.0729884 / 0.6978, common light 1.2810283 / 0.5780, RNA 1.2161971 / 0.5589) |
| smoke chains `smoke1` (exposed the defects above), `smoke2` | `smoke2` passes end to end |
| instrumentation non-perturbation (`nrc_identity_check.py`, tiny schedule, `identity_check_d2d6432/`) | Neither trainer is bit-deterministic run to run. Instrumented vs plain max weight difference: 8DNA 0.0202 against a plain-vs-plain floor of 0.0196; RNA 5.0e-8 against 5.0e-8. Both **pass** the declared rule (≤ 2 × floor; the tolerance was set after the first smoke check revealed the non-determinism, before any evidence run). |
| checkpoint-to-snapshot mapping | 8DNA: last snapshot == `last.ckpt` == export, bit for bit. RNA: 4 official/snapshot pairs of the same epoch have identical weight digests. |
| evaluation-path determinism | re-rendering the worklog-22 selections through this batch's path gives images **bit-identical** to the historical renders (max diff 0.0, all three tracks) |

### Environment

| item | value |
|---|---|
| host | RTX 5080 16 303 MiB, driver 596.49; AMD Ryzen 9 9950X3D; 63.6 GB RAM; Windows 11 Pro 10.0.26200; desktop applications open, no other GPU compute job |
| 8DNA / datasets | native Windows, Python 3.11.9, torch 2.8.0+cu128, Lightning 2.1.3, Mitsuba 3.5.1 / DrJit 0.4.6 (`cuda_ad_rgb`, LoopRecord off, VCallRecord on), 8DNA upstream `4a2157c` (clean) |
| RNA training / inference | WSL Ubuntu-22.04 (kernel 6.18.33.2), Python 3.10.12, torch 2.8.0+cu128, pytorch_lightning 2.5.5, RNA upstream `66b5b09` (the checkout holds untracked wheel and validation-EXR outputs, as before; no source change) |
| Blender | not used |
| WSL-Windows clock offset | +0.014 s before / +0.016 s after the WSL stage (round trip 0.05 s); the mean is applied |

### Commits

| commit | content |
|---|---|
| `094ff4f` | protocol, instrumentation, chain, evaluation, report, tests |
| `d2d6432` | identity-check fix. **All evidence runs**: chain `v1_d2d6432`, identity check `identity_check_d2d6432`; every process recorded `project_dirty: []` |
| `eba9be0`, `5d94e5c` | historical rescore diagnostic; it ran at `5d94e5c` (the `eba9be0` run was stopped because it omitted the 8DNA `last.ckpt`; its output was discarded) |
| `850614e`, `f3b5d07` | report figures (analysis only; exports in `results/evaluation/27/` written at `f3b5d07`) |

Nothing was pushed.

## 8DNA FULL-RECOMPUTE MEASUREMENT

### Phase timing (one process, T3 handed over -> export done)

| phase | seconds | share |
|---|---|---|
| process start-up, imports, Mitsuba/torch init | 3.0 | 0.05% |
| scene preparation (`PathSamplingDataset` constructor: T3 scene, integrator, buffers) | 0.14 | — |
| model init | 0.01 | — |
| sanity validation | 1.0 | — |
| **path generation** (online, 30 × `reload`) | **406.5** (13.5 s/epoch) | 6.4% |
| shuffle (30 × `resample`, 128⁴-sample permutation) | 227.9 (7.6 s/epoch mean; 21.5 s in epoch 0) | 3.6% |
| **optimisation** (30 × 8192 steps) | **5679.5** (189 s/epoch) | **89.4%** |
| per-epoch validation (`render_simple`, 128 spp) | 20.3 | 0.3% |
| checkpoint writes (top-8 + final) | 0.65 | — |
| export (`load_asset` layout) | 0.03 | — |
| snapshots (instrumentation) | 0.14 | — |
| other (Lightning loop overhead, environment record, teardown) | 15.0 | 0.2% |
| **total (one fixed 30-epoch schedule)** | **6354.0 = 1.76 h** | |

The epoch wall time averages 211.6 s. Worklog 22's run of the same schedule
took 2 h 04 min. This run had no other job on the GPU; whether that one did
is not recorded.

Memory: torch peak allocated 15.6 GiB (15 940 MiB; reserved 15.8 GiB of the
15.9 GiB card). Whole-GPU peak 15.5 GiB, 13.7 GiB over baseline. Host peak
working set 23.9 GB, mostly the 2.7 × 10⁸-sample CPU buffers.

### Quality vs wall-clock time (worklog-21 envmap regime, primary)

Each snapshot was rendered at 512² / 512 spp with seed 0 and scored with the
historical rule (ratio ≤ 1.25 **and** gain ≥ 0.5). "Usable" = that epoch's
checkpoint written + export. Full trace: `07_timing_8dna_envmap_primary.json`
and figure `01`.

| epoch | usable at | ratio | gain | rule |
|---|---|---|---|---|
| 0 | 4.0 min | 1.895 | 0.504 | no |
| 4 | 17.9 min | 1.372 | 0.606 | no |
| 9 | 35.5 min | 1.282 | 0.666 | no |
| **10** | **38.9 min** | **1.229** | 0.615 | **yes (first)** |
| 11–14 | 42.5–52.8 min | 1.26–1.42 | 0.60–0.65 | no |
| 15–18 | 56.4–67.0 min | 1.15–1.25 | 0.60–0.63 | yes |
| 19–20 | 70.5–73.9 min | 1.30–1.34 | 0.61–0.65 | no |
| 21–29 | 77.4–105.9 min | 1.09–1.23 | 0.59–0.68 | yes, every one |
| 29 (historical selection) | 105.9 min | **1.149** | **0.629** | **yes** |

- **time_to_first_recovery = 2333 s (38.9 min).** This is an oracle
  quantity: telling that epoch 10 has recovered needs the path-traced T3
  reference. The crossing is also fragile (1.229 against 1.25), and the next
  four checkpoints fail again.
- Descriptive, derived after the measurement (not a protocol quantity): from
  **epoch 21, 4641 s (77.4 min)**, every later checkpoint satisfies the rule.
- **time_to_historical_selection = total_fixed_schedule = 6353 / 6354 s
  (1.76 h).**
- **Reproduction gate: PASS.** The rebuild's `last.ckpt` recovers
  (1.149 / 0.629 against the historical 1.073 / 0.698), so the 8DNA timing is
  interpreted as a valid recovery baseline.

### Run-to-run comparison with worklog 22 (same rule, same evaluation path; figure `09`)

The worklog-22 run's surviving checkpoints, epochs 20–28: all 8 recover,
ratio 1.07–1.22, gain 0.68–0.73. This rebuild, epochs 20–29: 9 of 10
recover, ratio 1.09–1.34, gain 0.59–0.68. Both runs recover late in the
schedule. This rebuild sits slightly closer to the threshold, especially in
gain. That is training non-determinism on one host: the data are regenerated
online with the same seed, and the evaluation path is deterministic.

### Inference cost (existing evaluation path; separate from rebuild latency)

| | value |
|---|---|
| `load_asset` | 0.01 s |
| 512², 1 spp, upstream `neuralpath` (5 warm repeats) | **0.164 s** envmap / 0.159 s common light = 4.9× the 30 FPS / 9.8× the 60 FPS budget |
| 512², evaluation spp | 50.9 s at 512 spp (envmap), 100.7 s at 1024 spp (common light) ≈ 0.099 s per spp |

This is the offline Mitsuba/DrJit research integrator, not a real-time
implementation. No quality claim is made at 1 spp.

## RNA FULL-RECOMPUTE MEASUREMENT

**Static-quality caveat (unchanged, worklogs 22–24):** RNA reconstructs this
near-mirror nickel poorly (worklog-22 G1 margin 4%; heavy tray sparkle, visible
again in panel `05`). Its recovery reading is supporting evidence, not
co-equal with 8DNA's.

### Phase timing (T3 handed over -> validation-best usable)

| stage / phase | seconds |
|---|---|
| **H5 train targets** (200 views, 512², 512 spp) | **8450.9** |
| — view rendering | 8439.6 (42.2 s/view) |
| — start-up, scene preparation, H5 writes | 1.7 + 0.1 + 9.3 |
| **H5 validation targets** (40 views) | **1694.0** (1690.5 rendering) |
| inter-stage gaps (chain; incl. 3.2 s clock-offset probe + snapshot-dir setup before WSL) | 3.2 |
| **RNA training** (WSL, 250 epochs) | **1500.3** |
| — start-up and imports over the 9p `/mnt/c` mount | 48.2 |
| — H5 load (train + val) | 9.5 |
| — optimisation | 1050.6 (4.2 s/epoch) |
| — validation | 99.3 + 0.7 sanity |
| — checkpoint writes (official top-10 + last, 88 MB each, to `/mnt/c`) | 277.6 |
| — snapshots (instrumentation) | 1.2 |
| — other / status poll and teardown | 11.0 / 2.2 |
| **total fixed schedule** | **11 648.3 s = 3.24 h** |

Target generation is **87%** of the total: 10 130 s of view rendering. The
historical generation of the same files took 8 491 s + 1 691 s, and its
training took 34 min on LabServer63.

Memory: H5 generation peaked at 8.6 GiB whole-GPU and 6.8 GB host. Training
peaked at 3.3 GiB torch allocated (9.8 GiB reserved), 11.0 GiB whole-GPU
and 4.6 GB host RSS.

**Dataset reproducibility:** the regenerated `teaset_T3_train.h5` and
`teaset_T3_val.h5` are **bit-identical** to the historical files: all 9
channels, 200/200 and 40/40 views, every attribute except `project_commit`
(`eval/dataset_compare.json`).

### Quality vs wall-clock time

The first checkpoint (epoch 0) is usable at 170.3 min, i.e. after the
targets. 62 checkpoints were scored: 51 snapshots, 10 official top-k and
`last`.

| epoch | usable at | ratio | gain | val_psnr | rule |
|---|---|---|---|---|---|
| 0 | 170.3 min | 3.094 | 1.084 | 12.20 | no |
| 49 | 175.7 min | 2.020 | 0.990 | 17.05 | no |
| 99 | 180.6 min | 1.469 | 0.789 | 16.64 | no |
| **117 (validation-best, historical selection)** | 182.2 min (known only at 194.1 min) | **1.322** | **0.433** | **18.02** | **no** |
| 150–249 | 185–194 min | 1.31–1.46 | 0.48–0.70 | 16.9–17.9 | no |

- **No recovered checkpoint within the fixed schedule** (0 of 62). The
  minimum ratio was 1.312 (epoch 193).
- **time_to_historical_selection = 11 646 s (3.24 h).** The validation-best
  checkpoint cannot be selected before the last epoch has been validated.
- **total_fixed_schedule = 11 648 s (3.24 h).** Training was not extended.

### Reproducibility diagnosis: why worklog 22's borderline RNA pass did not recur

- **Targets: identical** (bit for bit).
- **Evaluation: identical.** The historical validation-best re-rendered
  through this batch's path equals the historical render exactly.
- **Training: different.** The historical run's 11 surviving checkpoints
  score ratio 1.18–1.28. Nine of them pass the rule; its validation-best is
  1.216 / 0.559. This run's late checkpoints score 1.31–1.39, and none pass.
  The validation PSNR is nearly the same: 18.11 dB historical vs 18.02 dB
  here. Same config, seed and data. The causes left are the training host
  (RTX 3080 Ti SM 8.6 vs RTX 5080 SM 12.0, with RNA's
  `float32_matmul_precision("medium")`) and run-to-run non-determinism. The
  identity check shows RNA is not bit-deterministic even on one host. One
  run per host cannot separate these two causes.
- Reading: worklog 22's RNA common-light "A, borderline" (1.22 against 1.25)
  is **not robust**. In this rebuild RNA's same-state recovery lands about
  0.1 above the ratio limit. Both runs are near the threshold. Per protocol,
  this is reported as "no recovered checkpoint within the fixed schedule" and
  is not fixed or tuned.

### Inference cost (separate)

| | value |
|---|---|
| inference inputs, 512² (G-buffer at 4×4 sub-pixel samples + 16 area-light samples each, current visibility) | 4.7–5.3 s Mitsuba compute + 12.0 s compressed `.npz` write; regenerated buffers bit-identical to the historical T3 buffers |
| network evaluation, validation-best model, 30.8 M queries (1.92 M hit samples × 16 light samples) | **3.04 s** (5 warm repeats; output identical to `rna_infer.py`'s, max diff 0.0) |
| model load / feature read / upload | 5.05 s / 3.34 s / 0.08 s |
| full `rna_infer.py` call | 6.8 s |

This is the project's offline evaluation path (16 × 16 samples per pixel),
not a real-time renderer.

## OPTIONAL COMMON-LIGHT CONTROL (8DNA, executed)

It needed no new training: the same 8DNA rebuild was evaluated at 1024 spp in
the common-light regime.

- 1 of 30 checkpoints satisfies the rule: epoch 17, 1.240 / 0.514, at
  63.5 min. It is isolated; epochs 16 and 18 fail.
- Historical selection (epoch 29): 1.364 / 0.595, **does not recover**. This
  is consistent with worklog 22's case B by the rule (1.281 / 0.578, an
  epoch-28 checkpoint). The worklog-22 run's own surviving checkpoints pass
  1 of 8 times (epoch 22).
- Gains scatter 0.31–0.69 between neighbouring epochs. That is the known
  large single-seed neural noise of this regime (8DNA T0 seed repeat 0.0207,
  worklog 22). A single crossing here carries little information.
- Cross-backbone, common light: 8DNA's rebuild costs 1.76 h and lands near
  but mostly above the ratio limit. RNA's costs 3.24 h and stays about 0.1
  above it. **Neither backbone recovers by the rule in this regime with its
  historical selection.** No capacity ranking is implied (worklog 22, rule 3).

## COST COMPARISON

### Between the neural baselines (same host, same teaset T3 change)

| | 8DNA | RNA |
|---|---|---|
| earliest checkpoint of any quality | 4.0 min | 170.3 min |
| first rule-satisfying checkpoint (oracle) | 38.9 min (envmap) | none |
| historical selection usable | 1.76 h | 3.24 h |
| dominant cost | optimisation 89% | target rendering 87% |
| irreducible part of the existing pipeline | one epoch ≈ 3.5 min (path generation + shuffle + 8192 steps) | 2.82 h of target rendering before the first step (200 + 40 views at 512 spp) |

### Frame-budget ratios (descriptive only)

| quantity | seconds | × 33.3 ms (30 FPS) | × 16.7 ms (60 FPS) | category |
|---|---|---|---|---|
| 8DNA first recovery (envmap) | 2 333 | 7.0 × 10⁴ | 1.4 × 10⁵ | C |
| 8DNA all later checkpoints recover (envmap, post hoc) | 4 641 | 1.4 × 10⁵ | 2.8 × 10⁵ | C |
| 8DNA full schedule | 6 354 | 1.9 × 10⁵ | 3.8 × 10⁵ | C |
| RNA full schedule (no recovery) | 11 648 | 3.5 × 10⁵ | 7.0 × 10⁵ | C |
| 8DNA epoch-0 checkpoint (not recovered; lower bound of this pipeline) | 239 | 7.2 × 10³ | 1.4 × 10⁴ | C |

### Worklog 26 as cost context only

Worklog 26 used a different scene (BMW27), resolution (1080p) and renderer
(Cycles), and its frames are not quality matched to anything here. For
context only, physically current GI after a rigid change cost 0.22 s
(16 spp), 0.76 s (64 spp) and 3.0 s (256 spp) per frame there. The fastest
neural rebuild measured here to reach its recovery rule (38.9 min) is about
three to four orders of magnitude longer than those frames. That is
**not** an equal-quality speed ratio, and no such ratio is claimed.

## QUALITATIVE REVIEW

`results/evaluation/27/` (manifest with sources and SHA-256 of every file).
Panels `03`–`05`: GT T3 | frozen T0-trained model at T3 | first recovered
checkpoint | historical selection. Each panel has a full frame, the
interaction-ROI crop and an error map; there is a crop-cycle GIF per track.

- **8DNA envmap (`03`).** The frozen model keeps the milk pot's mirror image
  and a dark smear on the tray at the pot's T0 position. At epoch 10 and at
  epoch 29 the reflection follows the moved pot and the smear is gone. Epoch
  10 still shows blotchier error at the pot base than epoch 29. Both are
  blurrier than GT, which is the released model's known static blur. The
  numerical recovery corresponds to the known transport change being
  restored.
- **8DNA common light (`04`).** The frozen model's dark stale image on the
  tray at the milk pot's T0 base disappears at epochs 17 and 29, and the
  reflection follows the moved pot. Large residual errors remain on the tray
  highlights and on the pot itself, higher than in the envmap regime, which
  is consistent with the near-threshold ratios.
- **RNA (`05`).** The validation-best rebuild changes the interaction ROI
  relative to the frozen model, but the tray and pot keep the strong
  sparkle. The error map is saturated over most of the ROI, as the frozen
  model's is. Nothing in the panel would make a reviewer call it restored.
- Figure `01`: quality vs wall time for all tracks. Figure `02`: phase
  breakdown. Figure `09`: this rebuild vs worklog-22 checkpoints by epoch.
  Table `06`: latency summary. Files `07`/`08`: machine-readable records.

## INTERPRETATION

- **Measured answer to the batch question.** On this host, after the accepted
  teaset change, the existing neural baselines rebuilt by their historical
  pipelines return a T3-valid representation in **tens of minutes at the
  earliest (8DNA: 39 min, oracle-identified and not yet stable; 77 min until
  it stays valid) and 1.8 h for the protocol's own selection**. RNA did not
  reach its recovery criterion at all in 3.2 h. Every measured latency is in
  category C (offline), five orders of magnitude above a 30/60 FPS frame.
- **Where the time goes differs by method, and neither part is incidental.**
  8DNA's cost is optimisation: path generation is online and cheap
  (13.5 s/epoch). RNA's cost is rendering the supervision: 2.8 h before the
  first gradient step. For RNA, any rebuild with this supervision protocol is
  bounded below by its target rendering. For 8DNA, by at least one epoch of
  about 3.5 min, and that epoch-0 checkpoint does not recover.
- **The schedules are not tight.** 8DNA first satisfies the rule a third of
  the way into its schedule. That might suggest a much cheaper rebuild, but
  this batch did not test it, and the trace makes it unlikely to reach a
  human timescale: no checkpoint before 38.9 min passed. A purpose-built fast
  refit is a different method from the baseline measured here and was out of
  scope.
- **Recovery itself is near-threshold and run-dependent outside the envmap
  case.** The envmap 8DNA recovery reproduces across two runs, so the
  stale-state reading of worklogs 21/22 stands. The common-light 8DNA and RNA
  recoveries are within about 0.1 of the ratio limit in both runs. RNA's
  worklog-22 pass did not recur. Same-state refit therefore restores T3 only
  partially and unreliably in the common-light regime for both backbones.
  That is a quality limit of full recomputation, separate from its cost.
- Scope: one asset, one rigid relation change, near-mirror material, fixed
  historical schedules, one host, one rebuild per backbone. The trainers are
  not bit-deterministic, so the epoch of first recovery would move between
  runs; the order of magnitude is not expected to.

## UNRESOLVED QUESTIONS

- How much faster can a rebuild get without changing what is being
  measured? Shorter schedules, smaller H5 sets, lower target spp, mixed
  precision and warm starts were all excluded by this batch's rules. They
  would be new baselines and would need their own protocol.
- RNA: is the 1.22 -> 1.32 shift a host effect (SM 8.6 vs 12.0, TF32-medium
  matmul) or run-to-run variance? It needs repeat runs on each host.
- 8DNA: the run-to-run spread of time-to-first-recovery. One rebuild plus the
  historical run's late checkpoints is not a distribution.
- The 1-spp 8DNA (0.16 s) and RNA (3 s network + 5 s inputs) inference costs
  are the offline evaluation paths. Real-time inference cost of either
  representation was not measured.
- Do other assets, materials or relation changes have different cost
  profiles? 8DNA's cost scales with its fixed sample budget, not obviously
  with scene size; RNA's scales with target rendering. Not measured.

## ARCHITECTURE IMPLICATION

Bounded to the question of whether full neural recomputation is already a
practical competitor to an incremental update:

- **For per-frame dynamic geometry, and for interactive or occasional edits
  at a human timescale (sub-second to seconds): no.** The existing full
  recomputation measured here is offline. Its first possibly usable result
  comes after 39 min (8DNA), its protocol result after 1.8–3.2 h, and in one
  of the two backbones there is no rule-satisfying result at all. The gap to
  a frame budget is about 10⁵ and the gap to a few seconds about 10³. That is
  a large latency gap, and it leaves room for investigating incremental
  reuse of transport state.
- **It remains a credible alternative only for offline per-configuration
  baking,** i.e. when the set of configurations is small and known in
  advance and hours per configuration are acceptable.
- This does not show that an incremental method can close the gap, or at
  what quality. Worklog 26's physical GI cost (0.2–3 s per frame on BMW27)
  remains the tighter context for what such a method would have to beat.

Artifacts: `results/neural_recompute_cost/v1_d2d6432/` (events, checkpoints,
renders, `report/timing_*.json`, `eval/*.json`,
`diag/historical_rescore_5d94e5c/`), `identity_check_d2d6432/`,
`results/evaluation/27/`.
