# Worklog — RNA Reference Reproduction and Environment Qualification on the Official Lego (2026-10-16 – 2026-10-17)

## SESSION QUESTION

User prompt "RNA Reference Reproduction & Environment Qualification" (2026-10-16). This is a
prerequisite baseline batch, not Architecture Batch 3.

> Can the released RNA implementation reproduce its published static rendering quality in our
> environment, and are the remaining differences sufficiently understood to use RNA as a credible
> architecture baseline?

**Short answer: yes, at the "comparable" level, with the remaining gap attributed.**

- **Native pipeline runs.** The upstream code at `66b5b098`, unmodified, together with the
  official custom-Cycles wheel, runs natively on our machine (WSL, RTX 5080).
- **Native training matches the released checkpoints.**
  - Setup: training from scratch with the released configs on a natively generated 400-view
    dataset.
  - On the same validation set the reproduced models match the released checkpoints to within
    0.2 dB: HQ 31.99 vs 32.16, small 27.97 vs 27.96. They track each other view by view.
- **Both sit below Table 1 by the same amount**: HQ −1.9 to −2.1 dB, small −0.8 dB. A
  seeded-replay diagnostic attributes the HQ gap to validation-target noise: the released config
  renders at 1024 spp, while the paper used 4096 spp. Evaluated against 4096-spp targets of the
  same 8 views, the released checkpoints score HQ 34.39 / small 29.22 (Table 1: 34.09 / 28.77).
- **Not reproducible:**
  - Rendering-performance claims (Fig. 6/7): their production path tracer is not released.
  - HQ training wall-clock: limited by our 16 GB GPU.

## RELATION TO EARLIER WORK

- **Worklogs 32–34:**
  - established unit ownership on Teaset RNA (32);
  - audited the integrated baseline (33: no integration defect; C1 representation-limited; C2
    gate noise-limited);
  - attributed the P16 seams (34).

  This batch qualifies RNA itself on the asset it was published with. Worklogs 21–34 and all
  their artifacts are unchanged.
- **Code and outputs:**
  - code: `experiments/rna_reproduction/`;
  - outputs: `results/rna_reproduction/`, workspace `results/rna_reproduction/work`;
  - reviewer exports: `results/evaluation/35/`.
- **Protocol:** `experiments/rna_reproduction/protocol/rr_v1.json`.
  - Written before any validation or training result (commit `51cc2db`).
  - Amendments:
    - `e9d08ab`: released-route outcome and the N1 rule;
    - `4a4627c`: data placement;
    - `317273d`: pause/resume.

## 1. PAPER REPRODUCTION MANIFEST

Paper: arXiv:2312.09398 v2 = TOG 44(1) art. 2, DOI 10.1145/3695866. Numbers are quoted only where
the paper states them. The full manifest came from a sub-agent and was spot-checked.

| target | published value | definition | class |
|---|---|---|---|
| **Table 1** Lego validation PSNR, 4×512 (HQ) / 4×64 (small) | **34.09 / 28.77**; training time 1h13m / 31m17s (A100) | "average PSNR on the validation data". Code: mean over views of PSNR(tonemap) on alpha > 0 pixels, tonemap = Reinhard x/(1+x) then ^(1/2.2) | reproduction with released equivalent artifacts (datasets not shipped; regenerated) |
| Released checkpoints' own logs | best val_psnr 33.94 (epoch 236) / 28.95 (epoch 222) on the authors' val set | Lightning ModelCheckpoint state inside `lego*.ckpt` | reference |
| **Fig. 8** Lego, point light, full image | **32.44 / 28.28** (NRHints 25.48) | view, mask, reference spp and transform unpublished | comparable only |
| Fig. 5 convergence curves | qualitative (no numbers) | — | comparable |
| Fig. 6/7 timings (7×, 60×, 105/46 ms per sample) | — | unreleased production path tracer (CPU + Vulkan) | **blocked** |
| Fig. 10, other tables/rows | partial / not Lego | variants not released | out of scope |

Paper-vs-code discrepancies recorded, not changed:
- Lego data is 4096 spp in the paper but 1024 in `generate_Lego.yml`.
- The paper keeps the 3 best checkpoints; the code keeps the top 10.
- The paper's blur starts at "4 px"; the code uses initial σ 8.
- Shadowed pixels are weighted ×10 in code (not mentioned in the paper).
- The render reference is OIDN-denoised 2048 spp.

## 2. ARTIFACTS AND RUNTIME QUALIFICATION

**Artifacts** (sha256 in `rr_v1.json` and `qualify.json`):
- `lego.ckpt`: epoch 249, step 100 000, 7 354 374 parameters;
- `lego_rt.ckpt`: 6 309 638 parameters, MLP width 64;
- `Lego.zip`: `Lego_rough_aovs_normalized.blend` (404 MB). It has 1 mesh and the AOVs
  position/tangent/normal/incoming, and the custom-Cycles `query_filename` hook is present;
- render configs (`lego_cmp.yml`, `rt_lego_cmp.yml`), `lego.json` (100 point-light frames),
  generate/train configs;
- **not shipped:** the Lego training/validation H5 files and the best-epoch checkpoints.

**Environment:**
- WSL Ubuntu-22.04;
- the RNA venv with Python 3.10, torch 2.8.0+cu128, Lightning 2.5.5, and the official bpy 3.5.1
  custom-Cycles wheel;
- RTX 5080 (16 GB);
- Ryzen 9 9950X3D (32 threads visible), 31 GB WSL RAM.

**Compatibility adaptations.** Each one is isolated, and none changes a number. The upstream
checkout is unmodified (`git diff` empty).
1. **Checkpoint unpickling.** The released checkpoints reference `neumat.config.BaseNeuMatConfig`,
   the package's pre-release name. Without help they fail to load (`ModuleNotFoundError: neumat`).
   `rr_run.py` registers `neumat` → `rna` aliases; it is the same dataclass under a renamed
   package.
2. **JSON path in the render configs.** `lego_cmp.yml` names `config/render/jsons/lego.json`, but
   the file is `config/render/lego.json`. The workspace holds a byte-identical copy at the named
   path, so the yml is unchanged.
3. **Generator output names.** The released train and val generate configs share one output name;
   the upstream `--dataset_name` option separates them. The validation stem gains `_val` twice
   (`Lego_hq_val_val.h5`). The train yml's paths `more_data/training_data/11-2023/Lego_hq{,_val}.h5`
   are symlinks.
4. **Training-data placement** (resource adaptation, amendment 2). The released yml keeps the
   dataset on the GPU (`device: 'cuda'`). On 16 GB this ran at ~7.5 min/epoch (31 h projected)
   with WDDM spill. `device: 'cpu'` is the DataModule's default and the yml's own commented
   option; with it the data is pinned in host RAM and each one-view batch moves to the GPU.
5. **Resume** (amendment 3). The user paused training at epoch 64. It was resumed with Lightning
   `fit(ckpt_path=...)` (`rr_run.py` `RR_RESUME_CKPT`), restoring weights, optimiser, LR
   scheduler, epoch and step (first resumed val 31.65 vs 31.68 before the pause). The shuffle RNG
   is not restored, so the run is labelled "resumed".

**Documentation errors found:**
- The README render command uses `--checkpoint`/`--mode`. The real flags are
  `--config --model --reference`, and `--mode` is unused.
- The JSON path in the render configs is wrong (item 2 above).

## 3. RESULTS

### 3a. Released checkpoints through the native render path

Setup: `scripts/render.py`, `lego_cmp.yml` frames 13/19/26/33/68, point light, 1024².
- **Neural renders:** the 4096² feature pass is evaluated in PyTorch and downsampled.
- **References:** Cycles CPU 2048 spp, OIDN-denoised.
- **Time per frame:** ≈ 1 min neural, ≈ 1.5 min reference.
- **Metric:** LDR = the code's tonemap. PSNR is masked to the reference alpha.

| | masked PSNR (mean of 5) | full-frame PSNR incl. background | logL1 (masked) | paper Fig. 8 "full image" |
|---|---|---|---|---|
| HQ | **32.70** (frames 31.75–34.20) | 39.63 | 0.0104 | 32.44 |
| small | **27.68** (frames 24.56–30.22) | 34.61 | 0.0171 | 28.28 |

These are comparable to Fig. 8: different, unpublished views and conventions, with the same level
and the same ~5 dB HQ–small gap. Background pixels inflate full-frame PSNR by ~7 dB, so a masked
metric is the meaningful one.

### 3b. Validation PSNR (Table 1 metric) on the natively generated 40-view set

Set: 1024², 1024 spp, released `generate_Lego_val.yml`, unseeded generator. Path: the unchanged
Lightning `validate`, with the blur pinned to its end-of-schedule value (σ = min). Final weights.

| | view-mean PSNR | pooled | Table 1 | Δ | logged best |
|---|---|---|---|---|---|
| released HQ | **32.16** | 32.01 | 34.09 | −1.93 | 33.94 (authors' val) |
| released small | **27.96** | 27.56 | 28.77 | −0.81 | 28.95 (authors' val) |
| **reproduced HQ** (resumed run) | **31.99** | 31.83 | 34.09 | −2.10 | 32.11 (epoch 182, our val) |
| **reproduced small** | **27.97** | 27.63 | 28.77 | −0.80 | 28.06 (epoch 248, our val) |

Reproduced and released checkpoints agree view by view (`03_val_psnr_per_view.png`).

### 3c. N1 — validation-target noise

Setup:
- `RR_SEED=1234` makes the generator replayable.
- The same 8 views were generated at 1024 and at 4096 spp. Cameras, lights, positions and alpha
  are bit-identical between the two sets (`n1_check.json`).
- The 1024-spp target vs the 4096-spp target scores 35.9 dB. Because the 4096 render reuses the
  1024 samples as a prefix, this is an upper bound on the true noise ceiling.

| released | vs 1024-spp targets | vs 4096-spp targets | Δ (all 8 views) |
|---|---|---|---|
| HQ | 32.71 | **34.39** | **+1.68** (+1.13 … +2.26) |
| small | 28.60 | **29.22** | +0.62 (+0.29 … +0.80) |

The predeclared rule (≥ 1.0 dB) holds: target noise accounts for a material part of the HQ gap.
Against targets at the paper's data spp, both released models land at the Table 1 level (8 views
only). Noise affects the more accurate HQ model more, as expected.

### 3d. Training cost (our hardware; not comparable to the paper's A100)

| | wall-clock | per epoch | paper |
|---|---|---|---|
| dataset generation | 6.4 h (400 + 40 views, ≈ 55 s/view, CPU 32 threads) | — | "4–6 h on 96 cores" at 4096 spp |
| storage | 17.8 GB train + 1.85 GB val (+ 5.5 GB `.lquery`) | — | ≈ 17.2 GB |
| HQ training | **6.0 h** (65 epochs in segment 1, 7 799 s; 185 resumed epochs, 13 878 s) | ≈ 62–76 s; epochs 2–5 slowed by a concurrent CPU job | 1h13m |
| small training | **30.8 min** | 7.4 s | 31m17s |

**Interpretation of the HQ time.**
- The HQ run used 14.8 GB dedicated plus 10.9 GB shared GPU memory (Windows counters), so its
  wall-clock reflects 16-GB memory pressure, not compute.
- **The first, GPU-resident attempt** ran at 7.5 min/epoch; it was stopped and is not a result.
- **The small model** fits comfortably and matches the paper's time.

## 4. CLASSIFICATION (predeclared rules)

| route | verdict | why |
|---|---|---|
| released checkpoints, native 40-view val | **COMPARABLE** | HQ −1.93 (within 2 dB, not within 1), small −0.81 |
| native training, HQ | **COMPARABLE** | final within 0.17 dB of the released checkpoint on the same set; best 32.11 is −1.98 dB from Table 1 |
| native training, small | **REPRODUCED** | final = released (27.97 vs 27.96); best 28.06 is within 1 dB of Table 1 |
| Fig. 8 | **COMPARABLE** | protocol unpublished |
| performance (Fig. 6/7) | **BLOCKED** | production renderer not released |
| **overall** | **COMPARABLE** | the gap to Table 1 is common to released and reproduced models and attributed to 1024- vs 4096-spp validation targets (N1) |

## 5. OBSERVATION (`results/evaluation/35/`)

- **`01_lego_frame*`, `02_lego_crops`:** on the official asset the released HQ model is visually
  indistinguishable from the Cycles reference at full frame.
  - Residual errors are thin lines at stud edges and shadow boundaries.
  - The small model is softer, with larger errors in dark, occluded regions.
  - This is qualitatively unlike the Teaset C1 failure (worklog 33: invented glints and glow on
    near-mirror nickel).
- **`03`:** per-view PSNR of the reproduced and released models overlaps on all 40 views.
- **`04`:** the reproduced curves converge by ~60 k steps; the paper says ~80 k.

## 6. REMAINING DISCREPANCIES AND LIMITATIONS

- **The validation set is not the authors'.** The generator is unseeded and our set uses the
  released 1024 spp, not the paper's 4096 spp. The N1 evidence covers 8 views, not the full 40.
- **Training data also follows the released 1024 spp**, not the paper's 4096. The released
  checkpoints' quality is nonetheless matched on the same set, so the training-data spp does not
  appear to limit the model at this level. That conclusion is drawn from the match, not tested
  directly.
- **The HQ reproduction is one seed and a resumed run.** HQ wall-clock is resource-bound.
- **Fig. 8 cannot be pinned exactly.** The render reference is OIDN-denoised.
- **Performance claims are untestable** without the production renderer.

## 7. FINAL REPORT

1. **Figures/metrics selected:** Table 1 (the only Lego numbers with a code-defined metric), the
   released checkpoints' logs, and Fig. 8 (comparable). Fig. 5 and Fig. 6/7 are qualitative or
   blocked.
2. **Official artifacts:** checkpoints, scene, configs and frame JSON are available. The datasets
   and best checkpoints are not.
3. **Native RNA executed:** yes. Generation, training, validation and the deferred renderer all
   ran with the unchanged upstream code and the official custom Cycles. Five isolated adaptations
   (section 2).
4. **Pretrained Lego quality:** comparable (32.16 / 27.96 on 1024-spp targets). At the Table 1
   level against 4096-spp targets (34.39 / 29.22, 8 views).
5. **Native training:**
   - succeeded, and matches the released checkpoints (31.99 / 27.97 on the same set);
   - small: REPRODUCED; HQ: COMPARABLE.
6. **Paper vs reproduction table:** sections 3b, 3c and 4.
7. **Reviewer comparisons:** `results/evaluation/35/` (manifest with sha256).
8. **Remaining discrepancies:** section 6.
9. **Classification:** **comparable**. Quality is reproduced up to an attributed validation-target
   noise effect. Performance is blocked.
10. **Is RNA a credible substrate?** Yes, on the material class it was published for, and when
    evaluated against adequately converged targets.
    - The Teaset problems (worklogs 22–34) are explained by the scene/material regime (C1
      near-mirror Ni) and by noisy evaluation targets (C2), not by a broken RNA.
    - Future static gates should use converged references (≥ 4096 spp or 8× the training spp), and
      should state the aggregation and the checkpoint rule.

Architecture Batch 3 was not started.
