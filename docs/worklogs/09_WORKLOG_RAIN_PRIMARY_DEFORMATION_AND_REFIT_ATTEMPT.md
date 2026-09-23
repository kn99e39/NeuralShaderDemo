# Worklog — Rain Primary Deformation and F3 Refit Attempt (2026-09-23)

## Scope

This session completed the fixed-camera, fixed-light Rain F1--F3 frozen-correspondence renders after the already-recorded F0 primary baseline. It also prepared a same-architecture F3 refit corpus because the deformation error was localized and materially larger than F0. Results here are limited to the stated fixed front-primary regime.

## What was measured

Rain uses the corrected `FLOAT_COLOR` canonical-position AOV adapter and the validation-best static checkpoint `epoch=233-val_psnr=30.83dB.ckpt`. The fixed primary camera/light is `results/batch1_hq_dynamic_failure/cloth/deformation/rain_primary_fixed_camera_light.json`. F1, F2 and F3 each produced current-geometry feature buffers, canonical-position capture, RNA output, physical reference output, and copied direct-visibility diagnostics under `results/batch1_hq_dynamic_failure/cloth/deformation/primary/`.

The full-frame PSNR fell from 34.52 dB at F0 to 31.68, 31.45 and 31.40 dB at F1, F2 and F3. The effect is spatially selective: the scarf-facing ROI fell from 31.88 to 27.39, 27.08 and 27.01 dB, while the tail-fold ROI fell from 34.27 to 27.54, 27.19 and 27.10 dB. The non-target lower-top ROI remained approximately stable (31.86, 31.70, 31.89 and 31.89 dB). Collar-cavity quality was similarly stable near 30.1 dB.

The F1 and F3 preserved diagnostic `diffuse_direct` masks differ on 354 of 262,144 pixels (0.135%). More importantly, the frozen renderer trace shows each state captured its own feature pass and current direct-visibility buffer before inference. The measured degradation therefore is not attributable to reusing stale direct visibility; it remains compatible with the frozen canonical lookup/radiance representation failing on the changed scarf configuration.

## Refit attempt and limit

A 200-view F3 training corpus (four independently seeded 50-view shards) and a separately seeded 40-view validation corpus were rendered at the same 512 resolution and 256 reference samples as the static baseline. Each shard and the merged training corpus passed `validate_rna_h5.py`. The generated data are under ignored `results/` paths and are not Git artifacts; individual checkpoint files, if produced, would remain below 500 MB.

The refit config preserves the static TriPlane/MLP architecture and optimization schedule. An omitted `data_interface: "NeuralSurfaceDataModule"` declaration was repaired before the final launch. However, full refit training and even isolated DataModule initialization repeatedly caused the WSL runtime to restart before any checkpoint was written. CUDA remained available after each restart. This is an execution-environment failure, not evidence that a current-geometry refit succeeds or fails; no refit result is claimed.

## Einar limitation

Einar static training completed at `epoch=249-val_psnr=24.79dB`, but its freshly audited canonical-position identity path did not pass: the P0 comparison gave p99 absolute error 0.2637 (far larger than the Rain control residual). Its P0 armature evaluation must be incorporated into the canonical attribute construction before any Einar frozen-deformation result can be interpreted. No Einar deformation verdict is made in this session.

## Interim interpretation

For Rain in the fixed primary regime, the evidence weakens the proposition that a static frozen RNA representation maintains deformation-invariant high-quality shading: degradation grows in the declared scarf interaction regions while nearby non-target material remains stable, and the direct-visibility selector is current. This is not a global claim across all assets or pose families, and it is not strengthened by a refit result because the refit could not complete.

## Source changes

Added reproducible F3 refit generation/training configs under `experiments/dynamic_transport_failure/configs/rain_f3_refit_*.yml`. The official Adobe RNA source remains untouched.