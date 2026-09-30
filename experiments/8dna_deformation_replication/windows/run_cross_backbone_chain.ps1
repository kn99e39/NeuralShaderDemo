# Evidence chain for the cross-backbone batch (run from a clean, committed tree).
#   pwsh windows/run_cross_backbone_chain.ps1
# Local steps run strictly serially: this GPU has 16 GiB and concurrent jobs
# slow each other down (an RNA dataset view takes 0.7 min alone and far longer
# while 8DNA trains).  Each step logs to results/8dna_replication/logs/ and a
# failing step stops the chain.  A step whose completion marker already exists
# (and, for datasets and feature buffers, is current for the protocol) is
# skipped, so the chain can be resumed after an interruption.
# run_rna_frozen_track.ps1 runs alongside and produces the frozen RNA x 8DNA
# comparison as soon as RNA T0 is trained; this chain covers everything else.
$track = 'chain'
. (Join-Path $PSScriptRoot 'chain_common.ps1')
Note "CHAIN START commit $(git -C $root rev-parse --short HEAD)"

# --- 1. GPU-bound Mitsuba work, serial ---------------------------------------
Step 'w21_reference_correction' 'w21_reference_correction/w21_reference_correction.json' @('w21_reference_correction.py')
# GT-only design: references, ROIs and the physical-signal gate for this regime.
# --require-pass stops the chain if the gate fails, so no neural work is done on
# a regime whose signal is not above the noise floors.
Step 'gt_design_common_light' 'gt_design/common_light/gt_states.json' `
    @('teaset_gt_states.py', '--protocol', 'protocol/teaset_common_light_gt_design.json', '--out', 'gt_design/common_light', '--require-pass')
Step 'frozen_8dna_common_light' 'frozen/teaset_common_light_8dna/frozen_eval.json' @('teaset_frozen_eval.py', '--protocol', $P)
foreach ($s in 'T0', 'T3') {
    foreach ($split in 'train', 'val') {
        Step "h5_${s}_$split" "rna_teaset/datasets/teaset_${s}_$split.h5" @('rna_bridge.py', 'h5', '--protocol', $P, '--state', $s, '--split', $split)
    }
}

# --- 2. Training --------------------------------------------------------------
# RNA trains on LabServer63 (no Mitsuba needed) while the Mitsuba-bound 8DNA
# refits run here. RNA T0 is launched by the frozen track as soon as its datasets
# exist; T3 queues behind it on the server's one GPU.
StartRnaOnServer 'T0'
StartRnaOnServer 'T3'
Step 'train_8dna_T0_retrain' 'refit/T0_retrain/last.ckpt' `
    @('train_8dna_state.py', '--state', 'T0', '--protocol', 'protocol/teaset_frozen_locked.json', '--device', '0',
      '--max_epochs', '30', '--seed', '9', '--log_path', "$res/refit", '--experiment_name', 'T0_retrain')
Step 'train_8dna_T3_refit' 'refit/T3_refit/last.ckpt' `
    @('train_8dna_state.py', '--state', 'T3', '--protocol', 'protocol/teaset_frozen_locked.json', '--device', '0',
      '--max_epochs', '30', '--seed', '9', '--log_path', "$res/refit", '--experiment_name', 'T3_refit')
# Normally produced by the frozen track already; regenerated here if not.
foreach ($s in 'T0', 'T1', 'T1b', 'T2', 'T3') {
    Step "features_$s" "rna_teaset/features/$s.npz" @('rna_bridge.py', 'features', '--protocol', $P, '--state', $s)
}
Step 'features_T0_B' 'rna_teaset/features/T0_B.npz' @('rna_bridge.py', 'features', '--protocol', $P, '--state', 'T0', '--light-seed', 'B')
WaitRnaOnServer 'T0'
WaitRnaOnServer 'T3'

# --- 3. Evaluation ------------------------------------------------------------
Step 'render_8dna_refits' 'refit/renders/renders.json' @('render_8dna_refits.py')
Step 'rna_frozen_render' 'rna_teaset/frozen/rna_render.json' @('render_rna_states.py', '--protocol', $P, '--part', 'frozen')
Step 'rna_refit_render' 'rna_teaset/frozen/rna_refit_render.json' @('render_rna_states.py', '--protocol', $P, '--part', 'refit')
Step 'cross_backbone_eval' 'cross_backbone/cross_backbone.json' @('cross_backbone_eval.py')
Note 'EVIDENCE DONE'

# --- 4. Static wavefront baselines (batch-1 correction, off the critical path) --
# Rendered with each scene's own Python integrator in wavefront mode, which is
# slow; they come last so they cannot delay the cross-backbone evidence.
Step 'static_seal_wavefront' 'static_baseline/seal_scene2_official_wavefront/baseline.json' `
    @('run_static_baseline.py', '--asset', 'seal', '--scene-fn', 'scene2', '--res', '256', '--spp', '256', '--ref-spp', '32768', '--ref-chunk', '32', '--tag', 'seal_scene2_official_wavefront')
Step 'static_teaset_wavefront' 'static_baseline/teaset_T0_512_wavefront/baseline.json' `
    @('run_static_baseline.py', '--asset', 'teaset', '--scene-fn', 'get_scene', '--res', '512', '--spp', '256', '--ref-spp', '2048', '--ref-chunk', '32', '--tag', 'teaset_T0_512_wavefront')
Note 'CHAIN DONE'
