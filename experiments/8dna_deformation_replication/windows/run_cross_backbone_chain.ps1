# Evidence chain for the cross-backbone batch (run from a clean, committed tree).
#   pwsh windows/run_cross_backbone_chain.ps1
# Steps run strictly serially: this GPU has 16 GiB and concurrent jobs slow each
# other down (an RNA dataset view takes 1.4 min alone and ~10 min while 8DNA
# trains).  Each step logs to results/8dna_replication/logs/ and a failing step
# stops the chain.  A step whose completion marker already exists is skipped, so
# the chain can be resumed after an interruption.
$ErrorActionPreference = 'Stop'
$root = Resolve-Path "$PSScriptRoot/../../.."
$run = Join-Path $PSScriptRoot 'run.ps1'
$log = Join-Path $root 'results/8dna_replication/logs'
$res = Join-Path $root 'results/8dna_replication'
$P = 'protocol/teaset_cross_backbone_locked.json'
$wslRoot = '/mnt/c/Projects/NeuralShaderDemo'
New-Item -ItemType Directory -Force $log | Out-Null
$chain = Join-Path $log 'cross_backbone_chain.log'
function Note([string] $m) { "$(Get-Date -Format o) $m" | Tee-Object -Append $chain | Write-Host }
Note "CHAIN START commit $(git -C $root rev-parse --short HEAD)"

# Run a project script unless $done already exists.
function Step([string] $name, [string] $done, [string[]] $argv) {
    if ($done -and (Test-Path (Join-Path $res $done))) { Note "SKIP $name (have $done)"; return }
    Note "START $name"
    pwsh -NoProfile -File $run @argv *> (Join-Path $log "$name.log")
    if ($LASTEXITCODE -ne 0) { Note "FAIL $name ($LASTEXITCODE)"; exit 1 }
    Note "END $name"
}

# Wait for a detached WSL job, then fail the chain if it did not exit 0.
function WaitWsl([string] $name, [string] $statusFile) {
    Note "WAIT $name"
    while (-not (Test-Path $statusFile)) { Start-Sleep -Seconds 60 }
    $status = (Get-Content $statusFile -Raw).Trim()
    Note "STATUS $name $status"
    if ($status -notmatch 'exit=0') { Note "FAIL $name"; exit 1 }
}

# Launch an RNA training run detached inside WSL (AGENTS.md "Running long jobs in local WSL").
function TrainRna([string] $state) {
    $status = Join-Path $res "logs/wsl/rna_teaset_$state.status"
    if (Test-Path (Join-Path $res "rna_teaset/ckpt/rna-teaset-$state-common-light")) { Note "SKIP rna_train_$state"; return }
    Remove-Item $status -ErrorAction SilentlyContinue
    Note "LAUNCH rna_train_$state"
    wsl.exe -d Ubuntu-22.04 -u root --exec bash "$wslRoot/experiments/dynamic_transport_failure/scripts/wsl_run_detached.sh" `
        "rna_teaset_$state" "$wslRoot/results/8dna_replication/logs/wsl" "$wslRoot/external/relightable-neural-assets" -- `
        .venv/bin/python scripts/train.py --config "../../experiments/8dna_deformation_replication/configs/rna_teaset_$state.yml" `
        --checkpoint_dir ../../results/8dna_replication/rna_teaset/ckpt --seed 0
    if ($LASTEXITCODE -ne 0) { Note "FAIL launch rna_train_$state"; exit 1 }
    WaitWsl "rna_train_$state" $status
}

# --- 1. GPU-bound Mitsuba work, serial ---------------------------------------
Step 'w21_reference_correction' 'w21_reference_correction/w21_reference_correction.json' @('w21_reference_correction.py')
Step 'frozen_8dna_common_light' 'frozen/teaset_common_light_8dna/frozen_eval.json' @('teaset_frozen_eval.py', '--protocol', $P)
foreach ($s in 'T0', 'T1', 'T1b', 'T2', 'T3') {
    Step "features_$s" "rna_teaset/features/$s.npz" @('rna_bridge.py', 'features', '--protocol', $P, '--state', $s)
}
foreach ($s in 'T0', 'T3') {
    foreach ($split in 'train', 'val') {
        Step "h5_${s}_$split" "rna_teaset/datasets/teaset_${s}_$split.h5" @('rna_bridge.py', 'h5', '--protocol', $P, '--state', $s, '--split', $split)
    }
}
Step 'static_seal_wavefront' 'static_baseline/seal_scene2_official_wavefront/baseline.json' `
    @('run_static_baseline.py', '--asset', 'seal', '--scene-fn', 'scene2', '--res', '256', '--spp', '256', '--ref-spp', '32768', '--ref-chunk', '32', '--tag', 'seal_scene2_official_wavefront')
Step 'static_teaset_wavefront' 'static_baseline/teaset_T0_512_wavefront/baseline.json' `
    @('run_static_baseline.py', '--asset', 'teaset', '--scene-fn', 'get_scene', '--res', '512', '--spp', '256', '--ref-spp', '2048', '--ref-chunk', '32', '--tag', 'teaset_T0_512_wavefront')

# --- 2. Training, serial ------------------------------------------------------
TrainRna 'T0'
Step 'train_8dna_T0_retrain' 'refit/T0_retrain/last.ckpt' `
    @('train_8dna_state.py', '--state', 'T0', '--protocol', 'protocol/teaset_frozen_locked.json', '--device', '0',
      '--max_epochs', '30', '--seed', '9', '--log_path', "$res/refit", '--experiment_name', 'T0_retrain')
Step 'train_8dna_T3_refit' 'refit/T3_refit/last.ckpt' `
    @('train_8dna_state.py', '--state', 'T3', '--protocol', 'protocol/teaset_frozen_locked.json', '--device', '0',
      '--max_epochs', '30', '--seed', '9', '--log_path', "$res/refit", '--experiment_name', 'T3_refit')
TrainRna 'T3'

# --- 3. Evaluation ------------------------------------------------------------
Step 'render_8dna_refits' 'refit/renders/renders.json' @('render_8dna_refits.py')
Step 'rna_frozen_render' 'rna_teaset/frozen/rna_render.json' @('render_rna_states.py', '--protocol', $P)
Step 'cross_backbone_eval' 'cross_backbone/cross_backbone.json' @('cross_backbone_eval.py')
Note 'CHAIN DONE'
