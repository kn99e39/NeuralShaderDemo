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

# Copy a state's datasets to LabServer63 and start RNA training there, detached.
# RNA training is pure PyTorch over the H5 files, so it does not need Mitsuba and
# is unaffected by that server's mismatched OptiX libraries; it therefore runs in
# parallel with the local Mitsuba-bound work instead of competing for this GPU.
# References and evaluation renders stay on the 5080.
function StartRnaOnServer([string[]] $states) {
    $todo = @($states | Where-Object { -not (Test-Path (Join-Path $res "rna_teaset/ckpt/rna-teaset-$_-common-light")) })
    if (-not $todo) { Note "SKIP rna_train (have checkpoints)"; return }
    ssh -o BatchMode=yes LabServer63 "mkdir -p ~/NeuralShaderDemo/results/8dna_replication/rna_teaset/datasets ~/NeuralShaderDemo/results/8dna_replication/logs ~/NeuralShaderDemo/experiments/8dna_deformation_replication/{configs,server}"
    foreach ($state in $todo) {
        Note "COPY datasets $state -> LabServer63"
        foreach ($split in 'train', 'val') {
            scp -o BatchMode=yes (Join-Path $res "rna_teaset/datasets/teaset_${state}_$split.h5") `
                "LabServer63:~/NeuralShaderDemo/results/8dna_replication/rna_teaset/datasets/"
            if ($LASTEXITCODE -ne 0) { Note "FAIL copy teaset_${state}_$split.h5"; exit 1 }
        }
        scp -o BatchMode=yes (Join-Path $root "experiments/8dna_deformation_replication/configs/rna_teaset_$state.yml") `
            "LabServer63:~/NeuralShaderDemo/experiments/8dna_deformation_replication/configs/"
        if ($LASTEXITCODE -ne 0) { Note "FAIL copy config $state"; exit 1 }
    }
    scp -o BatchMode=yes (Join-Path $PSScriptRoot "../server/train_rna.sh") `
        "LabServer63:~/NeuralShaderDemo/experiments/8dna_deformation_replication/server/"
    if ($LASTEXITCODE -ne 0) { Note "FAIL copy train_rna.sh"; exit 1 }
    # One detached shell runs the states in sequence (one server GPU); nohup +
    # setsid so it survives this ssh session closing.
    $seq = ($todo | ForEach-Object { "bash experiments/8dna_deformation_replication/server/train_rna.sh $_ > results/8dna_replication/logs/rna_teaset_$_.log 2>&1" }) -join '; '
    Note "LAUNCH rna_train $($todo -join ',') (LabServer63)"
    ssh -o BatchMode=yes LabServer63 "cd ~/NeuralShaderDemo && nohup setsid bash -c '$seq' > /dev/null 2>&1 < /dev/null & echo launched"
    if ($LASTEXITCODE -ne 0) { Note "FAIL launch rna_train"; exit 1 }
}

# Block until a server-side RNA run has written its .status file, then check it.
function WaitRnaOnServer([string] $state) {
    if (Test-Path (Join-Path $res "rna_teaset/ckpt/rna-teaset-$state-common-light")) { return }
    Note "WAIT rna_train_$state"
    while ($true) {
        $status = ssh -o BatchMode=yes LabServer63 "cat ~/NeuralShaderDemo/results/8dna_replication/logs/rna_teaset_$state.status 2>/dev/null"
        if ($status) { break }
        Start-Sleep -Seconds 120
    }
    Note "STATUS rna_train_$state $status"
    if ($status -notmatch 'exit=0') { Note "FAIL rna_train_$state"; exit 1 }
    Note "FETCH rna checkpoints $state"
    scp -o BatchMode=yes -r "LabServer63:~/NeuralShaderDemo/results/8dna_replication/rna_teaset/ckpt/rna-teaset-$state-common-light" `
        (Join-Path $res "rna_teaset/ckpt/")
    if ($LASTEXITCODE -ne 0) { Note "FAIL fetch rna checkpoints $state"; exit 1 }
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

# --- 2. Training --------------------------------------------------------------
# RNA trains on LabServer63 (no Mitsuba needed) while the Mitsuba-bound 8DNA
# refits run here, so the two do not compete for this GPU.
# One server GPU, so the two RNA runs go one after the other in a single
# detached shell; both still overlap the local 8DNA training.
StartRnaOnServer 'T0' 'T3'
Step 'train_8dna_T0_retrain' 'refit/T0_retrain/last.ckpt' `
    @('train_8dna_state.py', '--state', 'T0', '--protocol', 'protocol/teaset_frozen_locked.json', '--device', '0',
      '--max_epochs', '30', '--seed', '9', '--log_path', "$res/refit", '--experiment_name', 'T0_retrain')
Step 'train_8dna_T3_refit' 'refit/T3_refit/last.ckpt' `
    @('train_8dna_state.py', '--state', 'T3', '--protocol', 'protocol/teaset_frozen_locked.json', '--device', '0',
      '--max_epochs', '30', '--seed', '9', '--log_path', "$res/refit", '--experiment_name', 'T3_refit')
WaitRnaOnServer 'T0'
WaitRnaOnServer 'T3'

# --- 3. Evaluation ------------------------------------------------------------
Step 'render_8dna_refits' 'refit/renders/renders.json' @('render_8dna_refits.py')
Step 'rna_frozen_render' 'rna_teaset/frozen/rna_render.json' @('render_rna_states.py', '--protocol', $P)
Step 'cross_backbone_eval' 'cross_backbone/cross_backbone.json' @('cross_backbone_eval.py')
Note 'CHAIN DONE'
