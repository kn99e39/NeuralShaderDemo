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
# The server steps need an OpenSSH client. This machine has no Windows OpenSSH
# (System32\OpenSSH is on PATH but holds no ssh.exe), so use Git's, which reads
# the same ~/.ssh/config (LabServer63 alias).
$gitSsh = 'C:\Program Files\Git\usr\bin'
$ssh = if (Test-Path "$gitSsh\ssh.exe") { "$gitSsh\ssh.exe" } else { (Get-Command ssh -ErrorAction Stop).Source }
$scp = if (Test-Path "$gitSsh\scp.exe") { "$gitSsh\scp.exe" } else { (Get-Command scp -ErrorAction Stop).Source }
function Note([string] $m) { "$(Get-Date -Format o) $m" | Tee-Object -Append $chain | Write-Host }
Note "CHAIN START commit $(git -C $root rev-parse --short HEAD)"

# Run a project script unless $done already exists and is current for this
# protocol. An existing file is not enough on its own: a dataset or render left
# by a superseded regime would otherwise be skipped over and silently reused.
function Current([string] $done) {
    $path = Join-Path $res $done
    if (-not (Test-Path $path)) { return $false }
    if ($done -notlike '*rna_teaset/datasets/*.h5' -and $done -notlike '*rna_teaset/features/*.npz') { return $true }
    $check = pwsh -NoProfile -File $run 'check_dataset_current.py' '--protocol' $P '--h5' $path 2>&1
    if ($LASTEXITCODE -eq 0) { return $true }
    Note "STALE $done ($check)"
    return $false
}

function Step([string] $name, [string] $done, [string[]] $argv) {
    if ($done -and (Current $done)) { Note "SKIP $name (have $done)"; return }
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
    # A resumed chain must not start a second copy of a run already on the server.
    $running = & $ssh -o BatchMode=yes LabServer63 "pgrep -f '[s]erver/train_rna.sh' >/dev/null && echo yes"
    if ($LASTEXITCODE -eq 255) { Note "FAIL LabServer63 unreachable"; exit 1 }
    if ($running -eq 'yes') { Note "SKIP rna_train launch (already running on LabServer63)"; return }
    & $ssh -o BatchMode=yes LabServer63 "mkdir -p ~/NeuralShaderDemo/results/8dna_replication/rna_teaset/datasets ~/NeuralShaderDemo/results/8dna_replication/logs ~/NeuralShaderDemo/experiments/8dna_deformation_replication/{configs,server}"
    foreach ($state in $todo) {
        Note "COPY datasets $state -> LabServer63"
        foreach ($split in 'train', 'val') {
            & $scp -o BatchMode=yes (Join-Path $res "rna_teaset/datasets/teaset_${state}_$split.h5") `
                "LabServer63:~/NeuralShaderDemo/results/8dna_replication/rna_teaset/datasets/"
            if ($LASTEXITCODE -ne 0) { Note "FAIL copy teaset_${state}_$split.h5"; exit 1 }
        }
        & $scp -o BatchMode=yes (Join-Path $root "experiments/8dna_deformation_replication/configs/rna_teaset_$state.yml") `
            "LabServer63:~/NeuralShaderDemo/experiments/8dna_deformation_replication/configs/"
        if ($LASTEXITCODE -ne 0) { Note "FAIL copy config $state"; exit 1 }
    }
    & $scp -o BatchMode=yes (Join-Path $PSScriptRoot "../server/train_rna.sh") `
        "LabServer63:~/NeuralShaderDemo/experiments/8dna_deformation_replication/server/"
    if ($LASTEXITCODE -ne 0) { Note "FAIL copy train_rna.sh"; exit 1 }
    # One detached shell runs the states in sequence (one server GPU); nohup +
    # setsid so it survives this ssh session closing.
    $seq = ($todo | ForEach-Object { "bash experiments/8dna_deformation_replication/server/train_rna.sh $_ > results/8dna_replication/logs/rna_teaset_$_.log 2>&1" }) -join '; '
    Note "LAUNCH rna_train $($todo -join ',') (LabServer63)"
    & $ssh -o BatchMode=yes LabServer63 "cd ~/NeuralShaderDemo && nohup setsid bash -c '$seq' > /dev/null 2>&1 < /dev/null & echo launched"
    if ($LASTEXITCODE -ne 0) { Note "FAIL launch rna_train"; exit 1 }
}

# Block until a server-side RNA run has written its .status file, then check it.
function WaitRnaOnServer([string] $state) {
    if (Test-Path (Join-Path $res "rna_teaset/ckpt/rna-teaset-$state-common-light")) { return }
    Note "WAIT rna_train_$state"
    $unreach = 0
    while ($true) {
        $status = & $ssh -o BatchMode=yes -o ConnectTimeout=30 LabServer63 "cat ~/NeuralShaderDemo/results/8dna_replication/logs/rna_teaset_$state.status 2>/dev/null"
        if ($LASTEXITCODE -eq 255) {
            # ssh itself failed (e.g. Tailscale SSH asking for re-authentication):
            # say so in the chain log instead of waiting silently.
            $unreach++
            if ($unreach -eq 3 -or $unreach % 30 -eq 0) { Note "WARN LabServer63 unreachable ($unreach polls) while waiting for rna_train_$state" }
        } else { $unreach = 0 }
        if ($status) { break }
        Start-Sleep -Seconds 120
    }
    Note "STATUS rna_train_$state $status"
    if ($status -notmatch 'exit=0') { Note "FAIL rna_train_$state"; exit 1 }
    Note "FETCH rna checkpoints $state"
    & $scp -o BatchMode=yes -r "LabServer63:~/NeuralShaderDemo/results/8dna_replication/rna_teaset/ckpt/rna-teaset-$state-common-light" `
        (Join-Path $res "rna_teaset/ckpt/")
    if ($LASTEXITCODE -ne 0) { Note "FAIL fetch rna checkpoints $state"; exit 1 }
}

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
# RNA inference buffers (area-light samples) at the evaluation camera. Only RNA
# inference reads them, so they run after the local training, not before it.
foreach ($s in 'T0', 'T1', 'T1b', 'T2', 'T3') {
    Step "features_$s" "rna_teaset/features/$s.npz" @('rna_bridge.py', 'features', '--protocol', $P, '--state', $s)
}
Step 'features_T0_B' 'rna_teaset/features/T0_B.npz' @('rna_bridge.py', 'features', '--protocol', $P, '--state', 'T0', '--light-seed', 'B')
WaitRnaOnServer 'T0'
WaitRnaOnServer 'T3'

# --- 3. Evaluation ------------------------------------------------------------
Step 'render_8dna_refits' 'refit/renders/renders.json' @('render_8dna_refits.py')
Step 'rna_frozen_render' 'rna_teaset/frozen/rna_render.json' @('render_rna_states.py', '--protocol', $P)
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
