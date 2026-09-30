# Shared by the cross-backbone chain and the RNA frozen track (dot-sourced).
# Defines paths, Note/Current/Step, and the LabServer63 RNA launch and wait.
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
$timeoutExe = "$gitSsh\timeout.exe"
$remote = '~/NeuralShaderDemo'
if (-not $track) { $track = 'chain' }
function Note([string] $m) { "$(Get-Date -Format o) [$track] $m" | Tee-Object -Append $chain | Write-Host }

# Run a project script unless $done already exists and is current for this
# protocol. An existing file is not enough on its own: a dataset or render left
# by a superseded regime would otherwise be skipped over and silently reused.
function Current([string] $done) {
    $path = Join-Path $res $done
    if (-not (Test-Path $path)) { return $false }
    if ($done -notlike '*rna_teaset/datasets/*.h5' -and $done -notlike '*rna_teaset/features/*.npz') { return $true }
    $check = pwsh -NoProfile -File $run 'check_dataset_current.py' '--protocol' $P '--h5' $path 2>&1 | Select-Object -Last 1
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

function Remote([string] $cmd) {
    $out = & $ssh -o BatchMode=yes -o ConnectTimeout=30 LabServer63 $cmd
    if ($LASTEXITCODE -eq 255) { Note "FAIL LabServer63 unreachable"; exit 1 }
    return $out
}

# Copy one state's datasets to LabServer63 and queue its RNA training there,
# detached. RNA training is pure PyTorch over the H5 files, so it does not need
# Mitsuba and is unaffected by that server's mismatched OptiX libraries; it runs
# in parallel with the local Mitsuba-bound work. References and evaluation
# renders stay on the 5080. The server has one GPU, so server/queue_rna.sh waits
# for any other RNA training to finish before starting this one.
function StartRnaOnServer([string] $state) {
    if (Test-Path (Join-Path $res "rna_teaset/ckpt/rna-teaset-$state-common-light")) { Note "SKIP rna_train_$state (have checkpoints)"; return }
    foreach ($split in 'train', 'val') {
        if (-not (Current "rna_teaset/datasets/teaset_${state}_$split.h5")) { Note "FAIL rna_train_$state needs a current teaset_${state}_$split.h5"; exit 1 }
    }
    # A resumed chain must not start a second copy of a run already queued or running.
    $busy = Remote "pgrep -f 'queue_rna[.]sh $state`$|train_rna[.]sh $state`$' >/dev/null && echo yes; cat $remote/results/8dna_replication/logs/rna_teaset_$state.status 2>/dev/null"
    if ($busy -contains 'yes') { Note "SKIP rna_train_$state launch (already queued or running on LabServer63)"; return }
    if ($busy -match 'exit=0') { Note "SKIP rna_train_$state launch (finished on LabServer63)"; return }
    Remote "mkdir -p $remote/results/8dna_replication/rna_teaset/datasets $remote/results/8dna_replication/logs $remote/experiments/8dna_deformation_replication/{configs,server}" | Out-Null
    Note "COPY datasets $state -> LabServer63"
    foreach ($split in 'train', 'val') {
        & $scp -o BatchMode=yes (Join-Path $res "rna_teaset/datasets/teaset_${state}_$split.h5") "LabServer63:$remote/results/8dna_replication/rna_teaset/datasets/"
        if ($LASTEXITCODE -ne 0) { Note "FAIL copy teaset_${state}_$split.h5"; exit 1 }
    }
    & $scp -o BatchMode=yes (Join-Path $root "experiments/8dna_deformation_replication/configs/rna_teaset_$state.yml") "LabServer63:$remote/experiments/8dna_deformation_replication/configs/"
    if ($LASTEXITCODE -ne 0) { Note "FAIL copy config $state"; exit 1 }
    & $scp -o BatchMode=yes (Join-Path $PSScriptRoot '../server/train_rna.sh') (Join-Path $PSScriptRoot '../server/queue_rna.sh') "LabServer63:$remote/experiments/8dna_deformation_replication/server/"
    if ($LASTEXITCODE -ne 0) { Note "FAIL copy server scripts"; exit 1 }
    # setsid -f forks into a new session and returns at once. With the earlier
    # 'nohup setsid ... &' the ssh call made from PowerShell did not return until
    # the remote job ended (RNA T0: 34 min), which would block the local chain.
    Remote "cd $remote && setsid -f bash experiments/8dna_deformation_replication/server/queue_rna.sh $state > /dev/null 2>&1 < /dev/null; echo launched" | Out-Null
    Note "LAUNCH rna_train_$state (LabServer63)"
}

# Block until a server-side RNA run has written its .status file, then fetch it.
function WaitRnaOnServer([string] $state) {
    if (Test-Path (Join-Path $res "rna_teaset/ckpt/rna-teaset-$state-common-light")) { return }
    Note "WAIT rna_train_$state"
    $unreach = 0
    while ($true) {
        # Bounded: a Tailscale SSH re-authentication check keeps ssh waiting on
        # the server side indefinitely instead of failing, so cap each poll and
        # put the authentication link it printed into the chain log.
        $errFile = Join-Path $log "ssh_poll_$state.err"
        $status = & $timeoutExe 90 $ssh -o BatchMode=yes -o ConnectTimeout=30 LabServer63 "cat $remote/results/8dna_replication/logs/rna_teaset_$state.status 2>/dev/null" 2> $errFile
        if ($LASTEXITCODE -eq 255 -or $LASTEXITCODE -eq 124) {
            $unreach++
            $auth = Select-String -Path $errFile -Pattern 'https://login\.tailscale\.com/\S+' | Select-Object -Last 1
            if ($auth) { Note "WARN Tailscale SSH needs re-authentication for LabServer63: $($auth.Matches[0].Value)" }
            elseif ($unreach -eq 3 -or $unreach % 30 -eq 0) { Note "WARN LabServer63 unreachable ($unreach polls) while waiting for rna_train_$state" }
        } else { $unreach = 0 }
        if ($status) { break }
        Start-Sleep -Seconds 120
    }
    Note "STATUS rna_train_$state $status"
    if ($status -notmatch 'exit=0') { Note "FAIL rna_train_$state"; exit 1 }
    Note "FETCH rna checkpoints $state"
    $tmp = Join-Path $res "rna_teaset/ckpt/.fetch-$state"
    if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
    New-Item -ItemType Directory -Force $tmp | Out-Null
    & $scp -o BatchMode=yes -r "LabServer63:$remote/results/8dna_replication/rna_teaset/ckpt/rna-teaset-$state-common-light" $tmp
    if ($LASTEXITCODE -ne 0) { Note "FAIL fetch rna checkpoints $state"; exit 1 }
    # rename only once complete, so the directory's existence means "fetched"
    Move-Item (Join-Path $tmp "rna-teaset-$state-common-light") (Join-Path $res "rna_teaset/ckpt/")
    Remove-Item -Force $tmp
}
