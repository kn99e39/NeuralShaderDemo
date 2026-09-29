# Evidence chain for the cross-backbone batch (run from a clean, committed tree).
#   pwsh windows/run_cross_backbone_chain.ps1
# Each step logs to results/8dna_replication/logs/; a failing step stops the chain.
$ErrorActionPreference = 'Stop'
$root = Resolve-Path "$PSScriptRoot/../../.."
$run = Join-Path $PSScriptRoot 'run.ps1'
$log = Join-Path $root 'results/8dna_replication/logs'
$P = 'protocol/teaset_cross_backbone_locked.json'
New-Item -ItemType Directory -Force $log | Out-Null
git -C $root rev-parse HEAD | Out-File (Join-Path $log 'cross_backbone_chain_commit.txt')

function Step([string] $name, [string[]] $argv) {
    "$(Get-Date -Format o) START $name" | Add-Content (Join-Path $log 'cross_backbone_chain.log')
    pwsh -NoProfile -File $run @argv *> (Join-Path $log "$name.log")
    if ($LASTEXITCODE -ne 0) { "$(Get-Date -Format o) FAIL $name ($LASTEXITCODE)" | Add-Content (Join-Path $log 'cross_backbone_chain.log'); exit 1 }
    "$(Get-Date -Format o) END $name" | Add-Content (Join-Path $log 'cross_backbone_chain.log')
}

Step 'w21_reference_correction' @('w21_reference_correction.py')
Step 'h5_T0_train' @('rna_bridge.py', 'h5', '--protocol', $P, '--state', 'T0', '--split', 'train')
Step 'h5_T0_val' @('rna_bridge.py', 'h5', '--protocol', $P, '--state', 'T0', '--split', 'val')

# RNA T0 training: detached inside WSL (AGENTS.md "Running long jobs in local WSL")
$wslRoot = '/mnt/c/Projects/NeuralShaderDemo'
wsl.exe -d Ubuntu-22.04 -u root --exec bash "$wslRoot/experiments/dynamic_transport_failure/scripts/wsl_run_detached.sh" `
    rna_teaset_T0 "$wslRoot/results/8dna_replication/logs/wsl" "$wslRoot/external/relightable-neural-assets" -- `
    .venv/bin/python scripts/train.py --config ../../experiments/8dna_deformation_replication/configs/rna_teaset_T0.yml `
    --checkpoint_dir ../../results/8dna_replication/rna_teaset/ckpt --seed 0
"$(Get-Date -Format o) LAUNCHED rna_teaset_T0 (WSL detached)" | Add-Content (Join-Path $log 'cross_backbone_chain.log')

Step 'h5_T3_train' @('rna_bridge.py', 'h5', '--protocol', $P, '--state', 'T3', '--split', 'train')
Step 'h5_T3_val' @('rna_bridge.py', 'h5', '--protocol', $P, '--state', 'T3', '--split', 'val')
foreach ($s in 'T0', 'T1', 'T1b', 'T2', 'T3') { Step "features_$s" @('rna_bridge.py', 'features', '--protocol', $P, '--state', $s) }
Step 'frozen_8dna_common_light' @('teaset_frozen_eval.py', '--protocol', $P)
Step 'static_seal_wavefront' @('run_static_baseline.py', '--asset', 'seal', '--scene-fn', 'scene2', '--res', '256', '--spp', '256', '--ref-spp', '32768', '--ref-chunk', '64', '--tag', 'seal_scene2_official_wavefront')
Step 'static_teaset_wavefront' @('run_static_baseline.py', '--asset', 'teaset', '--scene-fn', 'get_scene', '--res', '512', '--spp', '256', '--ref-spp', '2048', '--ref-chunk', '16', '--tag', 'teaset_T0_512_wavefront')
"$(Get-Date -Format o) CHAIN DONE" | Add-Content (Join-Path $log 'cross_backbone_chain.log')
