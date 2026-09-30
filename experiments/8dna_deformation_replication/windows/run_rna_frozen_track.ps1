# RNA frozen track of the cross-backbone batch (run from a clean, committed tree,
# alongside run_cross_backbone_chain.ps1).
#   pwsh windows/run_rna_frozen_track.ps1
# The frozen RNA x 8DNA comparison needs only the T0 datasets, the RNA T0
# training on LabServer63 and a few minutes of local inference; the 8DNA refits
# (about 12 min per epoch here) are needed only for the refit controls. This
# track therefore produces the frozen comparison as soon as RNA T0 is trained,
# instead of after the refits. Its short GPU jobs (feature buffers, inference)
# overlap the main chain's local work; everything else waits on the server.
$track = 'rna-frozen'
. (Join-Path $PSScriptRoot 'chain_common.ps1')
Note "TRACK START commit $(git -C $root rev-parse --short HEAD)"

# Wait for the main chain to finish the T0 datasets.
while (-not ((Current 'rna_teaset/datasets/teaset_T0_train.h5') -and (Current 'rna_teaset/datasets/teaset_T0_val.h5'))) {
    Start-Sleep -Seconds 60
}
StartRnaOnServer 'T0'
foreach ($s in 'T0', 'T1', 'T1b', 'T2', 'T3') {
    Step "features_$s" "rna_teaset/features/$s.npz" @('rna_bridge.py', 'features', '--protocol', $P, '--state', $s)
}
Step 'features_T0_B' 'rna_teaset/features/T0_B.npz' @('rna_bridge.py', 'features', '--protocol', $P, '--state', 'T0', '--light-seed', 'B')
WaitRnaOnServer 'T0'
Step 'rna_frozen_render' 'rna_teaset/frozen/rna_render.json' @('render_rna_states.py', '--protocol', $P, '--part', 'frozen')
Step 'cross_backbone_frozen_eval' 'cross_backbone/cross_backbone_frozen.json' @('cross_backbone_eval.py', '--frozen-only')
Note 'TRACK DONE'
