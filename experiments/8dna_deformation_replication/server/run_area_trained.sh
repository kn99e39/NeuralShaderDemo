#!/usr/bin/env bash
# Post-hoc area-trained RNA sensitivity (protocol/teaset_rna_area_trained_sensitivity.json)
# on LabServer63: generate the T0 datasets on this server's GPU, then train RNA.
#   setsid -f bash server/run_area_trained.sh    (from a worktree checked out at the run's commit)
# Needs the 595.71.05 OptiX user-space libraries in ~/nvlibs/optix595 (this
# server's system libnvoptix/rtcore/gpucomp are 570.195.03 against a 595.71.05
# kernel driver) and ninja on PATH for the upstream 8DNA extension build.
set -uo pipefail
WT=$(cd "$(dirname "$0")/../../.." && pwd)
MAIN=${MAIN:-$HOME/NeuralShaderDemo}
export LD_LIBRARY_PATH=$HOME/nvlibs/optix595
export PATH=$MAIN/external/8dna26/.venv/bin:$PATH
PY=$MAIN/external/8dna26/.venv/bin/python
P=protocol/teaset_rna_area_trained_sensitivity.json
OUT=$WT/results/8dna_replication/rna_teaset/datasets_area_trained
LOG=$WT/results/8dna_replication/logs
STATUS=$LOG/area_trained_server.status
mkdir -p "$LOG"
rm -f "$STATUS"
cd "$WT/experiments/8dna_deformation_replication"
echo "START $(date -Is) commit $(git rev-parse --short HEAD)" > "$LOG/area_trained_server.log"
for split in val train; do
    s=$(date +%s)
    "$PY" -X faulthandler rna_bridge.py h5 --protocol "$P" --state T0 --split "$split" > "$LOG/h5_area_T0_$split.log" 2>&1
    # The cameras file is written only after the H5 is complete; trust it, not the
    # exit code (interpreter teardown can crash after the outputs are written).
    if [ ! -f "$OUT/teaset_T0_$split.cameras.json" ]; then echo "exit=1 stage=h5_$split" > "$STATUS"; exit 1; fi
    echo "h5_$split done in $(( $(date +%s) - s )) s" >> "$LOG/area_trained_server.log"
done
# RNA's configs read datasets relative to the main clone's RNA checkout.
mkdir -p "$MAIN/results/8dna_replication/rna_teaset" "$MAIN/experiments/8dna_deformation_replication/configs"
ln -sfn "$OUT" "$MAIN/results/8dna_replication/rna_teaset/datasets_area_trained"
cp configs/rna_teaset_T0_area_trained.yml "$MAIN/experiments/8dna_deformation_replication/configs/"
bash server/train_rna.sh T0_area_trained > "$MAIN/results/8dna_replication/logs/rna_teaset_T0_area_trained.log" 2>&1
rc=$?
echo "exit=$rc stage=train end=$(date -Is)" > "$STATUS"
exit $rc
