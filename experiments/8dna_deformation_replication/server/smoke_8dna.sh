#!/usr/bin/env bash
# 8DNA training smoke test on LabServer63: one epoch of the upstream trainer on
# teaset T0 (same state, seed and wrapper as the 5080 T0 retrain), queued behind
# the area-trained RNA run on this server's single GPU. Feasibility only: does
# 8DNA train here with the 595.71.05 OptiX user-space libraries, and how fast.
#   setsid -f bash server/smoke_8dna.sh    (from a worktree at the run's commit)
set -uo pipefail
WT=$(cd "$(dirname "$0")/../../.." && pwd)
MAIN=${MAIN:-$HOME/NeuralShaderDemo}
export LD_LIBRARY_PATH=$HOME/nvlibs/optix595
export PATH=$MAIN/external/8dna26/.venv/bin:$PATH
PY=$MAIN/external/8dna26/.venv/bin/python
LOG=$WT/results/8dna_replication/logs
STATUS=$LOG/smoke_8dna_server.status
OUT=$WT/results/8dna_replication/server_smoke
mkdir -p "$LOG"
rm -f "$STATUS"
# QUEUE=0 starts at once (the RNA run moved to the 5080, so nothing to wait for)
if [ "${QUEUE:-1}" = 1 ]; then
    while [ ! -f "$LOG/area_trained_server.status" ]; do sleep 60; done
fi
while pgrep -f 'train_rna[.]sh [A-Za-z0-9_]+$' > /dev/null; do sleep 60; done
cd "$WT/experiments/8dna_deformation_replication"
s=$(date +%s)
"$PY" -X faulthandler train_8dna_state.py --state T0 --protocol protocol/teaset_frozen_locked.json --device 0 \
    --max_epochs 1 --seed 9 --log_path "$OUT" --experiment_name T0_smoke_1ep > "$LOG/smoke_8dna_server.log" 2>&1
rc=$?
# judge by the checkpoint, not the exit code (teardown can crash after outputs are written)
ck=no; [ -f "$OUT/T0_smoke_1ep/last.ckpt" ] && ck=yes
echo "exit=$rc ckpt=$ck seconds=$(( $(date +%s) - s )) commit=$(git rev-parse --short HEAD) end=$(date -Is)" > "$STATUS"
