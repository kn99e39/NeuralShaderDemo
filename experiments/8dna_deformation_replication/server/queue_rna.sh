#!/usr/bin/env bash
# Queue one RNA training on LabServer63: wait until no other RNA training is
# running on this server's single GPU, then run server/train_rna.sh <state>.
#   nohup setsid bash server/queue_rna.sh <T0|T3> &
set -uo pipefail
STATE=${1:?usage: queue_rna.sh <T0|T3>}
ROOT=${ROOT:-$HOME/NeuralShaderDemo}
LOG=$ROOT/results/8dna_replication/logs
mkdir -p "$LOG"
# A running training's command line ends in its state; queued launchers do not
# match this pattern, so two queued states cannot wait on each other.
while pgrep -f 'train_rna[.]sh [A-Za-z0-9]+$' > /dev/null; do sleep 60; done
exec bash "$(dirname "$0")/train_rna.sh" "$STATE" > "$LOG/rna_teaset_$STATE.log" 2>&1
