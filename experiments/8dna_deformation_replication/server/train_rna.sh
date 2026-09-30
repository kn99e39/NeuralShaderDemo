#!/usr/bin/env bash
# Train RNA on LabServer63 (RTX 3080 Ti) for one locked teaset state.
#   bash server/train_rna.sh <T0|T3>
#
# RNA training is pure PyTorch over the H5 datasets, so it does not need
# Mitsuba and is unaffected by this server's mismatched OptiX libraries (see
# ../../docs/worklogs/22...). The datasets are produced on the RTX 5080 by
# rna_bridge.py and copied here; the reference and evaluation renders stay on
# the 5080, so no reference is ever split across machines.
#
# Writes <ckpt_dir>/rna-teaset-<state>-common-light/version_*/checkpoints/ and,
# on exit, a .status file the caller polls.
set -uo pipefail
STATE=${1:?usage: train_rna.sh <T0|T3>}
ROOT=${ROOT:-$HOME/NeuralShaderDemo}
RNA=$ROOT/external/relightable-neural-assets
CFG=$ROOT/experiments/8dna_deformation_replication/configs/rna_teaset_$STATE.yml
CKPT=$ROOT/results/8dna_replication/rna_teaset/ckpt
LOG=$ROOT/results/8dna_replication/logs
STATUS=$LOG/rna_teaset_$STATE.status

mkdir -p "$CKPT" "$LOG"
rm -f "$STATUS"
cd "$RNA"
echo "START $(date -Is) state=$STATE host=$(hostname)" >&2
.venv/bin/python scripts/train.py --config "$CFG" --checkpoint_dir "$CKPT" --seed 0
rc=$?
echo "exit=$rc end=$(date -Is)" > "$STATUS"
exit $rc
