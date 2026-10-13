#!/usr/bin/env bash
# LabServer63: generate the C2 training set (worklog-23 OptiX libraries), then run the C2 chain.
#   setsid -f bash rso_server_c2.sh <run dir>      (the C2 validation set must already be in <run dir>/datasets_c2)
set -u
R=$1
E=$(cd "$(dirname "$0")" && pwd)
REPL=$(cd "$E/../8dna_deformation_replication" && pwd)
MI_PY=${MI_PY:-$HOME/nsd_w22/external/8dna26/.venv/bin/python}  # worklog-23 venv (ninja, OptiX libs)
export RSO_PY=${RSO_PY:-$HOME/NeuralShaderDemo/external/relightable-neural-assets/.venv/bin/python}
mkdir -p "$R/c2_launcher"
D="$R/datasets_c2"
if [ ! -f "$R/c2_launcher/train_generated.done" ]; then
  echo "start c2_train_generation $(date +%s)" >> "$R/c2_launcher/events"
  ( cd "$REPL" && PATH="$(dirname "$MI_PY"):$PATH" LD_LIBRARY_PATH=$HOME/nvlibs/optix595 "$MI_PY" -X faulthandler "$E/rso_data_c2.py" --batch-dir "$R" --split train ) \
      > "$R/c2_launcher/c2_train.log" 2>&1
  rc=$?
  echo "end c2_train_generation rc=$rc $(date +%s)" >> "$R/c2_launcher/events"
  if [ $rc -ne 0 ]; then echo "exit=$rc stage=c2_train_generation" > "$R/c2_launcher/status"; exit $rc; fi
  touch "$R/c2_launcher/train_generated.done"
fi
bash "$E/rso_server_chain.sh" "$R" C2 "$D/teaset_T0_train.h5" "$D/teaset_T0_val.h5"
rc=$?
echo "exit=$rc end=$(date +%s)" > "$R/c2_launcher/status"
