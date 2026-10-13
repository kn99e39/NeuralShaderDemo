#!/usr/bin/env bash
# Worklog-32 chain for one configuration on LabServer63 (resumable; status file at the end).
#   setsid -f bash rso_server_chain.sh <run dir> <config name> <train h5> <val h5>
set -u
R=$1; CFG=$2; TRAIN=$3; VAL=$4
E=$(cd "$(dirname "$0")" && pwd)
PY=${RSO_PY:-$HOME/NeuralShaderDemo/external/relightable-neural-assets/.venv/bin/python}
mkdir -p "$R/chain" "$R/$CFG"
rm -f "$R/chain/status_$CFG"
stage() {
  local name=$1; shift
  [ -f "$R/chain/$name.done" ] && return 0
  echo "start $name $(date +%s)" >> "$R/chain/events"
  "$@" > "$R/chain/$name.log" 2>&1; local rc=$?
  echo "end $name rc=$rc $(date +%s)" >> "$R/chain/events"
  if [ $rc -ne 0 ]; then echo "exit=$rc stage=$name end=$(date +%s)" > "$R/chain/status_$CFG"; exit $rc; fi
  touch "$R/chain/$name.done"
}
stage partitions "$PY" "$E/rso_partition.py" --out "$R/partitions"
stage tests "$PY" "$E/tests/test_rso.py" --partitions "$R/partitions" --out "$R/tests"
"$PY" "$E/rso_make_config.py" --train "$TRAIN" --val "$VAL" --name "rso-$CFG" --out "$R/configs/$CFG.yml"
C="$R/configs/$CFG.yml"; P="$R/partitions"
stage "${CFG}_A"   "$PY" "$E/rso_train.py" --config "$C" --partition none --out "$R/$CFG/A"
stage "${CFG}_P16" "$PY" "$E/rso_train.py" --config "$C" --partition "$P/partition_P16.npz" --out "$R/$CFG/P16"
U=$("$PY" "$E/rso_unit_star.py" "$P/partition_P16.npz" "$R/$CFG/P16/train.json")
stage "${CFG}_REINIT" "$PY" "$E/rso_train.py" --config "$C" --partition "$P/partition_P16.npz" --out "$R/$CFG/REINIT"       --init "$R/$CFG/P16/state.pt" --train-units "$U"
stage "${CFG}_P4"  "$PY" "$E/rso_train.py" --config "$C" --partition "$P/partition_P4.npz" --out "$R/$CFG/P4"
stage "${CFG}_P1"  "$PY" "$E/rso_train.py" --config "$C" --partition "$P/partition_P1.npz" --out "$R/$CFG/P1"
stage "${CFG}_eval_main" "$PY" "$E/rso_eval.py" --config "$C" --partitions "$P" --out "$R/$CFG/eval_main"       --arms "A=$R/$CFG/A" "P1=$R/$CFG/P1" "P4=$R/$CFG/P4" "P16=$R/$CFG/P16" "REINIT=$R/$CFG/REINIT"
stage "${CFG}_P16w" "$PY" "$E/rso_train.py" --config "$C" --partition "$P/partition_P16.npz" --matched 0 --out "$R/$CFG/P16w"
stage "${CFG}_eval" "$PY" "$E/rso_eval.py" --config "$C" --partitions "$P" --out "$R/$CFG/eval"       --arms "A=$R/$CFG/A" "P1=$R/$CFG/P1" "P4=$R/$CFG/P4" "P16=$R/$CFG/P16" "P16w=$R/$CFG/P16w" "REINIT=$R/$CFG/REINIT"
echo "exit=0 end=$(date +%s)" > "$R/chain/status_$CFG"
