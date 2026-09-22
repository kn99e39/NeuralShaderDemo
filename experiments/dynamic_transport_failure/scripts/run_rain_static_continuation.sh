#!/usr/bin/env bash
# Continue the fixed Rain static corpus after an already-running batch03 PID.
set -euo pipefail

batch03_pid="${1:?pass the running batch03 PID}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
official_root="$repo_root/external/relightable-neural-assets"
python_bin="$official_root/.venv/bin/python"
dataset_root="$repo_root/results/batch1_hq_dynamic_failure/cloth/static/dataset"
scripts_root="$repo_root/experiments/dynamic_transport_failure/scripts"
configs_root="$repo_root/experiments/dynamic_transport_failure/configs"

while kill -0 "$batch03_pid" 2>/dev/null; do
  sleep 15
done

cd "$repo_root"
"$python_bin" "$scripts_root/validate_rna_h5.py" \
  --dataset "$dataset_root/rain_static_train_batch03.h5" --expected-images 50 \
  --output "$dataset_root/rain_static_train_batch03_integrity.json"

cd "$official_root"
"$python_bin" "$scripts_root/generate_rna_with_seed.py" \
  --official-root "$official_root" \
  --config "$configs_root/rain_static_hq_batch04_generate.yml" \
  --seed 1004 --record "$dataset_root/rain_static_train_batch04_launch.json"

cd "$repo_root"
"$python_bin" "$scripts_root/validate_rna_h5.py" \
  --dataset "$dataset_root/rain_static_train_batch04.h5" --expected-images 50 \
  --output "$dataset_root/rain_static_train_batch04_integrity.json"

cd "$official_root"
"$python_bin" "$scripts_root/generate_rna_with_seed.py" \
  --official-root "$official_root" \
  --config "$configs_root/rain_static_hq_val_generate.yml" \
  --seed 2001 --record "$dataset_root/rain_static_val_launch.json"

cd "$repo_root"
"$python_bin" "$scripts_root/validate_rna_h5.py" \
  --dataset "$dataset_root/rain_static_val.h5" --expected-images 40 \
  --output "$dataset_root/rain_static_val_integrity.json"
"$python_bin" "$scripts_root/merge_rna_h5.py" \
  --inputs "$dataset_root/rain_static_train_batch01.h5" "$dataset_root/rain_static_train_batch02.h5" \
    "$dataset_root/rain_static_train_batch03.h5" "$dataset_root/rain_static_train_batch04.h5" \
  --output "$dataset_root/rain_static_train.h5" --manifest "$dataset_root/rain_static_train_merge.json"
"$python_bin" "$scripts_root/validate_rna_h5.py" \
  --dataset "$dataset_root/rain_static_train.h5" --expected-images 200 \
  --output "$dataset_root/rain_static_train_integrity.json"
