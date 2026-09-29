#!/usr/bin/env bash
# Run a Python entry point of this experiment in the LabServer63 8DNA runtime.
#   bash server/run.sh <script.py> [args...]
# Puts the venv's bin (ninja) and the CUDA toolkit on PATH so torch can
# JIT-build the unmodified upstream extension (models/cu_ext) for sm_86.
set -euo pipefail
ROOT=${ROOT:-$HOME/NeuralShaderDemo}
VENV=$ROOT/external/8dna26/.venv
# nvcc 12.0 lives in /usr/bin here (distro package), matching torch's cu121 build
export CUDA_HOME=${CUDA_HOME:-/usr}
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.6
cd "$ROOT/experiments/8dna_deformation_replication"
exec "$VENV/bin/python" -X faulthandler "$@"
