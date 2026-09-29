#!/usr/bin/env bash
# Run a Python entry point of this experiment inside the WSL 8DNA runtime.
#   bash wsl/run.sh <script.py> [args...]
set -euo pipefail
export PATH=/opt/venvs/8dna26/bin:/usr/local/cuda-12.8/bin:$PATH CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=12.0
# DrJit 0.4.6 dlopens "libcuda.so" and "libnvoptix.so.1" through the ld cache;
# WSL exposes the driver libraries only as /usr/lib/wsl/lib/*.so.1.
export DRJIT_LIBCUDA_PATH=/usr/lib/wsl/lib/libcuda.so.1
export DRJIT_LIBOPTIX_PATH=/usr/lib/wsl/lib/libnvoptix.so.1
cd "$(dirname "$0")/.."
exec /opt/venvs/8dna26/bin/python -X faulthandler "$@"
