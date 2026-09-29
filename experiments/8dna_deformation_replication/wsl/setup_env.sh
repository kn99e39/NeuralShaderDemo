#!/usr/bin/env bash
# Build the WSL runtime used for the 8DNA replication on the RTX 5080 (SM 12.0).
#
# Upstream pins torch==2.3.1, whose CUDA wheels stop at SM 9.0 (worklog 19).
# The only deliberate deviation is torch 2.8.0+cu128, the build the project's
# RNA environment already runs on this GPU.  Mitsuba/DrJit/Lightning stay at
# the upstream pins.  nvcc 12.8 matches the torch CUDA build so the upstream
# JIT extension (models/cu_ext) compiles for sm_120 unmodified.
#
# Run as root inside Ubuntu-22.04:  bash setup_env.sh
set -euo pipefail
VENV=${VENV:-/opt/venvs/8dna26}

if ! command -v /usr/local/cuda-12.8/bin/nvcc >/dev/null; then
  cd /tmp
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
  dpkg -i cuda-keyring_1.1-1_all.deb
  apt-get update -q
  # Compiler, runtime and the library headers ATen/cuda/CUDAContext.h includes.
  DEBIAN_FRONTEND=noninteractive apt-get install -y -q \
    cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-nvtx-12-8 cuda-profiler-api-12-8 \
    libcublas-dev-12-8 libcusparse-dev-12-8 libcusolver-dev-12-8 libcurand-dev-12-8
fi

uv venv --clear --python /usr/bin/python3.10 "$VENV"
uv pip install --python "$VENV/bin/python" \
  --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match torch==2.8.0
uv pip install --python "$VENV/bin/python" \
  mitsuba==3.5.1 drjit==0.4.6 lightning==2.1.3 omegaconf tensorboard matplotlib \
  numpy==1.26.4 tqdm ninja scikit-image==0.24.0 "setuptools<70" trimesh==4.5.3 python-fcl==0.7.0.8

"$VENV/bin/python" - <<'EOF'
import sys, torch, mitsuba, numpy, lightning
mitsuba.set_variant('cuda_ad_rgb')
import drjit
print(sys.version.split()[0], torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())
print('mitsuba', mitsuba.__version__, 'drjit', drjit.__version__, 'numpy', numpy.__version__, 'lightning', lightning.__version__)
EOF
