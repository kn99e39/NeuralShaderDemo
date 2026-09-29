#!/usr/bin/env bash
# Build the LabServer63 (RTX 3080 Ti, SM 8.6) runtime for 8DNA training.
#
# Unlike the RTX 5080, SM 8.6 is covered by the upstream torch 2.3.1 wheels, so
# this environment uses the upstream pins unchanged: torch 2.3.1, Mitsuba
# 3.5.1, DrJit 0.4.6, Lightning 2.1.3.  It is used for training only; every
# reference and evaluation render stays on the RTX 5080 (see
# ../windows/run.ps1), because the DrJit LoopRecord behaviour is GPU-specific
# and references must not be split across machines.
#
# Run on the server:  bash setup_env.sh
set -euo pipefail
ROOT=${ROOT:-$HOME/NeuralShaderDemo}
VENV=${VENV:-$ROOT/external/8dna26/.venv}

uv venv --clear --python 3.10 "$VENV"
uv pip install --python "$VENV/bin/python" \
  --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match torch==2.3.1
uv pip install --python "$VENV/bin/python" \
  mitsuba==3.5.1 drjit==0.4.6 lightning==2.1.3 omegaconf tensorboard matplotlib \
  numpy==1.26.4 tqdm ninja "setuptools<70" h5py==3.12.1

"$VENV/bin/python" - <<'EOF'
import sys, torch, mitsuba, numpy, lightning
mitsuba.set_variant('cuda_ad_rgb')
import drjit
print(sys.version.split()[0], torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())
print('device', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print('mitsuba', mitsuba.__version__, 'drjit', drjit.__version__, 'numpy', numpy.__version__, 'lightning', lightning.__version__)
EOF
