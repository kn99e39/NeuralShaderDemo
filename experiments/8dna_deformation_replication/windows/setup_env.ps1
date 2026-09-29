# Build the native-Windows runtime for the 8DNA replication on the RTX 5080 (SM 12.0).
#
# Upstream pins torch==2.3.1; its CUDA wheels stop at SM 9.0 and the process
# aborted under it (worklog 19).  The one deliberate deviation is
# torch 2.8.0+cu128, which ships sm_120 kernels.  Mitsuba 3.5.1, DrJit 0.4.6 and
# Lightning 2.1.3 stay at the upstream pins.  The upstream JIT extension
# (models/cu_ext) is compiled unmodified by windows/run.ps1 with CUDA 12.8 +
# MSVC 14.38 for sm_120.
#
# Native Windows, not WSL: DrJit 0.4.6 cannot load OptiX under WSL, whose
# libnvoptix.so.1 is a dxcore loader shim without optixQueryFunctionTable
# (see wsl/README.md).
$ErrorActionPreference = 'Stop'
$root = Resolve-Path "$PSScriptRoot/../../.."
$venv = Join-Path $root 'external/8dna26/.venv-cu128'
py -3.11 -m venv $venv
$py = Join-Path $venv 'Scripts/python.exe'
& $py -m pip install --upgrade pip
& $py -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
& $py -m pip install mitsuba==3.5.1 drjit==0.4.6 lightning==2.1.3 omegaconf tensorboard matplotlib `
    numpy==1.26.4 tqdm ninja scikit-image==0.24.0 "setuptools<70" trimesh==4.5.3 python-fcl==0.7.0.8 h5py==3.12.1
& $py -c "import sys, torch, mitsuba, numpy, lightning; mitsuba.set_variant('cuda_ad_rgb'); import drjit; print(sys.version.split()[0], torch.__version__, torch.version.cuda, torch.cuda.get_arch_list()); print('mitsuba', mitsuba.__version__, 'drjit', drjit.__version__, 'numpy', numpy.__version__, 'lightning', lightning.__version__)"
