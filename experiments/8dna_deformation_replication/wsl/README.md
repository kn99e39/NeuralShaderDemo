# WSL runtime attempt (not used for results)

`setup_env.sh` and `run.sh` build a WSL Ubuntu-22.04 runtime with torch
2.8.0+cu128, the upstream Mitsuba 3.5.1 / DrJit 0.4.6 pins and nvcc 12.8. On the
RTX 5080 this gets as far as:

- torch sees the GPU with sm_120 kernels;
- the upstream `models/cu_ext` extension JIT-builds for sm_120 and imports;
- DrJit's CUDA backend initializes once `DRJIT_LIBCUDA_PATH` points at
  `/usr/lib/wsl/lib/libcuda.so.1`.

It stops at `mi.load_dict(...)` with `Could not initialize OptiX!`. DrJit 0.4.6
dlsyms `optixQueryFunctionTable` from `libnvoptix.so.1`, but under WSL that
library is a dxcore loader shim: it exports only `dxcore_*` functions, and
the real OptiX library is the opaque driver-store blob `nvoptix.bin`.
Every Mitsuba 3.5.x `cuda_*` variant needs OptiX for ray intersection. Using
WSL would therefore mean moving to Mitsuba 3.6+ / DrJit 1.x, which is an API
port of the upstream integrator. The runs use native Windows instead
(`../windows/`).

This failure is in the renderer runtime's WSL support. It is unrelated to
the WSL client-teardown issue in worklogs 16 and 20.
