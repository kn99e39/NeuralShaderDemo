# Run a Python entry point of this experiment in the native-Windows 8DNA runtime.
#   pwsh windows/run.ps1 <script.py> [args...]
# Imports the MSVC 14.38 x64 environment and CUDA 12.8 so torch can JIT-build
# the unmodified upstream extension for sm_120 on first import.
$ErrorActionPreference = 'Stop'
$root = Resolve-Path "$PSScriptRoot/../../.."
$vcvars = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
cmd /c "`"$vcvars`" -vcvars_ver=14.38 >nul && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item "env:$($Matches[1])" $Matches[2] }
}
$cuda = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
$env:CUDA_HOME = $cuda
$env:CUDA_PATH = $cuda
$env:PATH = "$cuda\bin;" + (Join-Path $root 'external\8dna26\.venv-cu128\Scripts') + ";$env:PATH"
$env:TORCH_CUDA_ARCH_LIST = '12.0'
$env:PYTHONUTF8 = '1'
Set-Location (Join-Path $root 'experiments\8dna_deformation_replication')
& (Join-Path $root 'external\8dna26\.venv-cu128\Scripts\python.exe') -X faulthandler @args
exit $LASTEXITCODE
