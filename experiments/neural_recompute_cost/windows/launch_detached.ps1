# Start a long native-Windows job fully detached from the calling shell.
#   pwsh windows/launch_detached.ps1 <job-name> <log-dir> <script.py relative to experiments/8dna_deformation_replication> [args...]
#
# The job is created through WMI (Win32_Process.Create), so its parent is the
# WMI provider host, not this shell: an agent tool timeout or a closed terminal
# cannot take it down (the native-Windows counterpart of wsl_run_detached.sh).
# It runs ../../8dna_deformation_replication/windows/run.ps1 <script> [args...],
# with stdout+stderr in <log-dir>/<job>.log; <log-dir>/<job>.status is absent
# while it runs and holds "exit=<code> end=<time>" when it is done.
param(
    [Parameter(Mandatory)][string] $Job,
    [Parameter(Mandatory)][string] $LogDir,
    [Parameter(Mandatory, ValueFromRemainingArguments)][string[]] $Argv
)
$ErrorActionPreference = 'Stop'
$run = Resolve-Path (Join-Path $PSScriptRoot '../../8dna_deformation_replication/windows/run.ps1')
New-Item -ItemType Directory -Force $LogDir | Out-Null
$LogDir = (Resolve-Path $LogDir).Path
$status = Join-Path $LogDir "$Job.status"
$log = Join-Path $LogDir "$Job.log"
Remove-Item -Force -ErrorAction SilentlyContinue $status
$pwsh = (Get-Command pwsh).Source
$quoted = ($Argv | ForEach-Object { "'" + ($_ -replace "'", "''") + "'" }) -join ' '
$inner = @"
`$ErrorActionPreference = 'Continue'
"START `$(Get-Date -Format o) pid=`$PID" | Out-File -Encoding utf8 -FilePath '$log'
& '$pwsh' -NoProfile -File '$run' $quoted *>> '$log'
`$rc = `$LASTEXITCODE
"exit=`$rc end=`$(Get-Date -Format o)" | Out-File -Encoding utf8 -FilePath '$status'
"@
$enc = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($inner))
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = "`"$pwsh`" -NoProfile -WindowStyle Hidden -EncodedCommand $enc"
    CurrentDirectory = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
}
if ($r.ReturnValue -ne 0) { throw "Win32_Process.Create failed: $($r.ReturnValue)" }
"started $Job pid=$($r.ProcessId) log=$log status=$status"
