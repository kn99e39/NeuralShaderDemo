# Run one WSL workload while capturing guest kernel and host event evidence.
#
# The guest kernel log is streamed through wsl.exe into a host file, so lines
# written before a utility-VM restart survive it.  Host-side WSL running state
# is polled without waking the VM.  After the workload ends, the prior-boot
# kernel journal (when a restart happened) and host event logs for the same
# window are collected, and summary.json separates boot-time dxgkrnl noise
# from lines logged during the workload.
#
# Example (from Windows PowerShell 7):
#   ./capture_wsl_gpu_pv_failure.ps1 -OutDir results/.../wsl_capture/run01 `
#     -Command 'cd /mnt/c/Projects/NeuralShaderDemo && ./render.sh'
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string] $Command,
    [Parameter(Mandatory)] [string] $OutDir,
    [string] $Distro = 'Ubuntu-22.04',
    [int] $PollSeconds = 2,
    [int] $SettleSeconds = 30
)
$ErrorActionPreference = 'Stop'

$out = New-Item -ItemType Directory -Force $OutDir
function Out-Path([string] $name) { Join-Path $out.FullName $name }
function Guest([string] $script) { wsl.exe -d $Distro -u root -- bash -lc $script 2>&1 }

$kernelPatterns = [ordered]@{
    dxg_ioctl_failed        = 'dxgk: .*Ioctl failed'
    wait_for_completion     = 'wait_for_completion failed'
    make_resident           = 'dxgkio_make_resident'
    fortify_field_spanning  = 'field-spanning write'
    warning                 = 'WARNING: CPU'
    call_trace              = 'Call Trace'
    panic_or_oops           = 'Kernel panic|Oops|BUG:'
    hung_task               = 'blocked for more than'
    oom                     = 'Out of memory|oom-kill'
}
$hostLogs = @(
    'System'
    'Application'
    'Microsoft-Windows-Hyper-V-Worker-Admin'
    'Microsoft-Windows-Hyper-V-Compute-Admin'
    'Microsoft-Windows-Hyper-V-Compute-Operational'
    'Microsoft-Windows-WerKernel/Operational'
    'Microsoft-Windows-Kernel-PnP/Driver Watchdog'
)

# --- Pre-run state -----------------------------------------------------------
$env:WSL_UTF8 = '1'
(wsl.exe --version) | Set-Content (Out-Path 'wsl_version.txt')
(nvidia-smi) | Set-Content (Out-Path 'host_nvidia_smi_before.txt')
$bootBefore = (Guest 'cat /proc/sys/kernel/random/boot_id' | Select-Object -Last 1).Trim()
# dxgkrnl logs ioctl failures and a FORTIFY WARN within seconds of every boot.
# Start the workload only after that has settled so it is not counted as
# workload-time evidence.
$uptime = [double]((Guest 'cut -d" " -f1 /proc/uptime' | Select-Object -Last 1).Trim())
if ($uptime -lt $SettleSeconds) { Start-Sleep -Seconds ([Math]::Ceiling($SettleSeconds - $uptime)) }
$guestBootTime = (Get-Date).AddSeconds(-[double]((Guest 'cut -d" " -f1 /proc/uptime' | Select-Object -Last 1).Trim()))
Guest 'uname -a; nvidia-smi' | Set-Content (Out-Path 'guest_versions.txt')
Guest 'dmesg --time-format=iso' | Set-Content (Out-Path 'kernel_boot_before.log')

# --- Background capture ------------------------------------------------------
$follow = Start-Process wsl.exe -PassThru -NoNewWindow `
    -ArgumentList @('-d', $Distro, '-u', 'root', '--', 'dmesg', '--follow', '--time-format=iso') `
    -RedirectStandardOutput (Out-Path 'kernel_follow.log') `
    -RedirectStandardError (Out-Path 'kernel_follow.stderr.log')

$pollFile = Out-Path 'host_poll.log'
$poller = Start-Job -ArgumentList $Distro, $PollSeconds, $pollFile -ScriptBlock {
    param($distro, $seconds, $path)
    $env:WSL_UTF8 = '1'
    while ($true) {
        $running = (wsl.exe -l --running -q) -join ' '
        $state = if ($running -match [regex]::Escape($distro)) { 'running' } else { 'stopped' }
        "{0:o} {1}" -f (Get-Date), $state | Add-Content $path
        Start-Sleep -Seconds $seconds
    }
}

# --- Workload ----------------------------------------------------------------
$start = Get-Date
"$($start.ToString('o')) START $Command" | Set-Content (Out-Path 'workload_times.log')
wsl.exe -d $Distro -- bash -lc $Command `
    1> (Out-Path 'workload_stdout.log') 2> (Out-Path 'workload_stderr.log')
$exitCode = $LASTEXITCODE
$end = Get-Date
"$($end.ToString('o')) END exit=$exitCode" | Add-Content (Out-Path 'workload_times.log')

# Let a dying VM finish tearing down before stopping the capture.
Start-Sleep -Seconds ([Math]::Max(10, 3 * $PollSeconds))
Stop-Job $poller; Remove-Job $poller
if (-not $follow.HasExited) { Stop-Process -Id $follow.Id -Force }

# --- Post-run state ----------------------------------------------------------
$bootAfter = (Guest 'cat /proc/sys/kernel/random/boot_id' | Select-Object -Last 1).Trim()
$restarted = $bootAfter -ne $bootBefore
Guest 'journalctl --list-boots --no-pager' | Set-Content (Out-Path 'journal_boots.txt')
Guest 'journalctl -k -b 0 --no-pager -o short-iso-precise' | Set-Content (Out-Path 'kernel_journal_current_boot.log')
if ($restarted) {
    Guest 'journalctl -k -b -1 --no-pager -o short-iso-precise' | Set-Content (Out-Path 'kernel_journal_prior_boot.log')
    Guest 'journalctl -b -1 --no-pager -o short-iso-precise -n 400' | Set-Content (Out-Path 'journal_prior_boot_tail.log')
}
(nvidia-smi) | Set-Content (Out-Path 'host_nvidia_smi_after.txt')

$windowStart = $start.AddMinutes(-2)
$windowEnd = (Get-Date).AddMinutes(1)
$hostEvents = foreach ($log in $hostLogs) {
    try {
        Get-WinEvent -FilterHashtable @{ LogName = $log; StartTime = $windowStart; EndTime = $windowEnd } -ErrorAction Stop |
            ForEach-Object {
                [pscustomobject]@{
                    time = $_.TimeCreated.ToString('o'); log = $log; provider = $_.ProviderName
                    id = $_.Id; level = $_.LevelDisplayName; message = $_.Message
                }
            }
    } catch [System.Exception] {
        if ($_.FullyQualifiedErrorId -notmatch 'NoMatchingEventsFound') {
            [pscustomobject]@{ time = $null; log = $log; provider = 'capture'; id = -1; level = 'CaptureError'; message = $_.Exception.Message }
        }
    }
}
$hostEvents = @($hostEvents | Sort-Object time)
$hostEvents | ConvertTo-Json -Depth 3 | Set-Content (Out-Path 'host_events.json')

# --- Summary -----------------------------------------------------------------
# Count kernel pattern hits only for lines stamped inside the workload window,
# so boot-time dxgkrnl noise is not attributed to the workload.
function Count-InWindow([string[]] $paths, [datetime] $from, [datetime] $to) {
    $lines = foreach ($p in $paths) { if (Test-Path $p) { Get-Content $p } }
    # Journals from both boots overlap the follow log; count each line once.
    $inWindow = foreach ($line in ($lines | Select-Object -Unique)) {
        if ($line -match '^(\d{4}-\d\d-\d\dT[\d:.,]+[+-]\d\d:?\d\d)') {
            # dmesg writes "…,123456+09:00"; journalctl writes "….123456+0900".
            $stamp = ($Matches[1] -replace ',', '.') -replace '([+-]\d\d)(\d\d)$', '$1:$2'
            $t = [datetimeoffset]::Parse($stamp)
            if ($t -ge $from -and $t -le $to) { $line }
        }
    }
    $counts = [ordered]@{}
    foreach ($k in $kernelPatterns.Keys) { $counts[$k] = @($inWindow | Where-Object { $_ -match $kernelPatterns[$k] }).Count }
    $counts
}
$kernelSources = @('kernel_follow.log', 'kernel_journal_prior_boot.log', 'kernel_journal_current_boot.log') | ForEach-Object { Out-Path $_ }
$stoppedPolls = @(Get-Content $pollFile -ErrorAction SilentlyContinue | Where-Object { $_ -match ' stopped$' })

[ordered]@{
    distro                        = $Distro
    command                       = $Command
    start                         = $start.ToString('o')
    end                           = $end.ToString('o')
    exit_code                     = $exitCode
    boot_id_before                = $bootBefore
    boot_id_after                 = $bootAfter
    vm_restarted                  = $restarted
    first_stopped_poll            = if ($stoppedPolls) { $stoppedPolls[0].Split(' ')[0] } else { $null }
    guest_boot_time               = $guestBootTime.ToString('o')
    kernel_hits_boot_baseline     = Count-InWindow @(Out-Path 'kernel_boot_before.log') $guestBootTime.AddSeconds(-5) $start
    kernel_hits_during_workload   = Count-InWindow $kernelSources $start $end.AddSeconds(30)
    host_event_count              = $hostEvents.Count
    host_event_ids                = @($hostEvents | Group-Object log, id | ForEach-Object { $_.Name + ' x' + $_.Count })
} | ConvertTo-Json -Depth 4 | Set-Content (Out-Path 'summary.json')

Write-Host "Capture written to $($out.FullName) (exit=$exitCode, vm_restarted=$restarted)"
