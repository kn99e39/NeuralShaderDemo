# Worklog — WSL Render Termination: Client Teardown, Not DXG (2026-09-28)

## Relation to prior record

This worklog revises the runtime-block readings of
`11_WORKLOG_F3_REFIT_RUNTIME_ENVIRONMENT_BLOCK.md`,
`13_WORKLOG_WSL_DXG_RUNTIME_DIAGNOSIS.md` and
`15_WORKLOG_RAIN_ATTRIBUTION_COVERAGE_AND_RUNTIME_BLOCK.md`, which attributed
the repeated WSL terminations to the DXG / GPU-PV bridge. It also supersedes
earlier uncommitted drafts of this worklog that named DXG/GPU-PV as the
leading candidate. No RNA quality, refit or attribution verdict changes;
only the runtime explanation and the resumption condition do. Worklog 17's
move of the Rain attribution run to `LabServer63` is unaffected.

## Conclusion

On this host the recorded "WSL DXG restarts" are **not a GPU-PV fault**.
Every one of them has the signature of the launching `wsl.exe` client being
terminated mid-run: the Linux child receives SIGHUP and exits without its
completion marker, and WSL then idles the distro down ~15 s later through an
orderly systemd poweroff. The same workloads complete when the client stays
attached or when they are detached correctly. The dxgkrnl ioctl failures and
the FORTIFY `WARN` are present on every distro start but did not coincide
with any failure. What terminated the client in the historical runs was not
recorded; that link is inferred from the matching signature, not observed.

## Environment

Windows 10.0.26200.9457, WSL 2.7.10.0, guest kernel
6.18.33.2-microsoft-standard-WSL2, WSLg 1.0.73.2, systemd enabled
(`/etc/wsl.conf`), no `.wslconfig`, default user root, host NVIDIA driver
596.49 / CUDA 13.2 (guest `nvidia-smi` reports the same package as
595.71.05), GeForce RTX 5080.

## Evidence

### Historical lifecycle (retained journal and host event logs)

- All 23 journal boots from 2026-09-23 to 2026-09-28, including the
  2026-09-23 refit attempts and the 2026-09-27 F0 diagnostic attempts, end
  with an orderly `systemd-poweroff` sequence ("System is powering down" ...
  "Journal stopped"). There was no panic, Oops, OOM, hung-task message or
  truncated journal.
- In the 2026-09-27 19:01–19:05 F0 window, the host event log shows only
  Hyper-V event 18508 ("the guest operating system shut down") and normal
  VM/network lifecycle events. There were no errors, `nvlddmkm`/Display TDR
  events or WER kernel reports.
- Many journal "boots" are distro restarts inside one VM kernel. A distro
  restart starts a fresh journald that re-reads the kernel ring buffer, so
  the boot-time `dxgkio_*` failures and the single FORTIFY `WARN` reappear
  in every such journal. A `wsl --shutdown` in this session did reboot the
  kernel, and its journal was marked "corrupted or uncleanly shut down". No
  historical failure journal has that mark, so none was a `wsl --shutdown`.
- The historical F0 diagnostic kept `rna_0.exr` and a trace ending at
  `model_render_done`, but no `ref_0.exr`. It died in the reference stage,
  which runs **CPU** Cycles (`scene.cycles.device = "CPU"` in
  `rna/renderers.py`). The GPU-PV bridge is not on that stage's critical
  path.

### Reproductions with the client attached (regime: runtime stability, not experiment results)

Each run was wrapped in
`experiments/dynamic_transport_failure/scripts/capture_wsl_gpu_pv_failure.ps1`,
which streams the guest kernel log and collects host events for the same
window. "Workload window" counts exclude the boot baseline.

| Workload (2026-09-28) | Result | Workload-window kernel / host |
|---|---|---|
| F0 provenance diagnostic render (`--model --reference`), fresh dir `wsl_repro/F0_run01` | exit 0 in 73 s; `rna_0.exr` **and** `ref_0.exr`; trace complete | 18 dxg ioctl failures, 0 WARN/Call Trace/panic/OOM; no VM restart; no host errors |
| F3 data module, CPU-staged (worklog 11 marker test) | `CPU_STAGED_READY` (train 200 / val 40) | 23 dxg ioctl failures across both modes; otherwise clean |
| F3 data module, GPU-resident | `GPU_RESIDENT_READY` (CUDA 482 MiB) | (same run as above) |
| Official `train.py`, F3 corpus, 5 epochs | exit 0 in 80 s | 23 dxg ioctl failures; otherwise clean |
| Official `train.py`, F3 corpus, 60 epochs | exit 0 in 5 min 32 s (~12k steps; GPU util mean 48.5 %, max 80 %; max 4.7 GiB) | 23 dxg ioctl failures; otherwise clean |
| Official `train.py`, F3 corpus, full 250-epoch schedule | exit 0 in 17 min 44 s (`max_epochs=250` reached); 528 host polls, none stopped | 23 dxg ioctl failures, 0 WARN/Call Trace/panic/OOM; host events only 3× System 7040 (service start-type change) |

The boot-baseline `make_resident` hit in the 250-epoch capture is a `?`
frame inside the boot-time FORTIFY WARN call trace, not a separate event.
The pattern counter matches stack frames as well as messages.

The soak checkpoints (`wsl_repro/soak_train_run0{1,2,3}`) are runtime probes
only. They are not an F3 refit result and must not enter the frozen/refit
comparison.

### Controlled lifecycle experiments (CPU-only heartbeat unless noted)

| Condition | Outcome |
|---|---|
| Foreground job; launching `wsl.exe` force-killed at 10 s | Job got SIGHUP 2 s later and exited without its marker. The distro powered down ~15 s later in orderly fashion. **Reproduces the historical signature.** |
| `nohup setsid` job; launching client **exits normally** | Job ran to completion (150 s and 60 s runs). The distro stayed up until ~15 s after the job ended. |
| `nohup setsid` job; launching client **force-killed** while lingering | Distro powered down ~16 s later and killed the detached job with it. |
| `systemd-run` transient service; client exits normally | WSL idle shutdown ignored the service and killed it at ~16 s. |
| Detached job running; an **unrelated** `wsl.exe` client force-killed | No effect; job completed. |
| Launcher exiting immediately after forking the job | Race: the child was torn down before `setsid` (empty logs). Fixed by waiting for the child's own session and START line. |
| F0 render through the race-fixed launcher (`wsl_repro/detached_F0_run03`, run by the user from PowerShell) | The launcher returned at once. The job exited 0 after 59.5 s with a complete trace, `rna_0.exr`, and `ref_0.exr` byte-identical in size to the attached run01 (4,203,359). Its stderr/stdout warnings (shape-key load messages, addon-unregister `TypeError` at exit) match run01. |

## Tools added

- `capture_wsl_gpu_pv_failure.ps1` — capture wrapper described above.
  It waits `-SettleSeconds` (default 30) of guest uptime before the workload,
  so boot-time dxgkrnl output is counted as baseline.
- `check_rna_datamodule_init.py` — the worklog 11 read-only data-module
  marker test (`CPU_STAGED_READY` / `GPU_RESIDENT_READY`), now retained.
- `wsl_run_detached.sh` — starts a job with `nohup setsid` and logs
  START/exit to `<log-dir>/<job>.{stdout,stderr}.log` and `<job>.status`. It
  returns only after the child has left the client's session. The launching
  client must then **exit normally**; polling from other short `wsl.exe`
  calls is safe.

## Remaining limits

- The historical client terminations were not observed directly; their
  cause, for example a tool-call timeout, is inferred.
- GPU runs were tested up to 17 min 44 s: the full F3 training schedule,
  run once, as a soak with seed 0. A rare GPU-PV fault in longer or
  repeated runs is not excluded.
- The detached F0 run was not wrapped in the capture tool, so its window
  has no kernel/host capture; only its exit status and outputs were checked.
- The rule that a force-killed launching client takes down even a detached
  job was measured on WSL 2.7.10.0 only.

## Resumption condition

Local WSL GPU work can resume. Launch long jobs with `wsl_run_detached.sh`
from a client that exits normally, and poll `<job>.status` from separate
short `wsl.exe` calls. Alternatively, run the job in the foreground of a
client that nothing will kill, such as the capture wrapper run as a
background task. The capture wrapper remains available if a termination
recurs.

This replaces the "stable Linux/CUDA host" condition of worklogs 11/13/15
for this machine. The F3 refit protocol from worklog 11 is unchanged.
