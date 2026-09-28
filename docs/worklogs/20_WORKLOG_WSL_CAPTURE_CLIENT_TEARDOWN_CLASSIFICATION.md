# Worklog — WSL Capture: Client-Teardown Classification (2026-09-28)

## Relation to prior record

This worklog corrects the capture wrapper described in
`16_WORKLOG_WSL_DXG_ROOT_CAUSE_DIAGNOSIS.md`. It checks the
`AGENTS.md` rule that a job ending without its completion marker must first
be shown to have outlived its launching client, by rerunning it under
`capture_wsl_gpu_pv_failure.ps1`. The check found that the wrapper, as
committed in `19e220a`, could not support that rule. Worklog 16's measured
results still stand; see the last section.

## Defects found

1. **`$` in `-Command` was expanded twice.** `wsl.exe -- bash -lc <cmd>`
   passes the arguments through the distro's default shell. That shell
   expanded `$(...)` before the inner bash saw it. A test command containing
   `$(seq 1 60)` failed at once with `syntax error near unexpected token '2'`
   (`exit 2`).
2. **Client teardown and job failure were indistinguishable.** The summary's
   `exit_code` was the `wsl.exe` client's code only, and the utility VM does
   not reboot on a teardown, so `vm_restarted` stays false. A killed client
   and a job exiting 1 therefore looked alike. The post-run poll could not
   show the distro's idle shutdown either: the wrapper's own
   `dmesg --follow` client keeps the distro alive during capture.

## Changes

`experiments/dynamic_transport_failure/scripts/capture_wsl_gpu_pv_failure.ps1`:

- Guest and workload calls use `wsl.exe --exec`, which does no default-shell
  re-parse.
- The workload runs inside a guest wrapper that writes
  `workload_guest_status.log`. The log holds a `start=` line, a
  `sighup=<time>` line if the launching client went away, and
  `job_exit=<code>` for the workload itself (129 means killed by SIGHUP). The
  wrapper forwards SIGHUP to the workload, so the workload dies as it would
  in the foreground; without forwarding, one probe run kept going after the
  teardown.
- `summary.json` gains `termination`, `guest_job_exit`, `guest_sighup` and
  `client_exit_code`, which replaces `exit_code`. `termination` takes one of
  these values:

| Value | Meaning |
|---|---|
| `completed` | the workload exited 0 with no SIGHUP |
| `job_failed` | the workload exited non-zero on its own |
| `client_teardown` | the launching client went away |
| `killed_without_exit` | the wrapper died with the job, e.g. VM/distro stop or SIGKILL |
| `no_guest_status` | the wrapper never ran, or its record was lost |

- The comment on the post-run wait now says that a "stopped" poll during
  capture means a VM/distro failure, never an idle shutdown.

## Verification (regime: capture-tool validation; A and B use CPU-only probe commands)

| Case | Command | Summary |
|---|---|---|
| A — job fails on its own | loop using `$(seq 1 5)`, then `exit 1` | `termination=job_failed`, `guest_job_exit=1`, no SIGHUP |
| B — launching client force-killed after 10 s | 60 s heartbeat using `$(seq 1 60)` | `termination=client_teardown`, SIGHUP at 17:29:15 (1 s after the kill), `guest_job_exit=129`, `client_exit_code=-1` |
| C — real workload | F0 provenance render (`wsl_repro/F0_run04`) | `termination=completed`, `guest_job_exit=0`; `rna_0.exr`, `ref_0.exr`, full trace; workload-window kernel hits: only 23 dxg ioctl failures |

A first B run without SIGHUP forwarding recorded the SIGHUP but no
`job_exit`, because the backgrounded workload had survived it. That run
motivated the forwarding change and is not used as evidence.

## Effect on earlier results

The committed commands in worklog 16's captures contained no `$`, so defect 1
did not affect them. That includes the F0 render, the loader check and the
soak launchers, all of them `bash <script>` or `VAR=… bash <script>`. Those
runs completed with the client attached, so their completion does not
depend on defect 2. The classification is new capability; it does not
revise any worklog 16 verdict.
