# Worklog — WSL DXG Runtime Diagnosis (2026-09-23)

## Relation to prior record

This worklog adds the retained kernel-log evidence to
`11_WORKLOG_F3_REFIT_RUNTIME_ENVIRONMENT_BLOCK.md`; it does not alter that
session's unmeasured-refit verdict.

## Diagnostic evidence

After the CPU-staged F3 data-loader process ended without its completion
marker, the prior-boot WSL journal contained a kernel warning from the
DirectX GPU bridge:

- `drivers/hv/dxgkrnl/dxgvmbus.c:3095`
- `dxgvmb_send_wait_sync_object_gpu`
- `dxgkio_submit_wait_to_hwqueue`

It was preceded and followed by repeated `dxgkio_query_adapter_info` failures,
then the WSL instance restarted.  This is host/runtime evidence, not a
training loss, an out-of-memory verdict, or an RNA result.

At observation time the installed runtime reported WSL 2.7.10.0 and kernel
6.18.33.2-2.  Because the fault is below the repository and the same outcome
has recurred for GPU-resident initialization, full training, and CPU-staged
loader initialization, no further refit attempts were run in this session.

## Resumption condition

Restore a stable WSL GPU bridge (or move the committed corpus/config to a
stable Linux/CUDA host), then rerun the read-only data-module completion
marker before beginning F3 training.  Preserve the frozen-model/refit
comparison protocol established for the primary Rain case.
