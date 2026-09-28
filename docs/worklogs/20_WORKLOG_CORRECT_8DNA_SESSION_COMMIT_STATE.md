# Worklog — Correct 8DNA Session Commit State (2026-09-28)

This worklog corrects only the repository-state statement in
`docs/worklogs/19_WORKLOG_8DNA_BASELINE_REPLICATION_GATE.md`.

Worklog 19 says the final project state was intentionally uncommitted. That was
true when the runtime attempt and first report draft were produced, but the
validated session artifacts were subsequently committed as:

`3117ba1` — `8DNA-baseline-replication-gate`

The experimental runtime attempt itself used project base commit
`e2056e0c92017d6d1fe7303a27e699a9a4560db2` plus the then-uncommitted
project-owned adapter. No experiment was rerun after commit, and the scientific
classification remains **BASELINE REPRODUCTION FAILED**.

This correction changes no measurement, observation, interpretation, source
audit, or classification. The unrelated modification to
`experiments/dynamic_transport_failure/scripts/capture_wsl_gpu_pv_failure.ps1`
remains untouched.
