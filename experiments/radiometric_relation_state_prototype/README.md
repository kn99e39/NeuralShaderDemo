# Radiometric relation-state prototype (worklog 29)

The second bounded architecture experiment. It holds worklog 28's sparse relation
structure fixed: the same K = 32 probes, frames, 20-channel geometric descriptor,
split, operator class, budget and seeds. It changes one factor: each current
probe hit also carries a runtime radiometric proxy, the frozen RNA's own
rendering of the remote hit toward the query under current light and
visibility. Matched ZERO and SHUFFLED controls separate *aligned* radiometric
content from extra channels. The exact-direction oracle (path-traced incident
radiance on the same 32 directions) is diagnostic only and runs only if the real
candidate fails.

Protocol: `protocol/rrs_v1.json`, written before any proxy computation or
training. Worklog 28 (`experiments/relational_residual_prototype/`) is imported
and never modified.

| file | role | runs in |
|---|---|---|
| `rrs_common.py` | proxy channels, encoding, ZERO / SHUFFLED / REAL assembly | both |
| `rrs_remote_light.py` | reproduce worklog-28 probes bit for bit; lit-scene re-cast; area-light samples at every hit | Windows (Mitsuba) |
| `wsl/rrs_proxy.py` | the proxy via rna_infer's renderer at each hit (self-check: reproduces worklog-28 base exactly) | WSL (RNA venv) |
| `wsl/rrs_train.py` | geom20 reproduction run + ZERO / SHUFFLED / REAL x 3 seeds (+ oracle) with worklog 28's loop | WSL |
| `rrs_eval.py` | R_delta (primary), success test, historical metrics, T1 control, state diagnostics, cost, exports | Windows |
| `rrs_oracle_radiance.py` | DIAGNOSTIC: path-traced incident radiance along the same 32 directions | Windows |
| `rrs_run.py` | stage driver (`--oracle` for the conditional diagnostic, `--smoke` for an untrained check) | Windows |
| `tests/` | `test_rrs_common.py` (numpy), `test_rrs_light_win.py` (Mitsuba synthetic contract), `test_rrs_wsl.py` (torch) | |
