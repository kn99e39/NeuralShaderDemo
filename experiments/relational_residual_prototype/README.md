# Relational-residual prototype (worklog 28)

First bounded test of the working contract in `docs/Architecture.md`:

```text
L_candidate(i, t) = L_frozenRNA(i, t) + dL_dynamic(i, t)
dL_dynamic = psi( g_i^t , persistent/local query inputs )
g_i^t      = aggregate_k phi( probe_k descriptor from the CURRENT geometry )
```

The frozen historical RNA is never trained or modified. The prototype covers only
the stable queries of the canonical teaset interaction ROI. It is a sidecar,
not the method. Protocol: `protocol/rrp_v1.json`, written before any real-data
run.

| file | role | runs in |
|---|---|---|
| `rrp_common.py` | probe directions, frames, stable-query and split logic, feature names (leakage guard) | both |
| `rrp_probes.py` | K = 32 probes per query into the current geometry; references; stable flags; probe timing | Windows (Mitsuba) |
| `wsl/rrp_features.py` | frozen RNA base per sample (checked against the historical renders), persistent triplane features | WSL (RNA venv) |
| `wsl/rrp_model.py` | relational candidate and parameter-matched local-only control | WSL |
| `wsl/rrp_train.py` | training on T0/T1b/T2 only, selection on spatial validation blocks, predictions, latency | WSL |
| `rrp_eval.py` | worklog-22 metrics (evaluate, rule, refit_block), accounting, exports | Windows |
| `rrp_run.py` | runs the four stages in order; evidence needs a clean tree | Windows |
| `tests/` | `test_rrp_common.py` (numpy), `test_rrp_probes.py` (Mitsuba: synthetic moving occluder + teaset T0/T3), `test_rrp_model_wsl.py` (torch) | |

```text
pwsh experiments/8dna_deformation_replication/windows/run.ps1 ../relational_residual_prototype/rrp_run.py --run-id v1_<commit> --worklog 28
```
