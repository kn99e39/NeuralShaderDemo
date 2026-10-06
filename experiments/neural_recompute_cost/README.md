# Neural full-recompute cost (worklog 27)

How long do the existing neural baselines take to produce a T3-valid
representation again after the accepted teaset T0 -> T3 change, when they are
rebuilt exactly as worklog 22 refit them? Measurement only; no method.

Protocol: `protocol/nrc_t3_recompute_v1.json` (written before any run).

| file | role |
|---|---|
| `nrc_records.py` | event log, phase accounting, checkpoint->wall-clock mapping, budgets, record schema (stdlib) |
| `nrc_8dna_train_timed.py` | timing wrapper around the historical `train_8dna_state.py` -> upstream `train.py` |
| `nrc_rna_h5_timed.py` | timing wrapper around the historical `rna_bridge.generate_h5` |
| `wsl/nrc_rna_train_timed.py` | timing wrapper around RNA's official `scripts/train.py` (WSL) |
| `nrc_chain.py` | runs the timed stages one at a time on the RTX 5080, then the untimed evaluation |
| `nrc_eval_8dna.py`, `nrc_eval_rna.py` | per-checkpoint T3 renders scored with the unchanged `cross_backbone_eval.refit_block` |
| `nrc_inference_timing.py` | steady-state inference cost, kept apart from rebuild latency |
| `nrc_identity_check.py` | instrumented vs plain training on a tiny schedule (weights compared) |
| `nrc_report.py` | timing records, quality-vs-time figures, review panels, `results/evaluation/<n>/` |
| `tests/` | `test_nrc_records.py` (stdlib), `test_historical_rule.py` (8DNA venv: reproduces worklog-22 verdicts) |

```text
python tests/test_nrc_records.py
pwsh experiments/8dna_deformation_replication/windows/run.ps1 ../neural_recompute_cost/tests/test_historical_rule.py
pwsh experiments/8dna_deformation_replication/windows/run.ps1 ../neural_recompute_cost/nrc_chain.py --run-id v1_<commit>
pwsh experiments/8dna_deformation_replication/windows/run.ps1 ../neural_recompute_cost/nrc_report.py --run results/neural_recompute_cost/v1_<commit> --worklog 27
```

The chain refuses a dirty tree and a run id that does not end in the current
commit. Outputs: `results/neural_recompute_cost/<run>/` (historical result
directories are only read).
