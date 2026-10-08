# Oracle selective-refit feasibility (worklog 30, gate F0+F1)

Does a pretrained neural transport representation admit a selectively refittable subset of
learned state, given an offline oracle of which state the T0 -> T3 teaset edit invalidated?
Substrate: released 8DNA teaset asset (external/8dna26 @ 4a2157c, unmodified); localized state =
triplane cells; protocol and substrate decision in `protocol/sro_v1.json`.

| file | role |
|---|---|
| `protocol/sro_v1.json` | predeclared protocol: substrate audit, oracle, controls, budgets, regions, gates, amendments |
| `sro_common.py` | state units (triplane cells, exact fetch_2d stencil), masks, model/checkpoint I/O, scene registration |
| `sro_paths.py` | upstream PathSamplingDataset chunk body for any scene (common random numbers) |
| `sro_oracle.py` | oracle: CRN T0/T3 path traces -> affected rays -> per-cell weights -> M95, spatial mask S, decompositions, cost |
| `sro_regions.py` | evaluation regions from geometry and references only (run before training) |
| `sro_train.py` | arms B, C, D, E, E_mask, S: matched warm-start / scratch refits, snapshots, validation, timing events |
| `sro_render.py` | renders snapshots at T3 exactly as the historical refit renders |
| `sro_metrics.py` | region errors, refit rule, change tracking, frozen-output identity, adaptation latency, gates |
| `tests/test_sro.py` | focused tests (state units, masking/equivalence, loss and dataset equivalence with upstream, CRN, timing, checkpoints, data separation, mask determinism) |
| `tests/test_sro_synthetic.py` | synthetic geometry-change fixture (semantics only) |

All scripts run through `experiments/8dna_deformation_replication/windows/run.ps1` with absolute
output paths under `results/selective_refit_oracle/`.
