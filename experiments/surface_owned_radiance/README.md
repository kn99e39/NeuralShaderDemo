# Surface-owned radiance units (worklog 31)

Representation-ownership study: a transport representation whose learned state is radiometric
(explicit outgoing >= 2-vertex radiance tables), owned by surface units of one object each, with no
learned shared decoder; tested for T0 quality (O0-like), all-local recovery of the teaset T0 -> T3
change, and whole-object oracle selective refit (O1-like) on this one substrate.
Protocol and decisions: `protocol/sor_v1.json`.

| file | role |
|---|---|
| `protocol/sor_v1.json` | predeclared protocol, prior-art audit summary, amendments |
| `sor_common.py` | units (per-part voxel x orientation anchors, canonical frame), stencil, octahedral tables, GPU surface sampler |
| `sor_targets.py` | queries -> rays -> worklog-24 `decompose()` MC targets (direct / >= 2-vertex), camera queries |
| `sor_fit.py` | online least-squares fit (trainable units as a sub-table), camera rendering, stencil helpers |
| `sor_units.py`, `sor_heldout.py` | units and held-out angular-fidelity sets |
| `sor_arm.py` | T0 pretraining and the T3 arms B, C, DG, E, E5, U |
| `sor_oracle.py` | CRN reference oracle: per-unit and per-object change energy, dirty sets, masks |
| `sor_metrics.py` | region / per-object errors, refit rule, tracking, preservation, latency, gates |
| `sor_exports.py` | reviewer panels for results/evaluation/31 |
| `sor_chain.py` | resumable evidence chain |
| `sor_pilot.py`, `sor_pilot2.py` | T0-only capacity pilots (resolution choice, before the protocol) |
| `tests/test_sor.py` | focused tests + synthetic fixture (analytic radiance in place of the tracer) |

Run every script through `experiments/8dna_deformation_replication/windows/run.ps1` with absolute paths.
