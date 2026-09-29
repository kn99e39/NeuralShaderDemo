# Worklog — 8DNA Cross-Model Geometry-Configuration Replication on the RTX 5080 (2026-09-29)

## Correction of the worklog-19 status

Worklog 19 is not edited. For current planning its interpretation is
superseded as follows:

- Its **BASELINE REPRODUCTION FAILED** verdict describes one runtime stack:
  native Windows with torch 2.3.1+cu121, which ships no SM 12.0 kernels. It
  was never a scientific negative result about 8DNA.
- This session reproduces the released 8DNA model on the same RTX 5080
  (section "Static baseline"). No 8DNA-specific incompatibility with the RTX
  5080 remains. The three runtime issues found and resolved are listed
  under "Runtime deviations".
- The RTX 5080 stays the primary host. LabServer63 / RTX 3080 Ti was not
  needed and was not used.

The RTX 5080 runtime history is read as worklogs 16 and 20
(`20_WORKLOG_WSL_CAPTURE_CLIENT_TEARDOWN_CLASSIFICATION.md`): the old "WSL
DXG failure" reports were launching-client teardown. That is unrelated to
anything in this session. The Rain state is not reinterpreted here. The
frozen RNA deformation failure is observed, the direct-visibility
explanation is weakened, the local/spatial explanation is plausible but not
causal, the fixed-target probe was PHYSICAL EFFECT TOO WEAK, and the
mechanism is unresolved.

Baseline roles are unchanged. RNA is the development base, 8DNA the
scientific replication baseline, RenderFormer-type the later contrast
baseline, and the method base is undecided. No 8DNA code, weight or
architecture was modified.

## Session question

On the RTX 5080, does a faithfully reproduced frozen 8DNA asset stay valid
when a physically meaningful cross-part geometry relation changes, under a
correspondence-valid evaluation?

**Primary classification: CROSS-MODEL GEOMETRY-CONFIGURATION FAILURE OBSERVED**
**Secondary characterization: TRANSPORT-LIKE**

Scope is one asset (`teaset`), one mover (milk pot), rigid translation
only, one camera and one envmap, all under the locked protocol. The
classification comes from the decision rule written into
`protocol/teaset_frozen_locked.json` before any moved-state neural render.

## IMPLEMENTATION FACT

### Environment (authoritative for this batch)

| Item | Value |
|---|---|
| Host | RTX 5080, 16 303 MiB, SM 12.0, driver 596.49; Windows 10.0.26200 |
| Python / venv | 3.11.9, `external/8dna26/.venv-cu128` (`windows/setup_env.ps1`) |
| PyTorch | 2.8.0+cu128 (sm_61 … sm_120) |
| Mitsuba / DrJit | 3.5.1 / 0.4.6 (upstream pins), variant `cuda_ad_rgb` |
| Lightning / NumPy | 2.1.3 (upstream pin; unused at inference) / 1.26.4 |
| Extension build | unmodified `models/cu_ext`, JIT-built by torch with CUDA 12.8 nvcc + MSVC 14.38 for sm_120 (`windows/run.ps1`) |
| Upstream | <https://github.com/lwwu2/8dna26> @ `4a2157ca24e506c5ac0831f27d656ecc50a64f07`, unmodified |

Release hashes match worklog 19's records:
- `scenes.zip`: `e60653896a978fb7a386c09c85f477cf98054082f41aa6745ea8d2538e0d1d19`
- `weights.zip`: `ef134a50bfade0431c97a71fd224dd833a56faeb79bf2ae14312f3a0be0bb13c`
- `teaset.ckpt`: `6feb5e1b00e17d662596fcc0e6abde686ebbbd48d5233a2100caa407780f5bbc`
- `seal.ckpt`: `9104c18b9ccaed9dd18c5c6ad2fe42cf9e4ffe48fd3382079d7beb0f065959fc`

The four teaset OBJ hashes are pinned in the locked protocol and checked by
`test_correspondence.py`.

### Runtime deviations from the upstream environment

These are execution-only changes. None of them changes model semantics,
weights, renderer estimators or the architecture.

1. **torch 2.8.0+cu128 instead of 2.3.1.** 2.3.1 has no sm_120 kernels, which
   was the cause of the worklog-19 abort. The upstream extension source is
   compiled unchanged.
2. **DrJit `VCallRecord` on for neural renders.** The notebook turns it off.
   With it off, DrJit 0.4.6's wavefront virtual-call dispatch never returns on
   this GPU once a render exceeds 8192 lanes: 64²×4 and 128²×4 time out, while
   4096 and 8192 lanes finish (`probe_vcall_dispatch.py`). At 4096 and 8192
   lanes and the same seed, on and off agree to 1.1e-5 linear. `LoopRecord`
   stays off as in the notebook.
3. **Native Windows instead of Linux/WSL.** Under WSL, torch and the extension
   work and DrJit's CUDA backend starts once pointed at
   `/usr/lib/wsl/lib/libcuda.so.1`. But DrJit 0.4.6 cannot load OptiX there:
   WSL's `libnvoptix.so.1` is a dxcore loader shim without
   `optixQueryFunctionTable` (`wsl/README.md`). Every Mitsuba 3.5 `cuda_*`
   variant needs OptiX.

This is therefore a documented compatibility environment with the released
model semantics. It is not the bit-identical official environment.

Two corrections to the record:
- Worklog 19's gate script used `neuralvolpath` for the seal demo. The
  released notebook's active line is `neuralpath`, and this session follows
  the notebook.
- Earlier in this session I attributed the first seal stall to reference
  cost. That was wrong. Both the seal and the first teaset stalls were the
  `VCallRecord` hang: once it was fixed, a 1024-spp seal reference took 7 s.

### Representation contract, verified in the pinned code

The existing `8DNA_REPRESENTATION_AUDIT.md` is factually correct and was not
rewritten. One execution detail matters for this experiment. In
`EightDNA.sample_asset`, when the BSDF-sampled ray from the first asset hit
lands on the same instance (another part included), the neural lobe is
always taken (`m = 1`). All intra-asset multi-bounce transport, part-to-part
reflections included, therefore comes from the frozen conditional
distribution `p(xo, wo | xi, wi)` and albedo. Direct emitter sampling at the
first hit, including occlusion by other parts, is recomputed from current
geometry. `xo` is an exit direction projected onto the asset bounding box
(the envelope). The asset has no part ID or surface ID input.

### Candidate assets inspected (selection without neural error)

All 11 released assets have matching checkpoints:
- **`teaset`**: four rigid rough-nickel meshes (tray, milk pot, biscuit tin,
  tea pot) in one shapegroup/instance with fixed topology. Exact analytic
  rigid correspondence is possible, the relation change is physical (sliding
  on a flat floor), and the GT is a surface-only `prb` render. **Chosen.**
- **`milk`**: a milk volume nested inside a dielectric glass. Relative
  motion would interpenetrate or leave the glass, so it is non-physical.
- **`seal`, `cat`, `dragon`, `candle`, `bunny`**: single mesh plus volume.
  A geometry change needs a validated volume warp.
- **`cloud`, `hair`, `curlhair`, `fabric`**: volume-only or curve assets.

teaset meets worklog 19's provisional conditions: a matching released
checkpoint, a reproduced baseline (below), and four parts that form the one
learned instance. The optional second asset was not attempted. No other
released asset offers physically separable rigid parts, so a second one
would be a new asset-conversion project.

### Correspondence contract (`teaset_parts.PartRigidAsset`)

- For a hit on part p translated by t_p, the adapter calls the unmodified
  upstream `eval_asset` / `sample_asset` with model-to-world `[I | t_p]`.
  Upstream then computes `xi_can = xi − t_p` and `wi_can = wi`, and the
  sampled `wo` is unchanged.
- The part is identified per lane from `si.shape`. The mesh pointers are
  found by probe rays aimed at each part's own vertices.
- Envelope, primary `attached`: the canonical asset box plus t_p. This is
  what upstream does for an instance translated by t_p. Declared
  sensitivity `fixed`: the canonical box, unmoved.
- Mode `upstream` is the released path with current-world queries. It is
  the T0 baseline and, at moved states, a declared coordinate-mismatch
  diagnostic, not a transport result.
- Nearest-XYZ correspondence is not used anywhere.

### Repository

- Experiment commit: **`126d6d8`**. It contains all code, the runtime
  scripts, the design revisions and the locked protocol, and was committed
  before the evidence run. `teaset_frozen_eval.py` refuses a dirty tree; the
  run recorded `project_dirty: []`.
- Presentation-only `teaset_review_exports.py` is committed with this
  worklog.
- Outputs (ignored) are under `results/8dna_replication/`. Review exports
  are in `results/evaluation/21/` with a SHA-256 manifest.

## MEASUREMENT

### Static baseline

Metrics are released 8DNA against a matched path-traced reference, display
space `clamp(x^(1/2.2))`.

| Asset / regime | 8DNA vs ref PSNR / SSIM | 8DNA seed repeat | ref A vs B | neural time | peak torch mem |
|---|---|---|---|---|---|
| seal `scene2`, official notebook protocol (256², 256 spp `neuralpath`); ref `prbvolpath` 2×32768 spp | 26.51 dB / 0.772 | 25.34 dB | 29.69 dB | 12.7 s | — |
| teaset `get_scene`, 512², 256 spp; ref `prb` 2×2048 spp | 29.13 dB / 0.891 | 34.66 dB | 43.22 dB | 25.4 s | 2.40 GB |
| teaset locked T0, 512², 512 spp | 29.41 dB / 0.905 | — | — | 61 s | — |

- Seal's 8DNA-vs-reference gap is at the level of the neural render's own
  seed-to-seed noise at the notebook's 256 spp.
- On teaset, the static error concentrates in part-to-part reflections: the
  pots seen in the tray look blurred and grainy. That is the released
  model's static accuracy, and it is subtracted when judging moved states.

### Correspondence gate (`correspondence_tests.json`, PASS)

- **Canonical identity.** Adapter (`attached` and `fixed`) against unmodified
  upstream, same seed, 256², 32 spp: max 7.9e-7 linear, display PSNR 163 dB.
  The seed-to-seed display MAE is 0.022.
- **Rigid pullback.** A probe state moved the milk pot +0.12 x and the tin
  +0.1 z. Every primary hit, pulled back, lies on its canonical part mesh
  (re-hit of the solo part: 100%, p99 position error ≤ 1.7e-7). All asset
  lanes are covered by a declared part.
- **Topology and materials.** In every state, vertex residual after
  pullback is 0, faces are identical, and all BSDF parameters equal
  canonical.

### Locked states (geometry: exact FCL mesh distances)

| State | Motion | milk pot–tea pot gap | rim clearance (min over sweep) |
|---|---|---|---|
| T0 | canonical | 0.187 | 0.050 |
| T1 | whole asset +z 0.15 | 0.187 (all relations unchanged) | 0.050 |
| T1b | milk pot +z 0.15 | 0.172 | 0.052 |
| T2 | milk pot +x 0.10 | 0.091 | 0.051 |
| T3 | milk pot +x 0.16 | 0.034 | 0.052 |

The moved pot keeps floor contact in every state, and the 64-step sweep
never touches another part. For scale, the canonical tin–tea pot gap is
0.035.

### Physical signal (GT only; `gt_design/v2`)

The interaction ROI is the stationary tea pot's surface within 0.15 of the
milk pot at T3, visible in every fixed state (4170 px). Display MAE is
measured against T0 with seed set A; repeat noise is A vs B.

| State | GT change | repeat noise | ratio | tea-pot→milk-pot visibility |
|---|---|---|---|---|
| T0 | 0 | 0.0072 | 0 | 0.139 |
| T1 | 0.0366 | 0.0072 | 5.1 | 0.139 |
| T1b | 0.0621 | 0.0071 | 8.8 | 0.157 |
| T2 | 0.0405 | 0.0074 | 5.5 | 0.207 |
| T3 | 0.0664 | 0.0076 | 8.8 | 0.266 |

The gate (> 3× noise in a non-control state) **passes** for T1b, T2 and T3.

Across processes the reference is not bit-reproducible: the locked
re-render and the design GT differ by display MAE 0.0023, about 1/3 of the
A-vs-B noise.

### Frozen 8DNA response

Mode `attached`, 512 spp, seed 0 for every state. All changes are paired
through surface correspondence. "Error" is display MAE against GT, compared
with the same surfaces at T0; dG and dN are the GT and model changes versus
T0; gain = ⟨dN, dG⟩ / ⟨dG, dG⟩ in linear RGB.

| State | interaction error (Δ vs T0) | dG | dN | gain | tray Δ | far Δ | mover Δ (gain) | full PSNR / SSIM |
|---|---|---|---|---|---|---|---|---|
| T0 | 0.0348 | — | — | — | — | — | — | 29.41 / 0.905 |
| T1 | 0.0339 (−5%) | 0.036 | 0.035 | 0.97 | −2% | +2% | −1% (0.96) | 29.35 / 0.905 |
| T1b | 0.0726 (+109%) | 0.062 | 0.005 | 0.03 | +34% | +10% | +52% (0.45) | 26.04 / 0.886 |
| T2 | 0.0547 (+57%) | 0.040 | 0.002 | 0.02 | +5% | −2% | +39% (0.45) | 27.74 / 0.894 |
| T3 | 0.0757 (+118%) | 0.066 | 0.004 | 0.02 | +11% | +2% | +66% (0.34) | 26.21 / 0.886 |

- **`fixed` vs `attached`**: every ROI value is identical to 1e-4.
- **`upstream` at moved states** (coordinate-mismatch diagnostic): the
  mover ROI error is +181% (T1b), +207% (T2), +289% (T3) and +194% (T1),
  against +39–66% under `attached`. Interaction values are identical,
  because the tea pot is stationary and its queries do not change.
- **Noise floors in the interaction ROI**: GT repeat 0.0070, neural seed
  repeat 0.0137. The model changes dN are small partly because every state
  shares seed 0.

## OBSERVATION

- In T1b, T2 and T3 the frozen model keeps the milk pot's canonical
  interreflection. Its mirror image on the tea pot's lower-left face and on
  the tray stays where the pot stood at T0, while in the path-traced
  reference the image follows the pot (`06`, `07`/`08` crops; `02`
  trajectory).
- A dark smear appears on the tray at the moved pot's base in T2 and T3.
- The tray ROI understates the visible tray error. The largest tray errors
  lie on pixels the pot covered at T0, and those are excluded from the
  "tray in every state" ROI.
- On the stationary tea pot, query inputs are identical to T0 by
  construction. The model's output there changes only through the
  physically recomputed direct term, giving gain ≈ 0.02 against a GT change
  of 5.5–8.8× noise.
- The moved pot's own surface is rendered plausibly under `attached`.
  `upstream` (current-world queries) visibly garbles it (`09`), and that is
  the coordinate-mismatch failure the adapter removes.
- T1, where the whole asset moved and relations are preserved, keeps its
  T0 error level. The model also follows the view-dependent GT change
  (gain 0.92–0.97).
- The far ROI (biscuit tin) stays within +10% in every state.
- Seal: jade appearance, subsurface glow and shape shading match the
  reference. The difference map is dominated by 256-spp neural speckle.

## INTERPRETATION

- **Decision rule.** Every relation-change state satisfies the locked
  failure condition: error rise ≥ 25% and gain < 0.5 (T1b +109%/0.03,
  T2 +57%/0.02, T3 +118%/0.02). The relation-preserving control stays
  below 25% (T1 −5%). This gives CROSS-MODEL GEOMETRY-CONFIGURATION FAILURE
  OBSERVED. TRANSPORT-LIKE follows from three conditions: the error rise
  sits where GT interreflection changed, the far ROI stays within 25%, and
  the two envelope conventions agree.
- **What the failure is.** It does not reduce to a coordinate mismatch. On
  the stationary surface the frozen queries equal the canonical ones
  exactly; the mismatch diagnostic is separately far worse on the moved
  part; and generic movement with preserved relations (T1) does not
  degrade. What remains is stale part-to-part transport. This matches the
  audit: in 8DNA, intra-asset transport after the first hit is owned
  entirely by the frozen distribution, so this outcome was structurally
  expected. The session measures it rather than discovering it.
- **The null control.** The pre-declared low-interaction control was not
  found. v1's T1a interpenetrates the rim, v1's T1b moves the pot's mirror
  image across the tea pot, and v2's T1 fails the 25% GT-change criterion
  (55%) through viewpoint parallax on near-mirror nickel. T1 is therefore
  reported as a relation-preserving motion control. It answers the
  generic-movement question the null was meant to answer, but not in the
  declared form. The redesign happened before any moved-state neural
  render, and both design revisions are committed.
- **Relation to RNA (phenomenon level only).** Both frozen asset-specific
  representations lose validity when geometry relations change, but the
  failure types differ:
  - RNA on Rain failed under non-rigid local deformation, with
    local/spatial confounds and a physically weak fixed-target probe.
  - 8DNA on teaset fails under rigid cross-part relation change with no
    local deformation, on a surface whose query inputs are unchanged, and
    the physical signal is strong.

  These results are not commensurate. Different assets, renderers and
  regimes are involved, and no PSNR is compared across them. No shared
  mechanism is claimed.
- **Project level.** The geometry-configuration phenomenon has begun to
  replicate outside RNA: one additional representation family, one asset,
  one deformation mechanism (cross-part approach). That fills the "8DNA ×
  articulated cross-part" cell of the evidence matrix. It is below the
  breadth the strategy document requires for project-level support.

## UNRESOLVED QUESTION

- Does the result hold beyond near-mirror conductors? On diffuse or glossy
  interreflection the stale component may be weaker, and it may be masked
  by the model's static blur.
- How does the error scale with gap below 0.034, down to contact? Contact
  release was not tested.
- What happens under part rotation, which would need an oriented-envelope
  convention?
- Would a GT-only design with a camera move make a true low-interaction
  null possible?
- What would a teaset refit at T3 show? That is the M2 control that
  separates stale state from capacity.
- Would a current-geometry-conditioned contrast (RenderFormer-type) follow
  dG? That is the next baseline role.
- Is the prb reference nondeterminism on this host kernel-level atomics or
  path divergence? It is below the repeat noise and does not affect the
  verdict.

## Reproduction

```text
pwsh experiments/8dna_deformation_replication/windows/setup_env.ps1
cd experiments/8dna_deformation_replication
pwsh windows/run.ps1 run_static_baseline.py --asset seal --scene-fn scene2 --res 256 --spp 256 --ref-spp 32768 --ref-chunk 64 --tag seal_scene2_official
pwsh windows/run.ps1 measure_teaset_geometry.py --states protocol/teaset_gt_design_v2.json --out <results>/gt_design/geometry_v2.json
pwsh windows/run.ps1 teaset_gt_states.py --protocol protocol/teaset_gt_design_v2.json --out gt_design/v2
pwsh windows/run.ps1 test_correspondence.py
pwsh windows/run.ps1 teaset_frozen_eval.py --protocol protocol/teaset_frozen_locked.json
pwsh windows/run.ps1 teaset_review_exports.py --protocol protocol/teaset_frozen_locked.json --worklog 21
```

Review exports in `results/evaluation/21/`:
- `01` T0 canonical GT vs 8DNA
- `02` trajectory GT / 8DNA / error
- `03` GT flicker T0↔T3
- `04` 8DNA flicker T0↔T3
- `05` T3 GT vs 8DNA
- `06`–`08` interaction-region crops and crop flickers
- `09` T3 query modes
- `10` ROI overlay
- `11` static baselines (teaset, seal)

Error maps are secondary evidence.
