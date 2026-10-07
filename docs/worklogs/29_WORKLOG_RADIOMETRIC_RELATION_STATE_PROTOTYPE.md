# Worklog — Second Bounded Prototype: Radiometric Content on the Same Sparse Relations (2026-10-07)

## SESSION QUESTION

Holding geometry sampling and training coverage fixed, does adding aligned,
runtime-constructible radiometric content to worklog 28's current sparse
relations make them sufficient to track the held-out T3 transport change? If
not, is the failure the runtime proxy's, or does even exact radiance on the
same K = 32 support fail?

One factor changed relative to worklog 28: each current probe hit carries an
RGB radiometric value. Everything else is worklog 28's: K = 32 directions,
query frames, ray casting, the 20-channel geometric descriptor, the canonical
pull-back, the T0/T1b/T2 training and T1/T3 hold-out split, the operator
class and widths, the 4000-step budget, seeds 0/1/2, the residual target and
the loss.

**Answer (canonical teaset ROI, common light, RTX 5080): no, and the oracle
does not rescue it. This is CASE D.**

- The real runtime radiometric candidate fails every predeclared success
  criterion. Held-out T3 R_delta is 0.98 / 0.94 / 0.92, against ZERO
  0.97 / 0.99 / 0.87, SHUFFLED 0.95 / 0.96 / 0.88 and worklog 28's geometric
  branch 0.93 / 1.12 / 0.91.
- The predeclared exact-direction oracle (path-traced incident radiance
  along the same 32 directions) also fails: R_delta 0.92 / 0.94 / 0.87.
- K = 32 angular support, aggregation and training-relation coverage
  therefore remain entangled. Per protocol the batch stops here.

## SETUP (step 0)

- **Repository.** Worklog 28 and every evidence commit were already on
  `origin/main` (`52ba954`). Fetching showed no new remote commits and no
  local-only commits.
- **Living documents.** They already recorded worklog 28's CASE C:
  geometry-only sparse state insufficient, local-only insufficient,
  lifecycle split kept, next question a runtime radiometric state
  (`52ba954`). No setup edit was needed.
- Worklog 28 and its code (`experiments/relational_residual_prototype/`)
  were not modified.

## IMPLEMENTATION FACT

### Audit: is the proxy constructible from the existing substrate?

Yes, with no new appearance model. RNA's renderer (`rna_infer.py`, canonical
mode) evaluates `forward(position, camera_dir, light_dir, normal)` with:

- the canonical pull-back normalised by the training AABB;
- the hit → camera direction;
- the camera-side normal;
- 16 area-light samples, whose per-sample visibility selects the lit or
  shadow branch, weighted by E / training intensity.

The proxy **L_hat_{j→i}** is exactly that renderer at the current probe hit
j, with the query i acting as the camera:

- position: j's canonical pull-back, as in worklog 28;
- camera_dir: j → i (minus the probe direction);
- normal: j's current shading normal faced toward i;
- light: 16 area-light samples at j from the existing
  `rna_bridge.area_light_samples` on the current lit scene, with the strata
  taken at their centres so the samples are deterministic;
- encoding: log1p(max(RGB, 0)) on hits; misses carry 0, with the existing
  `hit` channel as the explicit miss flag.

**What it contains:** the frozen RNA's persistent appearance at j, including
whatever canonical transport its triplane and MLP baked in, evaluated for the
current direction toward the query under current direct visibility at j.

**What it does not contain:** current indirect light at j beyond RNA's
visibility branch switch, any recursive relation, path tracing, references,
state identity or transforms. It is a one-hop runtime radiometric proxy, not
exact incident radiance.

Self-check: the same function applied to worklog 28's query samples
reproduces worklog 28's frozen base predictions **exactly**, with max
difference 0.0 in all six buffers.

### Branches (protocol `rrs_v1.json`, committed before any proxy computation or training)

| | probe channels | trainable params | proxy channels |
|---|---|---|---|
| A geometric (worklog 28, predictions reused) | 20 | 33 187 | none |
| B ZERO | 23 | 33 379 | 0 everywhere |
| C SHUFFLED | 23 | 33 379 | hit values permuted within each state (`default_rng(29029)`), misses 0 |
| D REAL | 23 | 33 379 | aligned proxy |
| O oracle (diagnostic, run because D failed) | 23 | 33 379 | log1p path-traced incident radiance along the same 32 directions (Mitsuba C++ path, wavefront, 256 spp; seed A trains, seed B checks noise) |

B, C, D and O differ only in the semantics of the three channels: same model
class (`rrp_model.RelationalResidual`), widths, optimiser, steps, split and
seeds. A differs by the first layer's 3 × 64 input weights.

### Worklog 28 reproduced and preserved

- **Probe state.** The unchanged `rrp_probes.cast_probes`, re-run on the
  stored queries, reproduces worklog 28's probe arrays bit for bit in all six
  buffers. The stage stops otherwise.
- **Training loop.** This batch's loop, run on the 20-channel descriptor with
  seed 0, reproduces worklog 28's relational_s0 validation curve exactly
  (max difference 0.0, best step 3900, best validation loss 0.10079).
- **Scenes.** Probe rays re-cast in the lit scene land on the same points as
  in the geometry-only scene (Δt = 0, Δnormal ≤ 3e-8).
- **Frozen RNA.** Its parameter digest is unchanged before and after.

### Tests (focused; isolated sidecar, so no full regression)

| suite | result |
|---|---|
| `test_rrs_common.py`: channel layout, encoding, ZERO zeroes only the proxy channels, SHUFFLED keeps the multiset of hit values / keeps misses at 0 / destroys alignment (< 2% of positions keep their value) / is deterministic, unknown variant refused, **source scan: the primary state builders never read references, decompositions, oracle files, mover names or state-keyed dicts**, training/model contract equals worklog 28 | 8/8 pass |
| `test_rrs_light_win.py` (synthetic): fixed plane, fixed remote sphere, blocker moving out of the sphere's light path. Same probes hit the same remote points; remote lit fraction changes 0.000 → 0.750; samples deterministic | 4/4 pass |
| `test_rrs_wsl.py`: proxy renderer reproduces worklog-28 base (to float tolerance on a 64-sample batch; the stage's full-array check is exact); responds to view, normal and visibility; deterministic; frozen RNA digest unchanged, no grad; hold-outs refused; only the first layer differs between 20 and 23 channels; state serialisation | 6/6 pass |
| smoke runs (20-step untrained models; oracle at 4 spp) | pass; found a negative decoder timing (a difference of two noisy medians), fixed by timing the decoder directly |

### Commits and environment

| commit | content |
|---|---|
| `c28845e` | protocol, proxy, controls, oracle, evaluation, tests. **All evidence runs** in `results/radiometric_relation_state_prototype/v1_c28845e` (stages, training, oracle radiance, oracle training, evaluation and exports); clean tree throughout |

Environment: RTX 5080 (driver 596.49), Windows 11, Mitsuba 3.5.1 / Python
3.11.9 (light sampling, oracle, evaluation); WSL Ubuntu-22.04, torch
2.8.0+cu128 (proxy, training). RNA upstream `66b5b09`; frozen checkpoint
`epoch=187-val_psnr=18.87dB.ckpt` (sha256 `81063b28…`). The previous session
ended while the oracle stage was running. The detached processes survived
and completed; only the watcher was lost.

## MEASUREMENT

### Primary metric: held-out T3 transport-change error (display space)

All values are on stable T3 pixels paired with identical T0 queries.
∣dG∣ = 0.0935.

| branch | R_delta s0 / s1 / s2 | E_delta s0 / s1 / s2 | T0 static error | T3 abs error |
|---|---|---|---|---|
| frozen RNA | 1.025 | 0.0959 | 0.089 | 0.124 |
| A geometric | 0.928 / 1.121 / 0.915 | 0.087 / 0.105 / 0.086 | 0.032–0.034 | 0.082–0.108 |
| B ZERO | 0.972 / 0.989 / 0.867 | 0.091 / 0.093 / 0.081 | 0.030–0.068 | 0.079–0.102 |
| C SHUFFLED | 0.948 / 0.961 / 0.880 | 0.089 / 0.090 / 0.082 | 0.028–0.056 | 0.080–0.091 |
| **D REAL** | **0.978 / 0.938 / 0.924** | 0.092 / 0.088 / 0.086 | 0.033–0.050 | 0.085–0.087 |
| O oracle (diagnostic) | 0.924 / 0.941 / 0.868 | 0.086 / 0.088 / 0.081 | 0.030–0.031 | 0.084–0.087 |

Linear-space R_delta is 1.03–1.36 for every branch. No branch or seed comes
near 0.5 in either space.

### Predeclared success test (D)

| criterion | result |
|---|---|
| 1. every seed R_delta(D) < R_delta(ZERO) | **no** (s0 0.978 > 0.972; s2 0.924 > 0.867) |
| 2. every seed R_delta(D) < R_delta(SHUFFLED) | **no** (s0 0.978 > 0.948; s2 0.924 > 0.880) |
| 3. ≥ 2/3 seeds R_delta(D) ≤ 0.50 | **no** (0/3) |
| 4. signed change map follows the moved structure better than A/B/C | **no** (see OBSERVATION) |

**D FAILS.** Oracle case: **O2** (0/3 seeds ≤ 0.50).

### Secondary (historical compatibility)

Own-T0 refit ratio and gain: D 1.74 / 2.50 / 2.53 with gain 0.49 / 0.63 /
0.50; the oracle 2.76–2.81 with gain 0.61–0.63. No branch satisfies the
worklog-22 rule.

### In-sample changes (training states, all pixels)

R_delta at T1b and T2:

| branch | T1b | T2 |
|---|---|---|
| A | 0.38 / 0.39 / 0.39 | 0.56 / 0.58 / 0.59 |
| B | 0.38 / 0.62 / 0.37 | 0.59 / 0.86 / 0.53 |
| C | 0.41 / 0.55 / 0.34 | 0.60 / 0.82 / 0.55 |
| D | 0.58 / 0.37 / 0.38 | 0.90 / 0.57 / 0.57 |
| O | 0.38 / 0.36 / 0.32 | 0.54 / 0.57 / 0.49 |

The radiometric channel does not improve the *in-sample* change fit either.

### Held-out T1 (relation preserved)

| | T1 abs error | own-T0 error | rise | gain | correction difference T0→T1 | correction norm T0 / T1 |
|---|---|---|---|---|---|---|
| frozen | 0.092 | 0.090 | +1% | 0.78 | 0 | 0 / 0 |
| A s0 | 0.050 | 0.035 | +46% | 0.51 | 0.039 | 0.085 / 0.089 |
| B s0 | 0.053 | 0.035 | +53% | 0.49 | 0.041 | 0.080 / 0.091 |
| C s0 | 0.056 | 0.038 | +46% | 0.52 | 0.042 | 0.084 / 0.095 |
| D s0 / s1 / s2 | 0.056 / 0.048 / 0.052 | 0.050 / 0.034 / 0.034 | +12% / +39% / +52% | 0.68 / 0.50 / 0.53 | 0.040 / 0.040 / 0.042 | s0 0.077 / 0.083 |
| O s0 | 0.051 | 0.031 | +60% | 0.63 | 0.043 | — |

The GT T0 → T1 change is 0.044. Every residual branch changes its correction
between T0 and T1 by about the same amount (0.039–0.043). Adding aligned
radiometric content does not create an extra relation-specific correction at
T1. The rises again reflect in-sample vs held-out-pixel static fit (worklog 28).

### State diagnostics

| | T0 | T1b | T2 | T3 |
|---|---|---|---|---|
| probe hit fraction | 0.516 | 0.517 | 0.534 | 0.554 |
| moved-part hits / query (mean, p90, max) | 3.1, 4, 6 | 3.4, 4, 6 | 5.2, 7, 10 | 7.3, 12, 16 |
| proxy luminance on hits (p50, p90, p99) | 0.03, 1.91, 11.1 | 0.02, 0.57, 9.2 | 0.03, 1.33, 10.9 | 0.03, 1.08, 10.9 |
| probes whose proxy changes vs T0 | — | 33% | 27% | 32% |
| per-query proxy change norm (encoded) | — | 3.11 | 1.75 | 2.40 |
| Pearson / Spearman: per-pixel proxy change vs GT change | — | 0.23 / 0.15 | 0.02 / 0.09 | 0.16 / 0.16 |

T3 extrapolation:

- No hit-probe proxy value lies outside the training-state range in any
  channel.
- Per query: 48% of T3 queries have a summed proxy outside their own training
  range and 74% have more moved-part hits than any training state. These
  overlap as 35% both, 13% radiometric only, 39% geometry only.

Oracle noise:

- per-probe repeat noise 0.012–0.016 (encoded), per-query 0.005;
- T0 → T3 change of the oracle state 0.089, i.e. 5.6× the per-probe noise
  (T1b 7.7×, T2 3.8×).

### Cost (T3; Q = 66 144, K = 32, QK = 2 116 608, 1 171 903 hits)

| stage | time |
|---|---|
| probe rays (worklog 28) | 5.8 ms |
| remote light sampling (lit-scene re-cast + 16 emitter samples and shadow rays per hit, 18.75 M shadow rays, incl. host arrays) | 1288 ms |
| proxy evaluation (18.75 M frozen-RNA queries, GPU-resident) | 1845 ms |
| persistent feature lookup (worklog 28) | 3.5 ms |
| encoder + aggregation | 13.3 ms |
| **dynamic-state update total** | **3156 ms** |
| residual decoder | 0.46 ms |
| frozen RNA rendering of the ROI queries (1.06 M queries) | 106 ms |
| oracle radiance (diagnostic, 2 × 256 spp) | ~155 s per state |

- Dynamic state: 32 floats per query (8.47 MB).
- Raw descriptors: 195 MB (float32).
- Training: 12.9–13.6 s per branch, peak about 1.2 GiB.
- The proxy raises the state update from about 16 ms (worklog 28) to about
  3.2 s. Per evaluated point it costs about the same as the frozen render
  (≈1.6 µs per point with 16 light samples each), but there are 1.17 M hit
  points against 66 k queries. This is a one-hop network evaluation, not path
  tracing. The implemented structure stays O(Q·K) per update, with a fixed 16
  light samples per hit.

## OBSERVATION

- **Signed T3 change maps** (`01_T3_signed_change_maps.png`,
  `02_T3_change_maps_all_seeds.png`). The reference change on the tea pot is
  a set of sharp, interleaved bright and dark bands: the moved milk-pot
  reflection and tray light newly blocked by it.
  - Every branch shows nearly the same pattern: a broad brightening over the
    part of the ROI facing the milk pot and a single dark band along the
    edge of the unchanged oval. That includes A, B, C, D, the oracle, and
    all three seeds of each.
  - None reproduces the band structure. D's maps are not closer to the
    reference than ZERO's or SHUFFLED's, and the oracle's are not either.
- **Absolute T3 images** (`03`). The same static correction appears in every
  residual branch, and the stationary regions outside the ROI are untouched
  by construction.
- **T1** (`04_T1_change_and_correction.png`). All residual branches show the
  same correction pattern at T0 and T1. There is no aligned-radiometry-
  specific correction where the relation is preserved.

## INTERPRETATION

- **Is aligned runtime radiometric content causal here? Not shown.** REAL,
  ZERO and SHUFFLED are statistically indistinguishable on T3, on the
  training-state changes and on T1. This is the CASE-B pattern within the
  D comparison: the "missing radiometric channel" hypothesis, as realised by
  this proxy on this support, is weakened.
- **The proxy changes, but not in the way the transport does.** A third of
  probe proxies change between states, but the change correlates weakly with
  where the reference changes (r ≈ 0.02–0.23). On near-mirror nickel, the
  query's outgoing radiance depends on incident light from a narrow mirror
  lobe. A frozen-RNA rendering of 32 uniformly spread remote points mostly
  reports the remote appearance, which includes its own baked canonical GI.
- **The exact-radiance oracle on the same 32 directions also fails (O2).**
  With the true current incident radiance along each probe, above noise, the
  same operator still leaves about 90% of the T3 change unexplained and does
  not improve the in-sample fit over the geometric branch. The bottleneck is
  therefore **not only** the runtime proxy. One or more of these is
  insufficient for this near-mirror case, and this batch cannot separate
  them (as the protocol anticipated):
  1. the sparse angular support: K = 32 uniform directions against a narrow
     specular lobe;
  2. the permutation-stable mean+max aggregation over those directions;
  3. training-relation coverage: two changed training configurations, both
     weaker than T3; 74% of T3 queries lie outside the training range of
     the geometric state.
- **Contrast with worklog 28's oracle.** That oracle gave the decoder the
  query's own reference *path-class radiance*, a pixel-level decomposition
  that sums to the answer, and it succeeded. Exact per-direction incident
  radiance on 32 directions is a different and much weaker kind of state,
  and it fails. Worklog 28's oracle showed the operator can use the answer
  when given it nearly directly. Worklog 29 shows that coarse directional
  incidence is not enough for this operator to reconstruct the change. The
  two results are consistent.
- **The lifecycle split is not falsified.** Every branch kept the frozen
  persistent RNA fixed and corrected through a shared operator. Nothing here
  shows that persistent appearance must change. What failed is every current
  state built on this K = 32 uniform support with this aggregation and this
  training coverage.
- **Not claimed.**
  - That radiometric relation content is useless in general.
  - That K = 32 is the cause on its own.
  - Any result beyond the canonical teaset ROI, its material and its
    relation family.
  - Any scene-scale complexity.

## FINAL REPORT (batch order)

1. **Worklog 28 reproduced and preserved?** Yes. Probe state bit-identical in
   all six buffers, the training curve reproduced exactly, frozen RNA
   unchanged, worklog-28 code and evidence untouched.
2. **The proxy.** Frozen RNA rendered at the current hit toward the query
   (canonical pull-back, j → i view, current faced normal, 16
   stratum-centred area-light samples with current visibility), log1p RGB.
   It contains persistent appearance plus baked canonical transport under
   current direct visibility. It does not contain current indirect light,
   recursion, path tracing, references or state identity.
3. **Did the state change?** Yes. 27–33% of probe proxies change vs T0, and
   the per-query change norm is 1.7–3.1. The change correlates only weakly
   with the reference change.
4. **Did ZERO and SHUFFLED behave like the geometric branch?** Yes: R_delta
   0.87–0.99 against A's 0.91–1.12, with similar in-sample fits and T1
   behaviour.
5. **Did aligned radiometry improve T3 change error in every seed?** No. It
   is better than ZERO only in s1, and better than SHUFFLED only in s1.
6. **R_delta ≤ 0.50?** No, 0 of 3 seeds (0.98 / 0.94 / 0.92).
7. **Held-out T1?** No aligned-radiometry-specific correction. The T0 → T1
   correction difference is 0.040–0.042 for D vs 0.039–0.043 for the
   controls.
8. **Do the signed maps recover the moved structure?** No. Every branch shows
   the same broad brightening and one dark edge band; the bands are missing.
9. **Cost?** State update about 3.16 s (light sampling 1.29 s, proxy RNA
   1.84 s, worklog-28 parts about 22 ms), decoder 0.46 ms, ROI frozen render
   0.11 s. The proxy costs about 200× the worklog-28 state.
10. **Exact K = 32 oracle?** Also fails (O2): R_delta 0.92 / 0.94 / 0.87.
11. **Which explanation is supported?**
    - Runtime proxy insufficient: yes, but not the *only* insufficiency.
    - Radiometric alignment useful: not shown.
    - K = 32 angular support / aggregation / training-range generalisation:
      unresolved and entangled (O2).
    - Higher-level lifecycle split weakened: no.
12. **Living-document change.** Record CASE D. The bottleneck is not
    attributable to the radiometric content alone; angular support,
    aggregation and training coverage are now the open, entangled
    questions. The lifecycle split is retained.

## UNRESOLVED QUESTION

- Which of the three entangled factors matters for near-mirror transport:
  angular support (uniform K = 32 vs the specular lobe), aggregation, or
  training-relation coverage? Each is a separate hypothesis needing its own
  batch. None was attempted here, per protocol.
- Would the same contract behave differently on a diffuse or glossy material,
  where incident light is integrated over a wide lobe? Untested.
- The proxy's cost (≈3 s for 1.17 M hits × 16 light samples) would need to
  fall by about two orders of magnitude even if it were informative.

## ARCHITECTURE DECISION

**CASE D.** Adding radiometric content to worklog 28's sparse relations,
whether as a runtime frozen-RNA proxy or as exact path-traced incident
radiance, does not make them sufficient on the canonical T3 case. The batch
stops without patching. The persistent / dynamic-state / shared-operator
lifecycle split is retained (not falsified). The current-state question is
reframed: the minimum dynamic state must resolve the transport-relevant
angular structure for this material, and K = 32 uniform support,
mean+max aggregation and two-configuration training coverage remain
unresolved and entangled. No third architecture was started.
