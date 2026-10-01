# Worklog — Reference-Only Transport Decomposition of the Teaset Failure (2026-10-01)

## Session question

In the stationary interaction ROI of the accepted worklog-21/22 teaset
experiment, which physical light-transport paths carry the GT change that the
frozen 8DNA and RNA representations do not track? And therefore, what
information is missing or stale in those representations? (M2 attribution;
no method development.)

Answer source: the physical reference renderer only. No neural network was
executed in this batch.

## Scope corrections to earlier worklogs

Worklogs 21–23 are unchanged. Two statements in worklog 23 are narrowed here:

- **RNA sensitivity scope.** Worklog 23's defensible conclusion is only that
  delta-light training-target fireflies are not sufficient to explain RNA's
  near-zero tracking. Its sentence "the shared pattern is not explained by
  RNA's poor static reconstruction alone" overstates this: the area-trained
  RNA failed G1, so worklog 23 does not show that RNA's static quality is
  irrelevant. RNA remains supporting evidence; the 8DNA case is the primary
  causal anchor.
- **Cross-host agreement wording.** Worklog 23 called LabServer63 and RTX 5080
  dataset views "bit-identical". The correct statement is: numerically
  equivalent within float16 storage tolerance — every channel identical except
  one normal value that differed by one float16 step (0.0024).

## IMPLEMENTATION FACT

### Decomposition semantics

`transport_decomposition.py` is a project-owned diagnostic re-implementation
of Mitsuba 3.5.1's prb/path primal estimator: emitter sampling with visibility
plus BSDF sampling, power-heuristic MIS, Russian roulette from depth 5,
unbounded depth, run in wavefront mode. Every radiance contribution is added to
exactly one class. With v = the number of scattering vertices x1..xv between
the camera and the emitter (x1 = the visible point):

| class | definition |
|---|---|
| camera_emitter | v = 0 |
| direct | v = 1: light reaching x1 directly, including its current shadowing |
| x2_\<part\> | v = 2: one inter-surface reflection, by the part of x2 (plate, teapot2 = moved milk pot, teapot3, teapot4 = the ROI's own part, other) |
| higher_mover / higher_other | v ≥ 3, with / without the milk pot among x2..xv |

Part identity comes from the hit mesh pointer (`teaset_parts.part_shape_ids`),
never from image-space or XYZ proximity. Reported groups: direct;
mover_single (x2 = milk pot); stationary_single (x2 on any other part, in
practice the tray); higher_mover; higher_other.

The canonical reference renderer is not modified. Classes, groups, sampling,
validity conditions and the A–E decision rule were written into
`protocol/teaset_transport_decomposition.json` before any decomposition run.

### What was rendered

- Both accepted regimes: `common_light` (worklog 22: lighting rev 3, 15° area
  light, black world) and `w21_envmap` (worklog 21: upstream envmap).
- States T0, T1, T1b, T2, T3 from `teaset_frozen_locked.json`; the interaction
  ROI and its T0 pairing (`mask_interaction`, `idx0_interaction`) from the
  accepted ROI files. The ROI is 4170 pixels on the tea pot (teapot4) within
  0.15 of the milk pot's T3 position (T1: 4412 pixels, paired to T0).
- ROI pixels only, 32768 spp (common light) / 8192 spp (envmap), two
  independent seeds A and B per state; 42 s per state and seed on the 5080.
- Full 512² component images of T0 and T3 in both regimes (1024 spp, seed A)
  for review only.

### Estimator limitations

- The classes partition the *estimate*. Whether a class contribution is
  "caused" by the milk pot is read from path vertices, not from counterfactual
  re-rendering.
- Path classes are defined relative to x1. Outside the ROI, for pixels whose
  x1 is on the milk pot itself, "mover-mediated" has a different meaning; the
  full-frame panels show those pixels but the analysis does not use them.
- No MIS-free or bounce-limited variants were rendered, and occlusion is not a
  separate class (see the interpretation for how it is identified).

### Focused tests (`test_transport_decomposition.py`, `transport_decomposition/tests.json`, all pass)

| test | result |
|---|---|
| partition: per-sample class sum = total | max abs error ≤ 1.9e-6 (float32) |
| estimator vs Mitsuba C++ path, image-wide lattice, T3 | common light: MAE 0.0092 vs path's own seed repeat 0.0094, mean ratio 1.006; envmap: 0.0039 vs 0.0039, 0.998 |
| part identity | four distinct ids in every state and regime; probing the T3 scene at T0 positions fails for the milk pot (negative control) |
| state identity | scene translations equal the locked protocol; 100% of ROI first hits on teapot4 in every state |

Two test defects were fixed before the recorded run: the path reference had
to be rendered in spp chunks (one 512²×1024-spp wavefront launch exhausted GPU
memory), and the two path seeds had overlapping chunk-seed ranges, which
understated path's own noise.

## MEASUREMENT

### Accounting against the canonical references (ROI, per state)

| regime | diagnostic/reference mean ratio (all states) | diag-vs-refA display MAE | refB-vs-refA display MAE |
|---|---|---|---|
| common light | 0.9997–1.0003 | 0.0047–0.0051 | 0.0047–0.0051 |
| envmap | 0.9994–1.0008 | 0.0051–0.0056 | 0.0063–0.0068 |

All validity conditions pass. Shares sum to 1.000000 by construction; the
seed-A and seed-B shares differ by ≤ 0.002 for every group and state.

### ROI radiance at T0 (linear, mean over ROI)

| | direct | mover single | stationary single (tray) | higher, mover | higher, other | total |
|---|---|---|---|---|---|---|
| common light | 0.0000 | 0.0130 | 0.7566 | 0.0196 | 0.0210 | 0.810 |
| envmap | 0.0955 | 0.0128 | 0.0843 | 0.0169 | 0.0188 | 0.229 |

Under the common light the ROI receives no direct light; 93% of its radiance
is the tray's reflection of the light (tea pot → tray → light).

### Component changes versus T0

Signed mean change (linear) and projection share s = ⟨dC, dG⟩/⟨dG, dG⟩.
Repeat noise (T0, single-render A vs B, mean |·|): common light — direct 0,
mover single 0.0004, stationary single 0.0043, higher mover 0.0010, higher
other 0.0007; envmap — 0.0012, 0.0004, 0.0015, 0.0007, 0.0005.

**Common light**

| state | mean \|dG\| | direct | mover single | stationary single | higher, mover | higher, other |
|---|---|---|---|---|---|---|
| T1 (control) | 0.189 | 0.000 / 0.00 | +0.002 / 0.01 | +0.018 / **0.99** | −0.001 / −0.01 | +0.003 / 0.01 |
| T1b | 0.322 | 0.000 / 0.00 | +0.013 / 0.04 | **−0.276 / 0.97** | +0.001 / −0.01 | +0.003 / 0.00 |
| T2 | 0.066 | 0.000 / 0.00 | +0.005 / 0.14 | **−0.026 / 0.76** | +0.014 / 0.10 | −0.003 / 0.01 |
| T3 | 0.117 | 0.000 / 0.00 | +0.010 / 0.18 | **−0.048 / 0.71** | +0.026 / 0.10 | −0.004 / 0.00 |

**Envmap**

| state | mean \|dG\| | direct | mover single | stationary single | higher, mover | higher, other |
|---|---|---|---|---|---|---|
| T1 (control) | 0.053 | +0.006 / **0.89** | −0.001 / 0.04 | +0.005 / 0.06 | 0.000 / 0.00 | −0.001 / 0.01 |
| T1b | 0.048 | +0.002 / 0.02 | +0.001 / **0.48** | −0.008 / **0.44** | +0.002 / 0.03 | −0.001 / 0.03 |
| T2 | 0.033 | +0.001 / 0.02 | +0.008 / 0.27 | −0.017 / **0.50** | +0.011 / 0.06 | −0.005 / 0.17 |
| T3 | 0.055 | +0.001 / 0.02 | +0.015 / 0.34 | −0.023 / **0.39** | +0.018 / 0.09 | −0.010 / 0.17 |

Every listed group change in T1b–T3 exceeds 3× its own repeat noise, except
the direct group: exactly zero under the common light (the ROI receives no
direct light), 2.1× at envmap T2. Within stationary single, the tray carries all of
the change (x2 on teapot3/teapot4/other contributes < 0.5% in every state).

### Decision (predeclared rule)

| regime | T1b | T2 | T3 (batch answer) |
|---|---|---|---|
| common light | D (stationary single, mover single) | D | **D — MIXED** |
| envmap | D (mover single, stationary single) | D | **D — MIXED** |

No group reaches the 0.60 share required for A (mover-mediated: common light
0.28 at T3, envmap 0.43), B (direct: 0.00 / 0.02) or C (higher order: 0.10 /
0.26).

Records: `results/8dna_replication/transport_decomposition/decomposition.json`
(per state, group and class), per-run `roi_<state>_<seed>.npz`. Reviewer
panels: `results/evaluation/24/` (raw component images T0/T3, signed
differences, interaction crops, T0↔T3 flickers, ROI overlay; manifest with
sources and SHA-256).

## OBSERVATION

- The GT change at the stationary tea pot is **not** a direct-light or
  direct-shadow change: direct carries 0.00 (common light) and 0.02 (envmap)
  of dG in every relation-changing state.
- It is a mixture of two signed effects of opposite sign:
  1. **Loss of tray-reflected light** on paths that never touch the milk pot
     (stationary single −0.008 to −0.276; plus higher-other −0.001 to −0.010
     in the envmap);
  2. **Gain of light reflected by the milk pot** (mover single plus
     higher-mover, +0.003 to +0.035).
  Under the common light (1) dominates (share of all stationary paths
  0.72–0.97); under the envmap the two are comparable (0.49–0.68 vs
  0.32–0.51).
- The predeclared hypothesis "moved-part interreflection dominant" (outcome A)
  is not supported in either regime.
- T1 (whole asset moved, relations preserved) changes almost entirely through
  stationary single under the common light (0.99) and through direct under the
  envmap (0.89). The frozen models track T1 (worklogs 21/22: gain 0.70–0.95),
  so the same path class that fails in T1b–T3 is tracked when its change comes
  from a relation-preserving motion.

## INTERPRETATION

- **Occlusion is identified without an extra class.** In T1b–T3 the tea pot,
  the tray and the light are stationary. For a path whose vertices are all
  stationary, the BSDF terms, geometry terms and emitted radiance are
  unchanged, so its expected contribution can change only through the
  visibility of its segments — that is, only through occlusion by the milk
  pot. The stationary-path change is therefore the milk pot shadowing the
  tray-reflected (and envmap multi-bounce) light that reaches the tea pot.
  Combining the classes this way, at T3: **occlusion of stationary indirect
  transport ≈ 0.72 (common light) / 0.58 (envmap); light reflected by the milk
  pot ≈ 0.28 / 0.43; direct ≈ 0.00 / 0.02.** This combination is derived after
  the measurement; the predeclared label is D (mixed).
- **What changes physically** is the relation between the query point and a
  different part: (a) which of the query point's indirect incidence paths are
  blocked by the moved part, and (b) the radiance the moved part sends toward
  it. Both are non-local: neither depends on the query point's own surface.
- **What the frozen models already receive at inference:** RNA gets the query
  point's (pulled-back) position, normal, view and light direction and the
  *current direct* light visibility; 8DNA gets the per-part correspondence of
  the query and evaluates direct emitter visibility with current geometry. The
  current direct-visibility input covers the one component that carries
  essentially none of dG here. This is consistent with — and explains — the
  observation in worklog 22 that RNA fails despite current visibility.
- **What is absent or frozen:** the cross-part transport state — inter-part
  occlusion of indirect light and the moved part's reflected radiance — is
  fixed in the canonical configuration inside both networks' parameters
  (8DNA's intra-asset neural lobe; RNA's position-indexed triplane).
- **Consequence for M3.** In T1b–T3 the ROI's query points are on a stationary
  part, so their current local surface state (position, normal, frame,
  same-part neighbourhood) is identical at T0 and T3. An H1 representation
  (persistent state + current local surface state, no cross-part data)
  therefore receives identical inputs in the ROI and cannot change its output
  there. This follows from the definitions, not from an experiment, and it
  applies to this ROI only. The first discriminating M3 test is accordingly an
  H2 test, and the decomposition specifies its information target:
  1. current visibility of the query point's **indirect** incidence (for
     example, whether directions toward the tray are blocked by another part)
     — the larger share;
  2. the **other part's presence and outgoing radiance** toward the query
     point (relative position/orientation of the moved part) — the remaining
     share.
  A test that supplies only (1) predicts recovery of ≈0.7 (common light) /
  ≈0.6 (envmap) of dG; supplying both predicts full recovery. Direct-visibility
  conditioning alone predicts none. No representation is designed here.
- **M2 judgment:** the missing information is identified as **cross-part
  relational transport state — nonlocal visibility of indirect light plus the
  moved part's reflected radiance — not direct visibility and not local
  surface state**, in the one tested asset and two lighting regimes. This is
  narrow enough to define the first M3 experiment without guessing an
  architecture.

## UNRESOLVED QUESTION

- Breadth: one asset family (`teaset`), near-mirror nickel, rigid
  translations. Whether diffuse or rough materials shift the balance toward
  reflection or occlusion is not measured.
- The occlusion/reflection split is derived from vertex classes plus the
  stationary-path argument; a counterfactual render (e.g. the milk pot made
  invisible to camera paths but kept as an occluder) would measure the two
  effects directly. Not run.
- Whether an H2 input of type (1) can be supplied cheaply enough to keep
  persistent reuse worthwhile is an M3/M7 question.
- The interaction ROI was chosen by proximity to the mover; other stationary
  regions (e.g. the tray around the milk pot) were not decomposed.

## Commits

| output | commit |
|---|---|
| protocol, diagnostic estimator, tests (code) | `92506ee` |
| reviewer-panel code | `19e76d8` |
| ROI and image renders | runs recorded at `19e76d8` / `2b6cad5`; `transport_decomposition.py` identical to `92506ee` (only a summary print changed later, `17c57c9`) |
| analysis (`decomposition.json`) and `results/evaluation/24/` | `17c57c9` |
| tests run (`tests.json`) | `eb26829` |

Execution host: RTX 5080 for everything (LabServer63 not used).
