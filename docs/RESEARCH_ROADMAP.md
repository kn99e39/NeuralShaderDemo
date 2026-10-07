# Research Roadmap — Dynamic Neural Shading / Neural Light Transport

**Last updated:** 2026-10-07

## Document Role

This document defines the research roadmap, milestone gates, stop conditions, and current decision frontier for the project.

It complements:

- `RESEARCH_CENTRIC_TOPIC.md` — what problem the project is fundamentally about;
- `BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md` — how baseline roles, evidence breadth, and attribution scope are separated;
- `Architecture.md` — the current working method architecture and representation-ownership contract.

This is a living research document. Worklogs remain append-only historical evidence; this document records the current interpretation and next decision frontier.

Do not treat a working architecture direction as a validated method.

---

# 0. Central Research Intent

The project investigates how high-quality neural light-transport representations should separate information with different validity lifecycles.

## Persistent information

Information that may remain reusable across geometry changes, for example:

- material identity;
- surface identity;
- local appearance / microstructure;
- reusable local scattering behavior;
- reusable transport rules.

## Configuration-dependent transport

Information whose validity depends on the current geometry configuration, including:

- mutual visibility;
- cross-surface occlusion;
- proximity/contact relationships;
- radiance arriving from other surfaces;
- inter-part interreflection;
- geometry-conditioned higher-order transport.

The core question is:

> **What information should remain persistent, and what transport information must be updated from the current geometry configuration so that high-quality neural transport remains reusable without full regeneration?**

Do not reduce this prematurely to MLP vs geometry, graph vs transformer, SH vs learned basis, or any one implementation mechanism.

---

# 1. Current Working Hypothesis

The evidence now supports a more concrete working hypothesis:

> **Persistent neural assets become invalid when configuration-dependent inter-surface transport is stored as persistent state learned from one fixed internal geometry configuration.**

The current method hypothesis is:

> **Reusable surface/material information should remain persistent, while configuration-sensitive inter-surface transport should be represented by a compact current state that can be updated from current geometry and interpreted by a shared transport/scattering operator.**

Internal shorthand:

> **Persistent appearance, dynamic transport state, reusable transport rule.**

This is now the working architecture direction, not yet a validated method.

---

# 2. Current Project State

## Established

### Failure existence

- RNA/Rain showed a real frozen-deformation failure, though with local/spatial confounds.
- The Rain fixed-target causal probe closed as **PHYSICAL EFFECT TOO WEAK** and did not decide the nonlocal mechanism.
- Worklog 21 produced the first clean controlled 8DNA teaset case with a stationary target query, changed surrounding part relation, strong physical transport change above noise, frozen near-zero tracking, and a stable relation-preserving control.
- Worklog 22 reproduced the same frozen-failure signature with RNA on the same teaset protocol under common light.
- 8DNA same-state T3 refit under the original envmap regime recovers close to T0 quality, strongly supporting stale persistent state rather than insufficient representation capacity in that tested regime. Worklog 27 reproduced this recovery in a second, independently timed rebuild (last checkpoint 1.149 / 0.629 vs the historical 1.073 / 0.698).
- Same-state refit in the common-light regime is near-threshold for both backbones and not robust: 8DNA fails the rule in both runs (worklogs 22, 27), and RNA's worklog-22 borderline pass (1.22) did not recur in the worklog-27 rebuild (≈1.32, no checkpoint passes; datasets bit-identical, training run differs).

### Full-recomputation cost

- **Physical (worklog 26):** BMW27, Cycles, RTX 5080, after a rigid change: ≈0.22 s / 0.76 s / 3.0 s per 1080p frame at 16 / 64 / 256 spp (persistent data); geometry sync is ≈2 ms, the cost is sampling (≈11.5 ms/spp).
- **Neural (worklog 27):** teaset T3, historical refit pipelines (scratch rebuilds; no supported warm start exists in either method), RTX 5080:
  - 8DNA: first rule-satisfying checkpoint after **38.9 min** (oracle-identified, not stable until 77 min); full 30-epoch schedule **1.76 h**, 89% optimisation.
  - RNA: **3.24 h**, 87% of it rendering the T3 training targets; no checkpoint satisfies the recovery rule.
  - Every value is offline: 10⁵–10⁶ frames of a 30/60 FPS budget. Different scene from worklog 26, so cost context only, never an equal-quality ratio.

### Physical attribution

Worklog 24 closed the main physical attribution question for the canonical teaset ROI.

The missing GT change is not primarily direct-light transport. At T3 it is explained mainly by:

1. **nonlocal occlusion of stationary indirect transport** — approximately 0.72 of the change under common light and 0.58 under envmap;
2. **radiance reflected from the moved milk pot** — approximately 0.28 under common light and 0.43 under envmap;
3. direct transport contributes approximately 0.00 / 0.02.

These shares are physical attribution measurements, not predictions of neural recovery.

### Code / methodology attribution

The subsequent RNA/8DNA code-level audit established that:

- both methods train inter-surface transport from one fixed internal configuration;
- RNA absorbs full-GI radiometric effects into position-indexed learned state plus MLP weights, with direct visibility used only as a current branch selector;
- 8DNA learns fixed-configuration multi-vertex intra-asset transport in its persistent conditional distribution;
- on the stable-first-hit subset of the canonical ROI, the learned query geometry is identical between T0 and T3;
- current direct/current-geometry checks exist, but neither learned transport component receives the current internal configuration needed to update the missing cross-part transport.

For the tested failure class:

> **stationary query + changed cross-part relation inside one persistent learned asset**

the failure is best treated as a **methodology / representation-contract consequence**, not an implementation bug.

### Research-evidence gate

The project now has enough controlled evidence to proceed to method design.

This does **not** mean broad generality or paper-level closure has been established.

## Current limits

- The strongest clean causal anchor is still one asset family (`teaset`).
- The cleanest mechanism is rigid cross-part relation change, not yet non-rigid deformation/contact-release generality.
- The material regime is near-mirror rough nickel.
- RNA remains supporting rather than co-equal causal evidence because its static teaset reconstruction margin is weak.
- The minimum sufficient dynamic transport representation has not been established.
- Both full-recomputation baselines are now calibrated (physical, worklog 26; neural, worklog 27), but the proposed method's own update cost does not exist yet, so no advantage has been measured.
- The neural baselines' rebuild cost was measured only for their fixed historical schedules; faster rebuild variants (shorter schedules, warm start, smaller target sets) are untested.
- The working architecture has not yet been implemented or validated.

---

# 3. Research Progression

```text
M0  Research Contract                         CLOSED
 ↓
M1  Failure Existence / Direct Evidence       CLOSED
 ↓
M2  Failure Attribution                       CLOSED
 ↓
M3  Information Sufficiency + Cost Boundary   ACTIVE (cost boundary calibrated; minimum state open)
 ↘
M4  Architecture Hypothesis Selection         ACTIVE — working contract selected
 ↓
M5  End-to-End Method Implementation          FUTURE
 ↓
M6  Cross-Representation Method Validation    FUTURE
 ↓
M7  Efficiency / Regeneration Trade-off       BASELINE CALIBRATION CLOSED (WL26/27); method comparison FUTURE
 ↓
M8  Generalization Boundary                   FUTURE
 ↓
M9  Paper-Level Evidence Closure              FUTURE
```

M3, M4, and the baseline calibration part of M7 overlapped intentionally.

Attribution is sufficiently closed to select a working representation contract; the exact dynamic-state parameterization remains an M3 question; the physical (worklog 26) and neural (worklog 27) full-recomputation benchmarks now supply the runtime context before implementation.

---

# 4. M0 — Research Contract

## Status

**CLOSED**

## Result

The project is not simply about making neural shading dynamic.

The target is:

> **How should information with different geometry-validity lifecycles be represented so that high-quality neural transport remains reusable under geometry change?**

Reopen only if later evidence invalidates this framing.

---

# 5. M1 — Failure Existence and Direct Visual Evidence

## Status

**CLOSED**

## Closure Evidence

The combined teaset evidence establishes that a frozen high-quality neural transport representation can lose validity under a meaningful geometry-configuration change even when correspondence and the stationary query are controlled.

M1 closes with:

> **Yes — there is a reproducible, physically interpretable geometry-configuration failure worth solving.**

This is an existence result, not a broad generalization claim.

---

# 6. M2 — Failure Attribution

## Status

**CLOSED for the canonical teaset mechanism case**

## Closure Result

The missing information has been narrowed to:

> **current cross-part relational transport state**

specifically:

- nonlocal visibility of indirect incidence;
- radiance arriving from the moved/remote part.

Alternative explanations weakened or removed in the canonical case include:

- correspondence failure;
- canonical/current query mismatch;
- stale direct visibility as the dominant mechanism;
- simple 8DNA representation incapacity.

The RNA/8DNA code audit further showed that the missing state is not supplied dynamically to the learned inter-surface transport component in either released method.

## Scope of Closure

M2 closure applies to the canonical teaset mechanism case.

It does not establish universal failure, broad material/deformation generality, or the final minimum dynamic representation.

---

# 7. M3 — Information Sufficiency and Cost Boundary

## Status

**ACTIVE**

M3 now has two jobs:

1. determine the minimum current transport information needed by the method;
2. establish the cost boundary that makes selective reuse worthwhile.

## H0 — Frozen Static Representation

**CLOSED — insufficient**

The canonical persistent representation remains stale under relation-changing teaset states.

## H1 — Current Local Surface State

**CLOSED as insufficient for the canonical stationary ROI**

On the stable stationary query subset, T0 and T3 provide the same surface identity, local position, normal, view, lighting, and same-part local state, while the correct radiance changes because another part changes the indirect transport relation.

Therefore local current state alone cannot distinguish the two configurations in this controlled case.

This is a logical insufficiency result for the canonical ROI, not a broad claim about every surface or scene.

## H2 — Current Nonlocal / Relational Transport State

**ACTIVE — required category identified, minimum representation not yet selected**

The dynamic state must be capable of representing at least:

- current visibility/occlusion of indirect incidence;
- current radiance arriving from other surfaces.

The working architectural interpretation is:

> **compact current incident-transport context**

rather than a full explicit inverse-rendering decomposition.

Open questions include directional basis vs another compact parameterization, how much nonlocal geometry must be explicit, how to prevent canonical GI leakage into persistent state, and what transport excitation is required during multi-configuration training.

## H3 — Full Current-State Regeneration / Recompute

**CALIBRATED — both baselines measured (worklogs 26, 27)**

Before claiming selective reuse is useful, the cost of obtaining current GI after geometry change had to be known. Both kinds are now measured on the RTX 5080:

- **Physical recomputation (worklog 26, BMW27 / BMW Garage XL, Cycles):** ≈0.22 s / 0.76 s / 3.0 s per 1080p frame at 16 / 64 / 256 spp after a rigid change (persistent data); without persistent data, +0.39 s on BMW27 and tens to hundreds of seconds for large non-instanced scenes.
- **Neural full-state regeneration (worklog 27, teaset T3, historical refit pipelines):** 8DNA 38.9 min to the first rule-satisfying checkpoint (oracle, unstable until 77 min), 1.76 h for the fixed schedule; RNA 3.24 h with no rule-satisfying checkpoint.

Important distinction, unchanged:

> Cycles physical recomputation is not the same as neural full-state regeneration.

The two were measured on different scenes and are not quality matched. Neural regeneration in its existing form is offline (category C) and is therefore not a practical per-change competitor; the tighter runtime reference for a future method remains physical recomputation. The later method evaluation must still compare against both, on the same scene.

## Completion Condition

M3 is complete when we can answer:

1. What minimum current transport state can represent the canonical failure?
2. How is that state prevented from collapsing back into persistent canonical GI?
3. What order-of-magnitude update budget must the method beat to remain meaningful relative to full recomputation/regeneration? — *Answered for the calibration scope:* neural regeneration costs tens of minutes to hours (worklog 27); physical recomputation costs ≈0.2–3 s per frame on BMW27 (worklog 26), which implies an update on the order of ~1–10 ms within a frame to be practically meaningful (worklog 26's estimate, not a validated threshold).

---

# 8. M4 — Architecture Hypothesis Selection

## Status

**ACTIVE — central representation contract selected; bounded prototype implementation is now authorized**

The working architecture is recorded in `docs/Architecture.md`.

Central structure:

```text
Persistent Surface / Material State
            +
Current Configuration-Dependent GI State
            +
Shared Scattering / Transport Operator
            ↓
Current Outgoing Radiance
```

### Current design commitments

- MLP/shared network should act primarily as a reusable rule, not as the storage location for one geometry configuration's GI.
- Configuration-sensitive transport must have a separate current-state path.
- Multi-configuration training is an identifiability mechanism, not the contribution by itself.
- Training configurations should excite transport-relation changes, not merely large geometric displacement.
- Dense all-to-all surface interaction is not acceptable as the default design.
- Current working efficiency direction: sparse near-field relations + compressed far-field transport + incremental/dirty-region update.

### Not yet committed

- graph neural network;
- transformer / sparse attention;
- exact surface-anchor granularity;
- exact latent size;
- exact MLP depth/width;
- spherical harmonics vs learned directional basis;
- exact neighbor-selection rule;
- exact multi-bounce update mechanism;
- final implementation substrate.

## Completion Condition

M4 is closed when one bounded implementation hypothesis is selected with explicit information ownership, runtime update path, anti-leakage training contract, expected computational scaling, and falsifiable success/failure criteria.

---

# 9. M5 — End-to-End Method Implementation

## Status

**FUTURE**

Do not begin the **full end-to-end method** merely because `Architecture.md` exists.

A bounded prototype is now explicitly allowed under M3/M4 after worklogs 26/27 closed the full-recomputation cost calibration. Its purpose is to test the representation contract on the canonical teaset failure, not to declare M5 started.

Promote to M5 only after that prototype narrows the minimum dynamic-state representation, initial surface/anchor granularity, expected update-cost target, and first development substrate.

Preserve original RNA/8DNA baselines as reproducible historical controls.

---

# 10. M6 — Cross-Representation Method Validation

## Status

**FUTURE**

Worklog 22 already supplies cross-backbone **problem replication**.

M6 asks a different question:

> **Does the eventual solution principle transfer beyond the first development backbone?**

Do not confuse repeated failure evidence with portability of the proposed method.

---

# 11. M7 — Efficiency and Regeneration Trade-off

## Status

**BASELINE CALIBRATION CLOSED (worklogs 26, 27); method-level comparison FUTURE**

## M7-A — Physical Full-Recomputation Calibration

**CLOSED (worklog 26).** BMW27 Cycles on the RTX 5080 after a rigid mover change: 38 ms / 85 ms / 219 ms / 763 ms / 2.98 s per 1080p frame at 1 / 4 / 16 / 64 / 256 spp with persistent data (geometry sync ≈2 ms; cost is sampling, ≈11.5 ms/spp); without persistent data +0.39 s per frame, and tens to hundreds of seconds for large non-instanced BMW Garage XL scenes. Denoised low-spp rendering was not measured and is the most important missing physical baseline.

This establishes intuition and an order-of-magnitude target. It does **not** validate the neural architecture.

## M7-A2 — Neural Full-Recomputation Calibration

**CLOSED (worklog 27).** The existing RNA and 8DNA refit paths of worklog 22 (both scratch rebuilds; neither method has a supported warm start), timed end to end on the RTX 5080 from "T3 geometry handed to the pipeline":

| baseline | first checkpoint satisfying the worklog-22 rule | historical selection | fixed schedule | dominant cost |
|---|---|---|---|---|
| 8DNA, envmap | 38.9 min (oracle; stable from 77 min) | recovers | 1.76 h | optimisation 89% |
| 8DNA, common light (same training) | 63.5 min (isolated) | fails | 1.76 h | — |
| RNA, common light | none | fails | 3.24 h | target rendering 87% |

All values are offline (10⁵–10⁶ frames of a 30/60 FPS budget). Full neural recomputation is therefore not a practical response to per-frame or interactive geometry change; it is plausible only as offline per-configuration baking. Untested: faster rebuild variants (shorter schedules, warm start, smaller target sets), run-to-run spread, other assets.

## M7-B — Method-Level Efficiency

After the method exists, compare:

1. physical full recomputation;
2. neural full-state re-encoding/refit/regeneration;
3. proposed incremental dynamic-state update.

Measure update latency, render latency, memory, training cost, amount of state updated, quality, scene-size scaling, and changed-region scaling.

## Kill Condition

If the proposed dynamic-state update costs approximately as much as full regeneration, the reuse-oriented contribution loses practical motivation.

---

# 12. M8 — Generalization Boundary

## Status

**FUTURE**

Expand only after a working method exists.

Relevant axes include another asset family, diffuse/glossy rather than near-mirror-dominated transport, non-rigid bend/twist, fold creation/disappearance, self-contact/contact release, compound deformation, and unseen configuration combinations.

Do not introduce topology change prematurely.

---

# 13. M9 — Paper-Level Evidence Closure

## Status

**FUTURE**

Paper-level closure requires coherent evidence across problem evidence, attribution evidence, method evidence, efficiency evidence, generalization evidence, qualitative review, and quantitative accounting.

The current teaset evidence is the canonical mechanism anchor, not the complete paper claim.

---

# 14. Primary Kill / Reframing Conditions

Reconsider or narrow the direction instead of patching indefinitely if:

- current relational transport cannot be represented compactly enough to beat regeneration;
- the dynamic path effectively requires whole-scene re-encoding every frame;
- the architecture still leaks most canonical GI into persistent state under transport-exciting multi-configuration training;
- dense pairwise interaction is required for acceptable quality;
- the physical full-recompute baseline is already cheap enough in the target regime that selective reuse provides little practical value (worklog 26: 0.2–3 s per 1080p frame at 16–256 spp on BMW27; a converged-quality spp and denoised low-spp frames were not established);
- a fast neural rebuild/refit variant reaches current validity at interactive cost (worklog 27: the existing rebuild pipelines take 39 min – 3.2 h; faster variants untested);
- the eventual method works only on the canonical teaset and does not survive broader material/configuration tests;
- another existing current-geometry method already provides the same ownership/update contract more directly.

Negative results remain valid when scoped to what was actually tested.

---

# 15. Agent Operating Contract

Before substantial research implementation, read:

1. `RESEARCH_CENTRIC_TOPIC.md`
2. this roadmap
3. `BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md`
4. `Architecture.md`

For every meaningful batch identify Direction, Purpose, Central Intent, preserved baseline, variable changed, explicit DO NOT list, and completion question.

Report separately:

- IMPLEMENTATION FACT;
- MEASUREMENT;
- OBSERVATION;
- INTERPRETATION;
- UNRESOLVED QUESTION.

Do not optimize toward a desired conclusion.

---

# 16. Current Immediate Next Steps

## A. Full-recomputation cost calibration — CLOSED

Physical (worklog 26) and neural (worklog 27) full recomputation are both measured; see M7-A / M7-A2. Neither validates the proposed method. Optional follow-ups only if a later decision needs them: denoised low-spp physical frames; a same-scene physical baseline on teaset; a separately protocolled fast neural refit variant.

## B. Minimal dynamic-state prototype — current main task

The cost-viability gate is now closed strongly enough to begin implementation, but only as a **bounded architecture experiment**.

Start from the canonical teaset T0/T3 failure and implement the smallest current directional/relational transport-state path that can represent the already-measured missing transport while persistent appearance/state remains fixed.

The first implementation question is:

> **Can a compact current state carrying indirect visibility and remote-surface incident-radiance information recover the T3 transport change without dense pairwise transport, full neural regeneration, or canonical-GI leakage into persistent state?**

Required controls for the first prototype:

- frozen historical baseline;
- historical full T3 recompute/refit control;
- unchanged persistent-state path;
- explicit accounting of dynamic-state size / changed fraction / update time;
- the existing quantitative recovery rule plus qualitative interaction-ROI review.

Do not begin with graph/attention/SH/latent-dimension sweeps. Choose the minimum representation needed to make the architectural question testable, then stop and judge the abstraction before expanding it.

---

# 17. Current Decision Queue

1. What compact current-state parameterization should represent indirect visibility and remote-surface incident radiance?
2. What surface/anchor granularity is the smallest useful persistent unit?
3. How should near-field sparse relations and far-field compressed transport divide responsibility?
4. How should multi-configuration training excite transport relations strongly enough to make the ownership split identifiable?
5. What anti-leakage constraint prevents persistent state/shared weights from memorizing canonical GI?
6. ~~What physical full-recompute budget does BMW27 / BMW Garage XL establish?~~ Answered by worklog 26 (≈0.2–3 s per 1080p frame at 16–256 spp); neural regeneration answered by worklog 27 (39 min – 3.2 h).
7. What update latency would count as a meaningful practical win? Working estimate from worklog 26: ~1–10 ms per change inside a frame, with quality above 1–4 spp physical frames; neural regeneration (worklog 27) is no tighter a constraint. Not a validated threshold.
8. Which implementation substrate should host the first bounded prototype?
9. After a working prototype exists, how broad is the generalization claim?

Resolve these in evidence order, not implementation convenience order.

---

# 18. Current Milestone Snapshot

| Milestone | Status | Main Question |
|---|---|---|
| M0 — Research Contract | CLOSED | What problem are we solving? |
| M1 — Failure Existence | CLOSED | Is the failure real and physically interpretable? |
| M2 — Failure Attribution | CLOSED | What transport state became stale and why? |
| M3 — Information Sufficiency + Cost Boundary | ACTIVE (cost boundary calibrated, WL26/27) | What current state is minimally sufficient, and what must it beat? |
| M4 — Architecture Selection | ACTIVE | What ownership/update contract should we implement? |
| M5 — End-to-End Method | FUTURE | Does the architecture work as a full system? |
| M6 — Cross-Representation Method Validation | FUTURE | Does the solution transfer beyond one backbone? |
| M7 — Efficiency Trade-off | BASELINES CALIBRATED (WL26/27); method comparison FUTURE | Is selective reuse meaningfully cheaper than recomputation/regeneration? |
| M8 — Generalization Boundary | FUTURE | How broad is the dynamic regime? |
| M9 — Paper Evidence Closure | FUTURE | Is the contribution fully supported? |

---

# 19. One-Line Rule for Future Work

> **The problem, canonical failure mechanism, and full-recompute cost boundary are sufficiently established; the current job is to test the lifecycle split in the smallest bounded dynamic-transport prototype and determine whether its current-state update is both sufficient and materially cheaper than rebuilding current transport from scratch.**