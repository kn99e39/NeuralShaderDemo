# Research Roadmap — Dynamic Neural Shading / Neural Light Transport

**Last updated:** 2026-10-06

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
- 8DNA same-state T3 refit under the original envmap regime recovers close to T0 quality, strongly supporting stale persistent state rather than insufficient representation capacity in that tested regime.

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
- The update-cost advantage over full recomputation/regeneration has not yet been measured.
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
M3  Information Sufficiency + Cost Boundary   ACTIVE
 ↘
M4  Architecture Hypothesis Selection         ACTIVE — working contract selected
 ↓
M5  End-to-End Method Implementation          FUTURE
 ↓
M6  Cross-Representation Method Validation    FUTURE
 ↓
M7  Efficiency / Regeneration Trade-off       EARLY CALIBRATION ACTIVE
 ↓
M8  Generalization Boundary                   FUTURE
 ↓
M9  Paper-Level Evidence Closure              FUTURE
```

M3, M4, and the early calibration part of M7 may overlap.

This overlap is intentional: attribution is sufficiently closed to select a working representation contract; the exact dynamic-state parameterization remains an M3 question; and the physical full-recomputation benchmark informs the runtime budget before implementation.

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

**ACTIVE as a calibration baseline**

Before claiming selective reuse is useful, measure the cost of obtaining physically current GI after geometry change.

The immediate calibration uses:

- official BMW27 Cycles scene as the reproducible baseline;
- BMW Garage XL only if the original scene is too light to expose useful scaling on RTX 5080;
- SPP / scene-scale / geometry-change cost curves.

Important distinction:

> Cycles physical recomputation is not the same as neural full-state regeneration.

The later method evaluation must compare both where relevant.

## Completion Condition

M3 is complete when we can answer:

1. What minimum current transport state can represent the canonical failure?
2. How is that state prevented from collapsing back into persistent canonical GI?
3. What order-of-magnitude update budget must the method beat to remain meaningful relative to full recomputation/regeneration?

---

# 8. M4 — Architecture Hypothesis Selection

## Status

**ACTIVE — central representation contract selected; implementation choices remain open**

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

Do not begin the full method merely because `Architecture.md` exists.

Begin only after M3/M4 narrow the minimum dynamic-state representation, initial surface/anchor granularity, expected update-cost target, and first development substrate.

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

**EARLY CALIBRATION ACTIVE; method-level comparison FUTURE**

## M7-A — Physical Full-Recomputation Calibration

Current task:

- measure geometry-change → current Cycles GI cost on RTX 5080;
- use BMW27 as the reproducible anchor;
- expand to BMW Garage XL only when needed for meaningful large-scene scaling;
- measure effective FPS and distance from 30/60 FPS budgets.

This establishes intuition and an order-of-magnitude target. It does **not** validate the neural architecture.

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
- the physical full-recompute baseline is already cheap enough in the target regime that selective reuse provides little practical value;
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

Two bounded activities are now justified in parallel.

## A. Full-recomputation cost calibration

Run the BMW27 / BMW Garage XL Cycles benchmark batch.

Purpose:

> establish the physical recomputation cost curve and a realistic update-latency target for the proposed method.

Do not treat this as neural-method validation.

## B. Minimal dynamic-state design

Without implementing the full architecture yet, determine the smallest current directional/relational transport representation that can express the already-measured teaset failure.

The first design question is:

> **How should current indirect visibility and remote-surface incident radiance be compressed without restoring dense pairwise transport or allowing canonical GI to leak back into persistent state?**

Do not prematurely choose graph/attention/SH/latent dimensions before this information question is closed.

---

# 17. Current Decision Queue

1. What compact current-state parameterization should represent indirect visibility and remote-surface incident radiance?
2. What surface/anchor granularity is the smallest useful persistent unit?
3. How should near-field sparse relations and far-field compressed transport divide responsibility?
4. How should multi-configuration training excite transport relations strongly enough to make the ownership split identifiable?
5. What anti-leakage constraint prevents persistent state/shared weights from memorizing canonical GI?
6. What physical full-recompute budget does BMW27 / BMW Garage XL establish?
7. What update latency would count as a meaningful practical win?
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
| M3 — Information Sufficiency + Cost Boundary | ACTIVE | What current state is minimally sufficient, and what must it beat? |
| M4 — Architecture Selection | ACTIVE | What ownership/update contract should we implement? |
| M5 — End-to-End Method | FUTURE | Does the architecture work as a full system? |
| M6 — Cross-Representation Method Validation | FUTURE | Does the solution transfer beyond one backbone? |
| M7 — Efficiency Trade-off | EARLY CALIBRATION ACTIVE | Is selective reuse meaningfully cheaper than recomputation/regeneration? |
| M8 — Generalization Boundary | FUTURE | How broad is the dynamic regime? |
| M9 — Paper Evidence Closure | FUTURE | Is the contribution fully supported? |

---

# 19. One-Line Rule for Future Work

> **The problem and canonical failure mechanism are now sufficiently established; the current job is to turn the lifecycle split into the smallest useful dynamic transport state and prove that updating it is materially cheaper than rebuilding current transport from scratch.**