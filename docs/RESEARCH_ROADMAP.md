# Research Roadmap — Dynamic Neural Shading / Neural Light Transport

**Last updated:** 2026-10-01

## Document Role

This document defines the research roadmap, milestone gates, stop conditions, and decision sequence for the project.

It complements:

- `RESEARCH_CENTRIC_TOPIC.md` — what problem the project is fundamentally about;
- `BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md` — how baseline roles, evidence breadth, and attribution scope are separated.

This is a living research document. Worklogs remain append-only historical evidence; this document records the current interpretation and next decision frontier.

Do not treat future milestones as pre-approved implementation tasks.

---

# 0. Central Research Intent

The project investigates whether high-quality neural shading / neural light-transport representations mix information with different validity lifecycles.

## Persistent information

Information that may remain reusable across geometry changes, for example:

- material identity;
- surface identity;
- local appearance priors;
- some microstructure information;
- reusable transport rules.

## Configuration-dependent information

Information whose validity depends on current geometry configuration, for example:

- mutual visibility;
- cross-part proximity;
- cavity/contact structure;
- inter-part interreflection;
- geometry-conditioned multiple scattering.

The core question is:

> **What information should remain persistent, and what information must be conditioned on, updated from, or recomputed from the current geometry configuration?**

Do not reduce this prematurely to MLP vs geometry, graph vs transformer, local vs global, or any one implementation trick.

---

# 1. Current Working Hypothesis

> **High-quality neural transport contains information with different validity lifecycles. Persistent asset information may remain reusable, while transport dependencies whose validity changes with surface configuration require current geometry state.**

A stronger sub-hypothesis is now experimentally motivated:

> **For some geometry-relation changes, current local surface state and direct/current visibility are insufficient; nonlocal configuration-dependent transport state becomes stale.**

This sub-hypothesis is supported in the current teaset regime but is not yet established broadly.

---

# 2. Current Project State

## Established

- RNA/Rain showed a real frozen-deformation failure but retained local/spatial confounds.
- The Rain fixed-target causal probe closed as **PHYSICAL EFFECT TOO WEAK** and did not decide the nonlocal mechanism.
- Worklog 21 produced the first clean controlled 8DNA teaset case:
  - stationary/canonical target query;
  - changed surrounding part relation;
  - strong GT interreflection change above noise;
  - frozen 8DNA near-zero tracking of that change;
  - stable relation-preserving control.
- Worklog 22 reproduced the same frozen-failure signature with RNA on the same teaset state protocol under a shared common-light reference.
- In Worklog 22, frozen 8DNA and frozen RNA fail the locked rule at T1b and T3 and pass the relation-preserving T1 control.
- 8DNA same-state T3 refit under the original envmap regime recovers close to T0 quality, strongly supporting stale persistent state rather than insufficient architecture capacity in that tested regime.

## Current limits

- The cleanest controlled evidence is still one asset family (`teaset`).
- The main clean mechanism is rigid cross-part translation/approach, not yet non-rigid deformation or contact/release.
- RNA's teaset static gate is valid but thin, so its cross-backbone replication remains secondary to the cleaner 8DNA anchor case.
- The exact transport component has not been decomposed beyond evidence that direct/current visibility alone does not explain the shared failure.
- Broad generality across materials, assets, and deformation families is not established.
- No final method architecture or method substrate is approved.

---

# 3. Research Progression

```text
M0  Research Contract                         CLOSED
 ↓
M1  Failure Existence / Direct Evidence       CLOSED
 ↓
M2  Failure Attribution                       ACTIVE
 ↓
M3  Representation Sufficiency Tests          NEXT AFTER M2
 ↓
M4  Architecture Hypothesis Selection         FUTURE
 ↓
M5  End-to-End Method Implementation          FUTURE
 ↓
M6  Cross-Representation Method Validation    FUTURE
 ↓
M7  Efficiency / Regeneration Trade-off       FUTURE
 ↓
M8  Generalization Boundary                   FUTURE
 ↓
M9  Paper-Level Evidence Closure              FUTURE
```

Each transition requires evidence. Implementation convenience is not a gate.

---

# 4. M0 — Research Contract

## Status

**CLOSED**

## Result

The project is not simply about making neural shading dynamic.

The target is:

> **How should information with different geometry-validity lifecycles be represented so that high-quality neural transport remains reusable under geometry change?**

Reopen only if literature or evidence invalidates this framing or a stronger formulation emerges.

---

# 5. M1 — Failure Existence and Direct Visual Evidence

## Status

**CLOSED**

## Closure Evidence

The combined Worklog 21/22 teaset evidence establishes that a high-quality frozen neural transport representation can lose validity under a meaningful geometry-configuration change even when correspondence is controlled.

Evidence includes:

- GT physical transport change above measured reference/model noise;
- exact/analytic correspondence rather than nearest-XYZ lookup;
- stable relation-preserving control;
- large error growth with near-zero tracking gain in relation-changing states;
- replication in both 8DNA and RNA under the common-light regime.

M1 closes with:

> **Yes — there is a reproducible, physically interpretable geometry-configuration failure worth investigating further.**

This is an existence result, not a broad generalization claim.

---

# 6. M2 — Failure Attribution

## Status

**ACTIVE — stale-state attribution is strong in the controlled teaset case; exact transport component and breadth remain open**

## Central Question

> **What configuration-dependent information became stale, and what alternative explanations remain?**

## Current evidence

The teaset controls weaken or remove several trivial explanations:

- coordinate/query mismatch is separately diagnosed and is not the primary stationary-surface failure;
- relation-preserving whole-asset motion is stable;
- stationary interaction queries remain canonical/current-identical;
- GT transport changes strongly only when the part relation changes;
- RNA recomputes current visibility, so stale direct visibility alone cannot explain the shared failure;
- 8DNA same-state T3 refit in the original envmap regime recovers close to T0 quality, showing that the architecture can represent T3 when state is rebuilt.

The common-light refits for 8DNA and RNA both land near the predeclared recovery threshold. Do not interpret their small split as a capacity difference without further evidence.

## Open attribution questions

- Which physical transport component dominates the missing change: inter-part interreflection, near-contact occlusion, higher-order transport, or a combination?
- Is a reference-only transport decomposition needed before M2 closure?
- Is current local geometry sufficient anywhere, or is explicit nonlocal relation/state required?

## Completion Condition

M2 is closed when the mechanism is narrow enough to define the information that the first M3 sufficiency experiment must add, without guessing a full architecture.

---

# 7. M3 — Representation Sufficiency Tests

## Status

**NEXT AFTER M2 CLOSURE — evidence-motivated, not yet approved for implementation**

This milestone tests information sufficiency before selecting a full method.

## H0 — Frozen Static Representation

Baseline already demonstrated:

> Does the canonical persistent representation remain valid after geometry relation change?

Answer in the controlled teaset case: **no**.

## H1 — Current Local Geometry Conditioning

Conceptually:

```text
persistent state + current local surface state + lighting/view
```

Possible local information:

- normal/tangent frame;
- curvature/differential shape;
- bounded local neighborhood descriptors.

Strict boundary: H1 must not silently include cross-part distance, remote visibility, cavity width, or other nonlocal relation data.

Central question:

> **Is current local surface state sufficient to repair the failure?**

If yes, do not continue claiming explicit nonlocal relation modeling is necessary.

## H2 — Current Nonlocal / Relational Geometry Conditioning

Conceptually:

```text
persistent state + current local state + current cross-surface relation
```

Possible relation information may include:

- relative position/orientation;
- proximity;
- mutual visibility/contact state;
- sparse transport context.

No graph, transformer, message-passing, token, or cache design is approved yet.

Central question:

> **Does providing current cross-surface relation recover failure that local state cannot explain?**

## H3 — Full Current-State Regeneration

Conceptually:

```text
current geometry -> rebuild transport state -> render
```

Central question:

> **Is selective persistent reuse actually useful compared with rebuilding the current representation?**

## Completion Condition

M3 is closed when we can answer:

1. Is local current geometry sufficient?
2. If not, what nonlocal/current relation is necessary?
3. Is selective reuse worthwhile relative to full regeneration?

---

# 8. M4 — Architecture Hypothesis Selection

## Status

**FUTURE**

Only begin after M1–M3 produce discriminating evidence.

A possible abstract contract is:

```text
Neural Asset
= Persistent State
+ Configuration-Dependent State
+ Transport Operator
```

This is conceptual only. It does not prescribe the feature type, graph structure, attention mechanism, decoder, coordinate system, or cache.

Select one bounded architecture hypothesis, not several full architectures at once.

---

# 9. M5 — End-to-End Method Implementation

## Status

**FUTURE**

RNA remains the current likely first development substrate because its geometry/network interface is inspectable, but the final method base is still undecided.

Preserve the original upstream baseline as a reproducible path.

The paper-level method should be expressed as an implementation-independent representation contract unless evidence shows that the contribution is genuinely RNA-specific.

---

# 10. M6 — Cross-Representation Method Validation

## Status

**FUTURE**

Important distinction:

> Worklog 22 already provides cross-backbone **problem replication**. M6 is future cross-backbone **method validation**.

Once a proposed solution principle exists, test whether it transfers beyond the first development backbone.

Do not confuse a repeated failure with successful portability of the eventual solution.

---

# 11. M7 — Efficiency and Reuse Trade-off

## Status

**FUTURE**

Central question:

> **Does preserving reusable state provide a meaningful advantage over regenerating transport from current geometry?**

Compare as relevant:

- update latency;
- full re-encoding cost;
- memory;
- training/update cost;
- render cost;
- quality.

If selective reuse costs approximately as much as full regeneration, the reuse-oriented architecture loses practical motivation.

---

# 12. M8 — Generalization Boundary

## Status

**FUTURE**

Expand only after a working method exists.

Relevant future axes include:

- another asset family;
- diffuse/glossy rather than near-mirror-dominated interaction;
- non-rigid bend/twist;
- fold creation/disappearance;
- self-contact / contact release;
- compound deformation;
- unseen deformation combinations.

Do not introduce topology change prematurely.

---

# 13. M9 — Paper-Level Evidence Closure

## Status

**FUTURE**

Paper-level closure requires a coherent evidence package across:

- problem evidence;
- attribution evidence;
- method evidence;
- efficiency evidence;
- generalization evidence;
- qualitative review;
- quantitative accounting.

One clean teaset case is a strong anchor, not a complete paper claim.

---

# 14. Primary Kill / Reframing Conditions

Reconsider the direction instead of patching indefinitely if:

- comparable high-quality systems remain robust under the target changes;
- the important failure reduces to coordinate/indexing or other straightforward engineering errors;
- current local geometry alone resolves the failure, in which case the nonlocal framing should narrow;
- full regeneration is cheap enough that persistent/dynamic separation offers little value;
- the proposed relational state effectively requires full scene re-encoding;
- the phenomenon proves backbone-specific under comparable tests.

Negative results are valid when scoped to what was actually tested.

---

# 15. Agent Operating Contract

Before substantial research implementation, read:

1. `RESEARCH_CENTRIC_TOPIC.md`
2. this roadmap
3. `BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md`

For every meaningful batch identify:

- Direction;
- Purpose;
- Central Intent;
- preserved baseline;
- variable changed;
- explicit DO NOT list;
- completion question.

Report separately:

- IMPLEMENTATION FACT;
- MEASUREMENT;
- OBSERVATION;
- INTERPRETATION;
- UNRESOLVED QUESTION.

Do not optimize toward a desired conclusion.

---

# 16. Current Immediate Next Step

The immediate next step is **not full architecture implementation**.

The project has moved beyond failure-existence hunting. The current frontier is to close M2 strongly enough to justify the smallest discriminating M3 experiment.

Priority order:

1. **Qualitatively review Worklog 22 exports**, especially whether RNA shows the same stale interaction pattern rather than merely poor static reconstruction.
2. **Decide whether reference-only transport decomposition is needed** to identify what physical component the frozen models fail to track.
3. **Do not spend a large batch chasing the common-light refit threshold split** unless that distinction changes the next architecture decision.
4. Choose one next axis:
   - **breadth** if generality is currently the limiting question; or
   - **H1/H2 sufficiency** if attribution is already strong enough to specify the missing information.

---

# 17. Current Decision Queue

1. Is one reference-only transport decomposition still needed before closing M2?
2. Should the next bounded batch prioritize breadth or H1/H2 sufficiency?
3. Is RNA retained as the first method-development substrate?
4. What exact information must be mutable at runtime?
5. Is explicit nonlocal relation modeling necessary?
6. What is the minimal sufficient current-state representation?
7. Does selective reuse beat full regeneration?
8. When is a current-geometry-conditioned contrast baseline necessary?
9. How broad should the eventual deformation/generalization claim become?

Resolve these in evidence order, not implementation order.

---

# 18. Current Milestone Snapshot

| Milestone | Status | Main Question |
|---|---|---|
| M0 — Research Contract | CLOSED | What problem are we solving? |
| M1 — Failure Existence | CLOSED | Is the failure real and physically interpretable? |
| M2 — Failure Attribution | ACTIVE | What configuration-dependent transport state became stale? |
| M3 — Sufficiency Tests | NEXT AFTER M2 | Local state vs nonlocal relation vs full regeneration? |
| M4 — Architecture Selection | FUTURE | What representation contract should we implement? |
| M5 — End-to-End Method | FUTURE | Does the architecture work as a full system? |
| M6 — Cross-Representation Method Validation | FUTURE | Does the proposed solution transfer beyond one backbone? |
| M7 — Efficiency Trade-off | FUTURE | Is reuse actually advantageous? |
| M8 — Generalization Boundary | FUTURE | How broad is the dynamic regime? |
| M9 — Paper Evidence Closure | FUTURE | Is the contribution fully supported? |

---

# 19. One-Line Rule for Future Work

> **The failure now has a clean cross-backbone controlled case; the next step is to identify the minimal missing current-state information before committing to a solution architecture.**
