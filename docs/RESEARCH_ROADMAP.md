# Research Roadmap — Dynamic Neural Shading / Neural Light Transport

## Document Role

This document defines the **research roadmap, milestone gates, stop conditions, and decision sequence** for the project.

It complements:

- `RESEARCH_CENTRIC_TOPIC.md`

The two documents serve different purposes:

- `RESEARCH_CENTRIC_TOPIC.md` defines **what problem this project is fundamentally about**.
- This roadmap defines **how the project should progress without losing that intent**.

This is a living research document.

It should be updated when:

- a milestone is closed,
- a working hypothesis is falsified,
- a new architecture direction is approved,
- a baseline model is selected,
- a major experimental result changes the project direction.

Do not treat future milestones as pre-approved implementation tasks.

---

# 0. Central Research Intent

The project investigates whether high-quality neural shading / neural light-transport representations mix together information with different validity lifecycles.

The main distinction is between:

## Persistent information

Information that may remain reusable across geometry changes.

Examples may include:

- material identity,
- surface identity,
- local appearance priors,
- some microstructure information,
- reusable transport rules.

## Configuration-dependent information

Information whose validity depends on the current geometry configuration.

Examples may include:

- mutual visibility,
- cross-part proximity,
- cavity structure,
- self-shadowing,
- self-contact,
- interreflection,
- geometry-conditioned multiple scattering.

The core architectural question is:

> **What information should remain persistent, and what information must be conditioned on, updated from, or recomputed from the current geometry configuration?**

The project must not reduce this question prematurely to:

- MLP vs geometry,
- local vs global features,
- graph vs transformer,
- canonical-space vs world-space,
- one specific implementation trick.

Those are possible implementation choices, not the central research question.

---

# 1. Current Working Hypothesis

The current working hypothesis is:

> **High-quality neural transport contains information with different validity lifecycles. Persistent asset information should remain reusable, while transport dependencies whose validity changes with surface configuration must be derived from or conditioned on the current geometry state.**

A stronger sub-hypothesis, not yet established, is:

> **For some deformation regimes, current local geometry alone is insufficient, and nonlocal cross-surface relations are required to recover correct transport.**

This sub-hypothesis is especially relevant to:

- cross-part approach,
- self-contact,
- contact release,
- compound deformation.

Do not treat either statement as proven.

---

# 2. Current Project State

## Status Summary

### Completed / established

- Central research territory has been defined.
- Static high-quality neural transport under dynamic geometry is the target research setting.
- The project explicitly distinguishes:
  - trivial coordinate/indexing failure,
  - local geometry update failure,
  - true geometry-conditioned transport invalidation.
- RNA has been used in initial experiments.
- A first dynamic deformation experiment exists.
- Batch 2 was designed to move away from synthetic-only evidence toward:
  - fold creation/disappearance,
  - cross-part approach/self-contact,
  - compound deformation.
- A representation-lifecycle working hypothesis has been articulated.

### Pending

- Human qualitative review of Batch 2 high-quality deformation results.
- Determination of whether the observed failure is strong enough to count as meaningful research evidence.
- Determination of whether the observed failure is:
  - RNA-specific,
  - implementation-specific,
  - local-geometry explainable,
  - or genuinely nonlocal/configuration-dependent.
- Final selection of the main implementation backbone.

### Not yet approved

- A final method architecture.
- Persistent/dynamic factorization implementation.
- Graph/message-passing architecture.
- Geometry-conditioned decoder.
- Online adaptation.
- 8DNA integration as a production research backbone.
- Cross-model generalization claims.
- Inverse-rendering extension.

---

# 3. Research Progression Overview

The intended progression is:

```text
M0  Research Contract
        ↓
M1  Failure Existence / Visual Evidence
        ↓
M2  Failure Attribution
        ↓
M3  Representation Sufficiency Tests
        ↓
M4  Architecture Hypothesis Selection
        ↓
M5  End-to-End Method Implementation
        ↓
M6  Cross-Representation Validation
        ↓
M7  Efficiency / Regeneration Trade-off
        ↓
M8  Generalization Boundary
        ↓
M9  Paper-Level Evidence Closure
```

A milestone should not be entered merely because implementation is possible.

Each transition requires evidence.

---

# 4. M0 — Research Contract

## Status

**CLOSED**

## Goal

Establish what the project is actually trying to solve.

## Result

The project is not simply about making neural shading dynamic.

The target is the representation-level question:

> **How should information with different geometry-validity lifecycles be represented so that high-quality neural transport remains reusable under geometry change?**

## Preserve

- `RESEARCH_CENTRIC_TOPIC.md`
- the distinction between persistent and configuration-dependent information
- the requirement that evidence precede architecture commitment

## Do Not Reopen Unless

- literature falsifies the problem framing,
- experiments show that the hypothesized failure is negligible,
- a clearly stronger research formulation emerges.

---

# 5. M1 — Failure Existence and Direct Visual Evidence

## Status

**ACTIVE — qualitative review pending**

## Central Question

> **Does a high-quality neural transport representation trained on a static geometry configuration lose validity under meaningful non-rigid deformation when correspondence and current geometry inputs are handled correctly?**

## Primary Evidence

Batch 2 deformation families:

1. Fold creation / disappearance
2. Cross-part approach / self-contact
3. Compound deformation

## Required Evidence

### Quantitative

- global image metrics,
- ROI-localized errors,
- deformation descriptors,
- error concentration in geometry-changed transport regions.

### Qualitative

Directly inspect:

- newly created fold/cavity regions,
- disappearing cavity regions,
- newly formed self-shadow,
- removed shadow,
- near-contact transport,
- self-contact,
- released-contact regions,
- interreflection changes.

## Key Rule

A heatmap alone is not sufficient.

The desired result is a failure that can be explained physically.

Example:

> A remote surface approaches the target patch.  
> The local patch remains nearly unchanged.  
> Ground-truth transport changes strongly.  
> The frozen neural representation remains biased toward the canonical configuration.

## Kill / Weakening Conditions

The current research direction weakens substantially if:

- high-quality frozen neural transport remains accurate under strong deformation,
- visible failures disappear after correspondence bugs are fixed,
- observed degradation is dominated by incorrect normals/tangents,
- results exist only in toy scenes and not in high-quality assets.

## Completion Condition

M1 is closed when we can answer:

> **Is there a reproducible, high-quality, physically interpretable deformation failure worth investigating further?**

Possible answers:

- Yes
- No
- Not yet attributable

All are valid outcomes.

---

# 6. M2 — Failure Attribution

## Status

**BLOCKED BY M1**

## Central Question

> **What actually causes the observed deformation failure?**

Do not assume the answer is nonlocal transport.

## Attribution Categories

### A. Coordinate / indexing failure

Examples:

- wrong canonical lookup,
- broken feature correspondence,
- stale surface identity.

### B. Local geometry failure

Examples:

- incorrect current normal,
- tangent mismatch,
- curvature mismatch,
- local deformation not represented.

### C. Nonlocal configuration failure

Examples:

- remote surface approach,
- new visibility dependency,
- self-contact,
- contact release,
- inter-part interreflection.

### D. Model capacity failure

The architecture may simply be unable to represent the target state even when retrained.

### E. Training-distribution failure

The representation may require deformation examples during training.

## Required Control

At least one deformed-state refit should be considered when needed.

If:

```text
Frozen(G0 → G1) fails
Re-fit(G1) succeeds
```

then stale state becomes a more plausible explanation.

If:

```text
Re-fit(G1) also fails
```

then architecture capacity or training may be the primary issue.

## Completion Condition

M2 is closed when the main failure can be attributed well enough to design a discriminating representation experiment.

---

# 7. M3 — Representation Sufficiency Tests

## Status

**FUTURE**

This milestone tests the architecture hypothesis without yet committing to a final method.

The intended sequence is deliberately hierarchical.

---

## H0 — Frozen Static Representation

### Question

Does the canonical representation remain valid after deformation?

This is the M1/M2 baseline.

---

## H1 — Current Local Geometry Conditioning

Conceptually:

\[
L_o =
F(
z^P,
g_{\text{local}}(G_t),
L,
V
)
\]

Possible local information:

- current normal,
- tangent frame,
- curvature,
- local differential shape,
- bounded local neighborhood descriptors.

### Strict Boundary

A local descriptor must not silently include nonlocal information.

Do not classify as "local":

- remote-surface distance,
- large-radius neighborhood context,
- visibility to unrelated surface regions,
- cavity width requiring another surface,
- cross-part proximity.

### Central Question

> **Is current local surface state sufficient to repair the failure?**

### Kill Condition for Nonlocal Hypothesis

If local conditioning explains nearly all relevant failure:

> Do not continue claiming that explicit nonlocal relation modeling is necessary.

---

## H2 — Current Nonlocal / Relational Geometry Conditioning

Conceptually:

\[
L_o =
F(
z^P,
g_{\text{local}}(G_t),
r_{\text{nonlocal}}(G_t),
L,
V
)
\]

Possible relational information may include:

- relative position,
- mutual orientation,
- proximity,
- current visibility,
- contact state,
- cross-part relation,
- sparse transport context.

No specific representation is approved yet.

Possible future realizations include:

- graphs,
- message passing,
- sparse attention,
- relation tokens,
- spatial hashes,
- learned transport probes,
- hierarchical relation features.

These are implementation options, not the hypothesis itself.

### Central Question

> **Does explicitly providing current cross-surface configuration recover failure that cannot be explained by local geometry alone?**

### Strong Evidence Pattern

A particularly strong experiment is:

\[
g_i^{local}(G_a)
\approx
g_i^{local}(G_b)
\]

while:

\[
T_i(G_a)
\neq
T_i(G_b)
\]

because a remote surface changed configuration.

If local conditioning fails and relational conditioning succeeds, this directly supports the nonlocal-dependency hypothesis.

---

## H3 — Full Current-State Regeneration

Conceptually:

\[
G_t
\rightarrow
R_t
\rightarrow
T_t
\]

This represents the opposite end of the design spectrum:

- re-encode the scene,
- regenerate current transport state,
- recompute large portions of the representation.

### Central Question

> **Is selective persistent reuse actually useful compared with simply regenerating the current-state representation?**

This is essential.

A more elegant decomposition is not useful if it provides no meaningful cost or latency advantage.

---

## Completion Condition

M3 is closed when we can answer:

1. Is local current geometry sufficient?
2. If not, does nonlocal current relation provide necessary information?
3. Is selective reuse viable compared with full state regeneration?

---

# 8. M4 — Architecture Hypothesis Selection

## Status

**FUTURE**

Only begin this milestone after M1–M3 provide evidence.

## Candidate Abstract Contract

A possible high-level form is:

\[
\text{Neural Asset}
=
\text{Persistent State}
+
\text{Configuration-Dependent State}
+
\text{Transport Operator}
\]

where:

\[
z_i^P
\]

represents persistent information and:

\[
c_i(G_t)
\]

represents mutable configuration-dependent context.

The renderer may then evaluate:

\[
L_o =
F_\theta(
z_i^P,
c_i(G_t),
L,
\omega_o
)
\]

## Important

This is a conceptual contract.

It does not yet prescribe:

- feature type,
- graph structure,
- attention,
- cache,
- MLP layout,
- coordinate system,
- exact decomposition.

## Architecture Selection Criteria

The candidate architecture must be judged on:

- physical interpretability,
- information lifecycle correctness,
- reuse efficiency,
- generalization across deformation,
- implementation complexity,
- runtime cost,
- training stability,
- debugging observability,
- cross-backbone portability.

## Completion Condition

Select one bounded architecture hypothesis that can be implemented as a controlled experiment.

Do not select multiple competing full architectures in the same implementation batch.

---

# 9. M5 — End-to-End Method Implementation

## Status

**FUTURE**

## Implementation Strategy

Use an established high-quality neural transport system as the execution substrate.

Current likely candidate:

- RNA

But the final decision should be made after M1/M2 evidence review.

## Required Baseline Preservation

The original upstream model must remain reproducible.

Conceptually maintain:

```text
upstream baseline
        +
research modules
        =
new end-to-end framework
```

The project may modify:

- feature representation,
- decoder,
- geometry interface,
- training path,
- renderer integration,
- transport state representation.

But the baseline path must remain available.

## Scientific Abstraction Rule

The paper-level architecture must not be defined as:

> "RNA + module X"

unless the contribution truly is RNA-specific.

Prefer an implementation-independent representation contract.

---

# 10. M6 — Cross-Representation Validation

## Status

**FUTURE**

## Candidate Secondary Backbone

8DNA is currently a strong candidate.

## Purpose

Determine whether the discovered principle is:

- RNA-specific,
- regression-specific,
- or more general to neural asset transport representations.

## Strong Outcome

If the same ownership/lifecycle principle improves both:

- RNA-like asset transport,
- 8DNA-like higher-dimensional transport,

then the contribution becomes substantially stronger.

## Important

Do not integrate 8DNA merely because it is newer.

Only use it when a concrete architectural principle exists to transfer.

---

# 11. M7 — Efficiency and Reuse Trade-off

## Status

**FUTURE**

## Central Question

> **Does preserving reusable state provide a meaningful advantage over regenerating transport from the current geometry?**

Required comparison may include:

- update latency,
- full re-encoding cost,
- memory,
- training/update cost,
- rendering cost,
- quality,
- dynamic responsiveness.

## Kill Condition

If:

\[
C_{\text{selective reuse}}
\approx
C_{\text{full regeneration}}
\]

and quality is similar, the reuse-oriented architecture may have weak practical motivation.

Likewise, if relational state effectively requires re-encoding the entire scene every frame, the proposed decomposition may be architecturally hollow.

---

# 12. M8 — Generalization Boundary

## Status

**FUTURE**

Only after a working method exists.

Expand from controlled deformation toward:

- larger non-rigid deformation,
- articulation,
- fold creation/disappearance,
- self-contact,
- contact release,
- compound deformation,
- unseen deformation combinations.

Later, if justified:

- cloth,
- hair/fibers,
- soft body,
- topology change,
- fracture/cutting.

Do not introduce topology change prematurely.

## Central Question

> **How broad is the validity region of the proposed representation?**

---

# 13. M9 — Paper-Level Evidence Closure

## Status

**FUTURE**

The project reaches paper-level closure only when the evidence supports a coherent claim.

Expected evidence classes include:

### Problem Evidence

- high-quality failure exists,
- failure is reproducible,
- failure is not an implementation artifact.

### Attribution Evidence

- local vs nonlocal dependency is distinguished,
- relevant transport regions are identified.

### Method Evidence

- the proposed representation addresses the identified failure,
- ablation supports the information-lifecycle hypothesis.

### Efficiency Evidence

- persistent reuse provides measurable value.

### Generalization Evidence

- method works beyond one deformation,
- ideally beyond one neural transport backbone.

### Qualitative Evidence

- failure and recovery are visually interpretable.

### Quantitative Evidence

- image metrics,
- ROI metrics,
- transport-region accounting,
- runtime/update cost,
- memory where relevant.

---

# 14. Primary Kill Conditions

The following outcomes should cause the project direction to be reconsidered rather than patched indefinitely.

## Kill A — No meaningful failure

High-quality neural transport remains robust under the target deformations.

## Kill B — Failure is trivial

The observed problem is explained by:

- coordinate mapping,
- stale normals,
- implementation bugs,
- other straightforward engineering issues.

## Kill C — Local geometry is sufficient

If local-current-geometry conditioning resolves the important failures, do not continue claiming that nonlocal relational state is necessary.

A narrower local-deformation problem may still remain valuable, but the research framing must change.

## Kill D — Full regeneration is effectively free

If rebuilding current transport state is cheap enough, persistent/dynamic separation may offer little practical benefit.

## Kill E — Proposed relational representation is equivalent to full scene re-encoding

If the "selective" dynamic state ends up requiring almost complete global recomputation, the architecture motivation weakens.

## Kill F — Failure is backbone-specific

If the phenomenon exists only in one peculiar implementation and disappears in comparable neural transport systems, broader claims must be reduced.

---

# 15. Branch / Session Responsibilities

To avoid mixing implementation work with scientific interpretation, maintain two conceptual tracks.

## Track A — Implementation / Experimental Platform

Responsibilities:

- model setup,
- renderer integration,
- deformation pipeline,
- training,
- inference,
- metrics,
- review exports,
- reproducibility,
- performance profiling.

Primary question:

> **Did the experiment run correctly, and what evidence did it produce?**

## Track B — Research Evidence / Contribution Validation

Responsibilities:

- qualitative review,
- failure attribution,
- literature boundary,
- novelty risk,
- hypothesis strength,
- kill decisions,
- architecture interpretation.

Primary question:

> **What does the evidence actually mean for the research direction?**

Do not allow implementation convenience to decide the scientific interpretation.

---

# 16. Agent Operating Contract

Any Agent working in this repository should read:

1. `RESEARCH_CENTRIC_TOPIC.md`
2. this roadmap

before performing substantial research implementation.

The Agent must distinguish:

- current milestone,
- approved scope,
- future hypotheses,
- forbidden premature work.

## Before implementation

The Agent should identify:

- current milestone,
- central question,
- preserved baseline,
- experimental variable,
- completion condition.

## During implementation

Do not:

- broaden the architecture automatically,
- tune until the hypothesis appears true,
- convert a diagnostic into a method without approval,
- proceed into future milestones just because implementation is convenient.

## After implementation

Report separately:

- IMPLEMENTATION FACT
- MEASUREMENT
- OBSERVATION
- INTERPRETATION
- UNRESOLVED QUESTION

Do not declare architecture success automatically.

---

# 17. Current Immediate Next Step

The immediate next step is **not architecture implementation**.

It is:

> **Human qualitative review of Batch 2 deformation results.**

The review should determine:

1. Is the failure clearly visible?
2. Is it physically interpretable?
3. Which deformation family is strongest?
4. Is the failure localized to geometry-dependent transport regions?
5. Does the evidence justify moving into M2 attribution?
6. Does the result look strong enough to continue investigating as a contribution-level failure case?

Only after this review should the implementation backbone and first architecture experiment be finalized.

---

# 18. Current Decision Queue

The following decisions remain intentionally open:

1. Is Batch 2 evidence strong enough?
2. Which failure family becomes the canonical demonstration case?
3. Is RNA retained as the main implementation substrate?
4. When should 8DNA be integrated?
5. Is local geometry conditioning sufficient?
6. Is explicit nonlocal relation modeling necessary?
7. What is the minimal sufficient current-state representation?
8. Does selective reuse beat full regeneration?
9. How broad should the deformation generalization claim become?

Do not collapse these into one decision.

Resolve them in order as evidence becomes available.

---

# 19. Project Success Criterion

The project is successful only if it produces a defensible answer to the representation-level question:

> **How should high-quality neural light transport separate reusable information from geometry-configuration-dependent information so that the representation remains useful under meaningful dynamic geometry changes?**

A successful outcome does not require the original hypothesis to survive unchanged.

A strong negative result that clearly identifies the true boundary is preferable to a weak positive result produced by tuning.

---

# 20. Current Milestone Snapshot

| Milestone | Status | Main Question |
|---|---|---|
| M0 — Research Contract | CLOSED | What problem are we solving? |
| M1 — Failure Existence | ACTIVE | Is the deformation failure real and meaningful? |
| M2 — Failure Attribution | BLOCKED | What causes it? |
| M3 — Sufficiency Tests | FUTURE | Local geometry vs nonlocal relation vs full regeneration? |
| M4 — Architecture Selection | FUTURE | What representation contract should we implement? |
| M5 — End-to-End Method | FUTURE | Does the architecture work as a full system? |
| M6 — Cross-Representation Validation | FUTURE | Does the principle transfer beyond one backbone? |
| M7 — Efficiency Trade-off | FUTURE | Is reuse actually advantageous? |
| M8 — Generalization Boundary | FUTURE | How broad is the dynamic regime? |
| M9 — Paper Evidence Closure | FUTURE | Is the contribution fully supported? |

---

# 21. One-Line Rule for Future Work

> **Do not implement the next architecture because it is plausible; implement it only when the current milestone produces evidence that makes the next hypothesis worth testing.**
