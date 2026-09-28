# Baseline Roles and Evidence Strategy — Dynamic Neural Shading / Neural Light Transport

**Decision date:** 2026-09-28  
**Status:** Active research operating decision

## Document Role

This document records the current decision on:

- how RNA, 8DNA, and geometry-conditioned rendering systems should be used;
- how inductive problem discovery and deductive mechanism testing must be separated;
- what level of evidence is required before the project-level hypothesis may be weakened or falsified;
- why a negative result on Rain/RNA alone must not be generalized to the whole research direction.

This document complements:

- `RESEARCH_CENTRIC_TOPIC.md`
- `RESEARCH_ROADMAP.md`

It should be treated as the authoritative reference when deciding whether a model is being used as:

- an implementation/development base,
- a scientific replication baseline,
- an architecture-contrast baseline,
- or a future method substrate.

---

# 1. Current Research State

The project has so far studied one configuration in substantial depth:

```text
RNA × Rain × scarf deformation
```

This work has been useful for developing:

- canonical correspondence controls;
- current-geometry input auditing;
- direct-visibility diagnostics;
- production-asset deformation experiments;
- failure attribution methodology;
- fixed-target causal-probe design.

However, this is still one model family and one main asset/deformation environment.

Therefore:

> **Rain/RNA is currently a mechanism-development bench, not sufficient evidence for a project-level generalization or falsification.**

A negative result in Rain/RNA may falsify a narrow Rain/RNA mechanism hypothesis.

It does **not** by itself falsify the broader research question:

> **Do high-quality learned shading / neural light-transport representations lose validity when geometry relationships change, and does this expose a representation-lifecycle problem between persistent information and current geometry-dependent transport state?**

---

# 2. Research Process: Inductive Breadth Before Deductive Closure

The intended scientific process has two distinct stages.

## 2.1 Inductive Phenomenon Discovery

First ask:

> **Across different model and deformation families, what failure patterns recur?**

The purpose is breadth, not perfect causal attribution.

Representative failure categories include:

- local appearance collapse;
- directional-input OOD;
- spatial feature contamination / ownership ambiguity;
- stale cavity illumination;
- missing or stale contact shadow;
- stale interreflection or color bleeding;
- self-contact / contact-release failure;
- disocclusion or coverage failure.

At this stage, one model-specific failure is informative but not sufficient for a broad claim.

## 2.2 Deductive Mechanism Testing

After a recurring transport-like phenomenon is identified, construct strong controls that isolate the hypothesized mechanism.

Example strong control:

```text
same target surface identity
same target position
same target normal
same view direction
same light direction
same direct-visibility branch

but

different surrounding geometry
different physical nonlocal transport
```

This is the role of the current Rain fixed-target experiment.

The fixed-target experiment is a **rigorous causal probe**, not the sole project-level kill test.

---

# 3. Two Hypothesis Layers Must Remain Separate

## 3.1 Phenomenon Hypothesis

> **High-quality learned shading / light-transport representations can lose validity under meaningful intrinsic geometry change.**

This should be evaluated primarily through cross-model and cross-asset observation.

It is an inductive hypothesis.

## 3.2 Mechanism Hypothesis

> **Some failures occur because persistent learned representation owns transport information whose validity depends on the current nonlocal geometry configuration.**

This requires controlled attribution.

It is a deductive hypothesis.

A failure of one mechanism test does not automatically falsify the phenomenon hypothesis.

Likewise, observing degradation does not prove the mechanism hypothesis.

---

# 4. Baseline Role Taxonomy

The word "baseline" must not be used without specifying its role.

## 4.1 RNA — Implementation / Development Base

**Current role:** primary experimental workbench and implementation substrate.

Why it is useful:

- explicit surface representation;
- inspectable geometry-to-network interface;
- practical deformation instrumentation;
- canonical correspondence can be controlled;
- current normal/view/light inputs can be audited;
- renderer-derived direct visibility can be separated from persistent learned state.

RNA is therefore the current best environment for:

- developing diagnostics;
- testing correspondence semantics;
- prototyping controlled deformation experiments;
- implementing early architecture hypotheses if later evidence justifies them.

Important:

> **RNA being the current development base does not mean the final method must be RNA-based.**

The future method substrate remains undecided.

---

## 4.2 8DNA — Scientific Replication Baseline

**Current role:** independent high-quality asset-specific neural-transport replication baseline.

8DNA should not currently replace RNA as the development base.

Its purpose is to answer:

> **Does a comparable failure recur in another high-quality asset-specific neural light-transport representation, or is the observed phenomenon mainly RNA-specific?**

Initial 8DNA work should prioritize:

1. faithful reproduction of the official high-quality setup;
2. a small number of meaningful material-preserving geometry changes;
3. broad qualitative/quantitative failure observation;
4. only then, if a transport-like failure exists, deeper correspondence and causal controls.

Do **not** begin by porting every RNA diagnostic into 8DNA.

Do **not** use 8DNA as the implementation base merely because it is newer or more transport-centric.

---

## 4.3 RenderFormer-Type Systems — Architecture Contrast Baseline

**Current role:** current-geometry-conditioned architectural contrast.

The purpose is not to rank models by image quality.

The contrast question is:

> **What changes when the representation is recomputed or conditioned directly from the current scene geometry instead of relying on a frozen canonical neural asset state?**

A geometry-explicit/current-scene system can help separate:

```text
"deformation is difficult for neural rendering in general"
```

from:

```text
"frozen/persistent asset-specific transport state loses validity when geometry changes"
```

This is an architecture contrast, not necessarily a drop-in implementation baseline.

---

## 4.4 Geometry-Aware / Dynamic Transport Methods — Future Positive Controls

Methods that explicitly rebuild, update, or condition transport state from current geometry may become useful positive controls.

Their role is to test the opposite side of the lifecycle hypothesis:

```text
persistent frozen state
vs
current geometry-conditioned state
```

Do not integrate these methods merely to expand the model list.

Use them only when they answer a concrete evidence question.

---

## 4.5 Future Method Base — Undecided

No final implementation substrate has been selected.

Possible outcomes include:

- extending RNA;
- adapting another existing transport system;
- implementing the research contract in a minimal independent framework;
- combining ideas across systems.

The final method base must be selected **after** the representation problem is sufficiently established.

---

# 5. Immediate Experimental Sequence

The current intended sequence is:

## Step A — Finish the Rain Fixed-Target Causal Probe

Purpose:

> Test whether physical radiance at fixed, canonically observed target points changes because surrounding geometry changes while frozen RNA receives effectively identical local inputs.

Interpretation rule:

- positive result: a clean Rain/RNA causal case is obtained;
- negative result: the mechanism is not observed in that Rain/RNA regime.

A negative result must **not** be reported as project-level falsification.

## Step B — Add 8DNA as an Independent Replication Baseline

First reproduce the official high-quality setup.

Then perform a bounded broad failure sweep on meaningful geometry changes.

Do not begin with deep architecture modification.

## Step C — Add a Geometry-Conditioned Contrast

Use a RenderFormer-type/current-geometry-conditioned system where practical.

The goal is to compare representation lifecycle assumptions, not to declare a winner.

## Step D — Expand Asset and Deformation Breadth

Do not remain on Rain alone.

At minimum include another asset family such as an articulated/body-like case.

Useful deformation families include:

- large bend/twist;
- fold creation/disappearance;
- cross-part approach;
- near-contact/self-contact;
- contact release.

---

# 6. Minimum Evidence Before Project-Level Falsification

The broad research hypothesis should not be killed from one Rain/RNA case.

Before making a strong project-level negative judgment, seek evidence across at least:

- **2 or more representation families**;
- **2 or more asset families**;
- **2 or more deformation mechanisms**;
- **1 or more strong controlled cases** where local state and nonlocal geometry effects are explicitly separated.

A strong negative body of evidence would look like:

1. comparable high-quality representations remain robust under meaningful geometry change; or
2. observed failures repeatedly reduce to local directional OOD, coverage, spatial lookup, or straightforward engineering causes; and
3. fixed-target/current-local-state-controlled tests fail to reveal missing nonlocal transport behavior; and
4. transport-like failure patterns do not recur across independent models/assets.

Only then should the broader representation-lifecycle research direction be considered substantially weakened.

---

# 7. Evidence Matrix

The project should gradually populate a matrix like:

| Model / Role | Cloth fold / self-approach | Articulated cross-part | Fixed-target nonlocal interaction |
|---|---|---|---|
| RNA — development base | In progress / historical Rain evidence | Einar or replacement asset later | Current causal probe |
| 8DNA — scientific replication | Required candidate | Candidate if practical | Only after a transport-like failure is observed |
| Current-geometry contrast | Required where practical | Required where practical | Architecture-level contrast |

The matrix is not a requirement to fill every cell.

The objective is sufficient diversity to avoid concluding from one model/asset pair.

---

# 8. Interpretation Rules

## Rule 1 — Do not over-generalize a Rain result

Say:

> "The mechanism was not observed in the tested Rain/RNA regime."

Do not automatically say:

> "The dynamic neural transport hypothesis is falsified."

## Rule 2 — Do not confuse baseline roles

RNA:
- implementation/development base.

8DNA:
- scientific replication baseline.

RenderFormer-type system:
- architecture contrast baseline.

Future method base:
- undecided.

## Rule 3 — Breadth and depth have different jobs

Broad sweep:

> discover recurring phenomena.

Deep control:

> identify mechanism.

Do not require full causal proof from every exploratory model.

Do not build a new architecture from a single unexplained failure.

## Rule 4 — Negative results remain valuable

A model-specific negative result can eliminate:

- one mechanism;
- one deformation regime;
- one representation-specific explanation.

It should be recorded at the scope actually tested.

---

# 9. Current Decision

As of 2026-09-28:

1. **Keep RNA as the implementation/development base.**
2. **Finish the current Rain fixed-target causal probe.**
3. **Use 8DNA next as a scientific replication baseline, not as the immediate method-development backbone.**
4. **Use a current-geometry-conditioned system such as RenderFormer as an architecture contrast when the experimental interface is practical.**
5. **Do not select the final method substrate yet.**
6. **Do not infer project-level falsification from Rain/RNA alone.**
7. **Return to inductive breadth after the current Rain causal probe, regardless of whether that probe is positive or negative.**

---

# 10. One-Line Operating Rule

> **Use breadth to establish that the phenomenon is real and recurring; use depth to prove why it happens; do not let one deeply studied model/asset pair decide the existence of the entire research problem.**
