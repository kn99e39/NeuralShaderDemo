# Baseline Roles and Evidence Strategy — Dynamic Neural Shading / Neural Light Transport

**Original decision date:** 2026-09-28  
**Last updated:** 2026-10-07 (worklog 28)  
**Status:** Active research operating decision

## Document Role

This document records the current baseline roles, evidence scope, and research-process rules for the project. It complements `RESEARCH_CENTRIC_TOPIC.md` and `RESEARCH_ROADMAP.md`.

The central distinction remains:

- **inductive breadth** establishes whether a phenomenon recurs;
- **deductive controls** identify why it occurs;
- neither one should be mistaken for the other.

---

# 1. Current Research State

The project now has two major evidence lineages.

## 1.1 RNA × Rain

RNA/Rain remains the mechanism-development bench that established:

- a real frozen deformation failure;
- correspondence and canonical-query controls;
- direct-visibility diagnostics;
- strong local-input/spatial confounds;
- a fixed-target causal-probe design.

The Rain fixed-target probe closed as **PHYSICAL EFFECT TOO WEAK**. It did not provide a positive or negative result for nonlocal transport failure because the target GT radiance did not change above render noise.

Therefore Rain remains useful historical evidence, but it is no longer the cleanest mechanism case.

## 1.2 Teaset cross-part configuration

Worklogs 21 and 22 provide the current strongest controlled evidence.

- In Worklog 21, a stationary teaset interaction surface kept its canonical query while another rigid part moved.
- The physical interreflection changed strongly above render noise.
- Frozen 8DNA barely followed that change, while relation-preserving whole-asset motion remained stable.
- A same-state T3 refit under the original envmap regime recovered close to T0 quality, strongly supporting stale persistent state rather than insufficient architecture capacity in that tested regime.
- In Worklog 22, RNA reproduced the same frozen-response signature on the same teaset states under a common-light regime.
- Both backbones fail the predeclared rule at T1b and T3 and pass the relation-preserving T1 control.

This is the project's first controlled **cross-backbone** support for the representation-lifecycle hypothesis.

Important limitation:

> The cleanest evidence is still one asset family and one main relation-change mechanism. It is strong mechanism evidence, not broad generality.

---

# 2. Hypothesis Layers

## 2.1 Phenomenon Hypothesis

> **A high-quality frozen neural shading / light-transport representation can lose validity when geometry relations change.**

This hypothesis now has controlled support in teaset across two representation families.

## 2.2 Mechanism Hypothesis

> **Some failures occur because persistent learned state owns transport information whose validity depends on the current nonlocal geometry configuration.**

The teaset evidence strongly supports this mechanism in the tested regime, but the exact transport component and generality remain open.

Do not generalize this to all neural transport methods, all materials, or all deformation families.

---

# 3. Baseline Role Taxonomy

## 3.1 RNA — Implementation / Development Base

RNA remains the primary experimental workbench and likely first method-development substrate because its geometry/network boundary is inspectable and its correspondence semantics are practical to control.

RNA is useful for:

- diagnostics;
- controlled correspondence experiments;
- future H1/H2 sufficiency tests;
- early architecture prototyping if evidence justifies it.

RNA being the current development base does not mean the final method must be RNA-based.

Worklog 28 used RNA this way for the first bounded prototype: the historical T0 RNA is a frozen, bit-exact persistent base, and the new current-state path and residual operator live in a project-owned sidecar. This keeps the historical baseline reproducible while testing the architecture.

## 3.2 8DNA — Scientific Replication Baseline

8DNA has now completed its first intended scientific role:

> **independent replication of the frozen geometry-configuration validity failure outside RNA.**

It remains valuable for:

- controlled replication;
- same-state refit/capacity controls;
- later testing of whether a proposed principle transfers beyond RNA.

Do not automatically promote 8DNA to the main method-development backbone.

## 3.3 RenderFormer-Type / Current-Geometry Systems — Architecture Contrast

These remain future contrast baselines.

The intended question is:

> **What changes when transport is derived from or conditioned on current geometry instead of relying on persistent canonical asset state?**

Use such a system only when it answers a concrete evidence question. Do not add it merely to increase the baseline count.

## 3.4 Future Method Base — Undecided

No final substrate has been selected.

Possible outcomes still include:

- extending RNA;
- adapting another transport framework;
- implementing the research contract in a small independent framework;
- combining ideas across systems.

---

# 4. Evidence Interpretation Rules

## Rule 1 — Do not over-generalize one asset

The teaset result is a clean controlled case, not a universal claim.

Current defensible statement:

> **The representation-lifecycle hypothesis has controlled support across RNA and 8DNA on one shared geometry-configuration benchmark.**

Not yet defensible:

> **All neural transport representations fail under dynamic geometry.**

## Rule 2 — Separate problem replication from method validation

Worklog 22 is cross-backbone **problem evidence**.

Future cross-backbone testing of a proposed solution is a different milestone and should not be conflated with the evidence already obtained.

## Rule 3 — Keep stale-state and capacity questions separate

If frozen state fails and same-state refit succeeds, stale state is strongly supported.

If refit also fails, capacity / optimization remains plausible.

The original 8DNA envmap refit is a strong stale-state control, and it reproduced in a second, independent rebuild (worklog 27: last checkpoint 1.149 / 0.629 vs 1.073 / 0.698).

The common-light refits are near the predeclared threshold and are **not robust** across runs: 8DNA failed the ratio rule in both runs (1.28, worklog 22; 1.36, worklog 27); RNA's worklog-22 pass (1.22) did not recur in the worklog-27 rebuild (≈1.32, no checkpoint passes, although targets and evaluation were bit-identical). Treat both as partial recovery, never as evidence that one backbone has more capacity.

Note for reading worklog 22: its 8DNA `last.ckpt` held epoch-28 weights (worklog 27); its verdicts are unchanged.

## Rule 4 — Breadth and depth still have different jobs

Depth has now produced a clean mechanism case.

Breadth is still needed across:

- another asset family;
- another geometry-change mechanism;
- less mirror-like transport/material regimes;
- ideally non-rigid or contact/release cases.

## Rule 5 — Negative results remain valid

A negative result should only weaken the hypothesis at the scope actually tested.

---

# 5. Current Evidence Matrix

| Model / role | Rain non-rigid scarf | Teaset cross-part relation | Same-state refit |
|---|---|---|---|
| RNA — development base | Failure observed; mechanism confounded | **Frozen failure reproduced** under common light; thin static-quality margin | Near threshold; WL22 borderline pass did not recur in the WL27 rebuild |
| 8DNA — scientific replication | Not tested | **Controlled frozen failure observed** | **Envmap T3 refit recovers** (reproduced, WL27); common-light near threshold, fails the rule in both runs |
| Current-geometry contrast | Not tested | Future | Future |

Additional dimensions still missing:

- another asset family;
- another deformation/relation mechanism;
- diffuse/glossy rather than near-mirror-dominated interaction;
- contact/release or true non-rigid interaction.

---

# 6. Minimum Evidence Before Broad Project Claims

The project now satisfies:

- at least two representation families in one shared controlled regime;
- one strong case separating stationary local state from changed nonlocal geometry;
- one strong same-state refit control in 8DNA.

It does **not** yet satisfy broad generality across:

- two or more clean asset families;
- two or more clean deformation/relation mechanisms.

Therefore the correct current position is:

> **The mechanism has strong controlled support in the tested teaset regime and has replicated across two backbones, but broad generality remains open.**

---

# 7. Current Experimental Sequence

## Step A — Rain fixed-target probe

**CLOSED:** PHYSICAL EFFECT TOO WEAK.

## Step B — 8DNA independent replication

**CLOSED for first objective:** controlled teaset failure observed.

## Step C — Same-scene cross-backbone replication

**CLOSED for first objective:** RNA reproduces the frozen failure signature on the same teaset state protocol under common light.

## Step D — Close attribution enough for a representation test

**CLOSED (worklogs 24, 25):** the missing information is current cross-part relational transport state (nonlocal visibility of indirect light plus the moved part's reflected radiance); both methods own it in persistent learned state by their representation contract.

## Step D2 — Full-recomputation cost baselines

**CLOSED (worklogs 26, 27):** physical Cycles recomputation on BMW27 costs ≈0.2–3 s per 1080p frame at 16–256 spp; the existing RNA/8DNA refit pipelines need 39 min (8DNA, first rule-satisfying checkpoint) to 1.8–3.2 h (fixed schedules) on the teaset T3 state. Different scenes; cost context only. These are the *existing* baselines' costs, not a property of neural transport in general.

## Step E — Current-geometry contrast / additional breadth

Future, when it answers a concrete decision.

---

# 8. Current Decision

As of 2026-10-07:

1. **Keep RNA as the implementation/development base.**
2. **Treat the Worklog 21/22 teaset case as the current canonical controlled failure case.**
3. **Treat 8DNA as a successful scientific replication baseline, not the default method-development substrate.**
4. **Do not reopen the old Rain mechanism as the main evidence path unless a new question specifically requires it.**
5. **Do not select the final method architecture yet.**
6. **Do not spend a large batch resolving the common-light refit split unless that distinction changes the next architecture decision.** Worklog 27 showed it is within run-to-run variation (RNA 1.22 -> 1.32 with identical data); neither common-light refit is a reliable recovery.
7. **M2 is closed for the canonical case (worklogs 24/25); the next representation experiment must supply current cross-part relational transport state.**
8. **Use worklog 26 (physical) and worklog 27 (neural regeneration) as the cost baselines a future method is compared against; existing neural regeneration is offline and is not the binding constraint.**
9. **Continue to separate cross-backbone problem evidence from future cross-backbone method validation.**

---

# 9. One-Line Operating Rule

> **The phenomenon now has a clean cross-backbone controlled case; the next job is to identify the minimal missing current-state information without over-generalizing from one asset or prematurely committing to a solution architecture.**
