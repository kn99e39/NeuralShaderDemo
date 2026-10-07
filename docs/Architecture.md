# Architecture — Persistent Appearance + Dynamic Transport

**Status:** Working method architecture  
**Last updated:** 2026-10-07  
**Scope:** Central representation contract for the next method-design phase. This is not yet a frozen implementation architecture.

---

## 1. Why this document exists

The research-evidence phase established a controlled failure mode in which a persistent neural transport representation remains tied to the geometry configuration on which it was learned.

In the accepted teaset case, a stationary query changes physically because another part moves even though the target surface identity, local geometry, view, light, and current direct-light handling remain valid. The missing physical change is dominated by nonlocal occlusion of indirect incidence plus radiance reflected from the moved part.

The RNA/8DNA code-level audit then showed that both released methods learn inter-surface transport from one fixed internal configuration and do not provide the learned transport component with current internal-configuration state capable of updating that transport.

This document converts that diagnosis into a method-design contract.

---

## 2. Central design question

The method is not primarily asking:

> How can a larger MLP model dynamic global illumination?

It is asking:

> Which information should remain persistent when geometry changes, and which transport information must be updated from the current geometry configuration?

The architecture must preserve neural compression/reuse without allowing configuration-dependent inter-surface transport to become stale persistent state.

---

## 3. Central structure

```text
Persistent Surface / Material State
            +
Current Configuration-Dependent GI State
            +
Shared Scattering / Transport Operator
            ↓
Current Outgoing Radiance
```

For a surface anchor or query `i` at configuration `t`, the conceptual form is:

```text
L_o^t(i, view) = F_theta(p_i, g_i^t, view, local_geometry)
```

where:

- `p_i` is persistent across geometry configurations;
- `g_i^t` changes with the current geometry configuration;
- `F_theta` is a shared reusable rule rather than an asset/configuration-specific GI memory.

The exact parameterization remains open.

---

## 4. Representation ownership

### 4.1 Persistent Surface / Material State

The persistent state should contain information whose validity does not depend on the current placement of remote surfaces.

Candidate contents include:

- material identity;
- local BRDF/BSDF-like behavior;
- local microstructure statistics;
- texture / surface identity;
- locally reusable scattering characteristics;
- other appearance information shown to remain valid across configuration changes.

Important boundary:

> Current cross-part arrangement must not be required to construct or refresh the persistent state during normal runtime updates.

The persistent state must not become a hidden store for canonical configuration-specific GI merely because that reduces training loss.

### 4.2 Current Configuration-Dependent GI State

The dynamic state represents transport information whose validity depends on the current geometry relation.

It must be capable of representing at least:

- current nonlocal visibility of indirect incidence;
- current cross-surface occlusion relationships;
- radiance arriving from other surfaces;
- proximity/contact-dependent transport context;
- other configuration-sensitive inter-surface transport discovered later.

A useful current interpretation is:

> **compact current incident-transport context**

This is intentionally weaker than full explicit inverse rendering. The method does not currently require exact decomposition into named physical variables, but its information path must prevent configuration-dependent transport from being owned only by persistent state.

### 4.3 Shared Scattering / Transport Operator

The shared network should behave primarily as a reusable computation rule, not as storage for one asset configuration.

Its intended role is:

> Given persistent local appearance, current incident-transport state, local geometry, and view, predict current outgoing radiance.

Geometry configuration changes should therefore be expressed mainly through the dynamic state, not by changing the shared operator weights.

---

## 5. Dynamic GI compression

A full directional incident-radiance field is too expensive to store or recompute densely at every surface.

The current preferred direction is a compact directional representation:

```text
L_i(direction) ≈ sum_k c_ik * B_k(direction)
```

where `B_k` is a shared directional basis and `c_ik` is the current configuration-dependent coefficient state for anchor `i`.

Possible realizations include learned directional bases or other compact directional parameterizations. This is a design direction, not a frozen choice.

---

## 6. Avoiding pairwise O(N²) transport

Naively relating every surface element to every other surface element recreates the pairwise cost that neural compression is meant to avoid.

### 6.1 Sparse near-field transport

Explicitly maintain only transport-relevant relations with a small neighborhood of surfaces likely to create strong configuration-sensitive effects, such as near-contact surfaces, strong occluders, specularly coupled surfaces, cavity-forming geometry, or nearby surfaces with strong reflected contribution.

If each anchor maintains `k` important relations with `k << N`, the target scaling is closer to `O(Nk)` than `O(N²)`.

### 6.2 Compressed far-field transport

The aggregate influence of the remaining scene should be represented coarsely rather than by explicit pairwise relations.

Candidate forms include:

- directional low-rank coefficients;
- hierarchical spatial aggregation;
- clustered transport summaries;
- another learned compact representation.

Dense all-to-all attention is not the default design.

---

## 7. Incremental update principle

Geometry change should not automatically invalidate the entire neural asset.

```text
Geometry change
    ↓
Identify changed transport relations / affected region
    ↓
Update only affected configuration-dependent GI state
    ↓
Reuse persistent surface/material state
    ↓
Reuse shared transport operator
    ↓
Render
```

The target is:

```text
incremental update cost << full transport recomputation cost
```

The project does not claim that runtime recomputation disappears completely. The goal is to replace expensive full-state regeneration/refitting with a much smaller current-state update.

---

## 8. Multi-bounce transport

Global illumination is recursive, so a future realization may update the dynamic state iteratively over sparse current relations.

This may resemble message passing, but no graph neural network is selected yet.

The architectural principle is:

> A reusable transport rule may be persistent, while the relation topology/state over which transport propagates must reflect the current geometry.

---

## 9. Training contract

Multi-configuration training is necessary to make the desired ownership identifiable, but it is not the research contribution by itself.

The same persistent surface identity should be shared across configurations, while the dynamic state changes when transport relations change.

Training diversity should target **transport-relation diversity**, not just large geometric displacement. Relevant changes include:

- mutual visibility;
- occlusion topology;
- proximity/contact;
- relative direction/orientation;
- strong interreflection paths;
- creation/removal of important indirect paths.

This is an identifiability requirement.

---

## 10. What the method must beat

Future evaluation must distinguish:

1. **Physical full recomputation** — current geometry → path-traced/current GI result;
2. **Neural full-state recomputation** — current geometry → full neural re-encoding/refit → render;
3. **Proposed incremental update** — persistent-state reuse + affected current transport-state update → render.

Relevant metrics include update latency, render latency, effective frame rate, memory, training cost, quality, amount of state updated, scene-size scaling, and changed-region scaling.

Measured reference points so far (RTX 5080; different scenes, not quality matched, cost context only):

| baseline | measured cost after a rigid configuration change | source |
|---|---|---|
| physical full recomputation (Cycles, BMW27, 1080p, persistent data) | ≈0.22 s / 0.76 s / 3.0 s per frame at 16 / 64 / 256 spp | worklog 26 |
| neural full-state recomputation, 8DNA (teaset T3, historical refit) | 38.9 min to the first rule-satisfying checkpoint; 1.76 h fixed schedule | worklog 27 |
| neural full-state recomputation, RNA (teaset T3, historical refit) | 3.24 h fixed schedule (87% target rendering); no rule-satisfying checkpoint | worklog 27 |

Existing neural regeneration is offline and is not the binding constraint; physical recomputation is the tighter runtime reference. Faster neural refit variants have not been measured.

**Implementation viability gate (worklogs 26/27): PASS for a bounded prototype.** The existing alternatives leave a large enough latency gap to justify testing selective reuse, but this does not establish that the proposed update can close that gap or preserve quality.

A method that updates almost the whole scene at nearly full-regeneration cost does not satisfy the intended contribution.

---

## 11. Current method hypothesis

> **Dynamic neural light transport can retain the compression and fidelity advantages of persistent neural assets if reusable surface/material information is stored persistently, while configuration-sensitive inter-surface transport is represented by a compact current state that can be updated locally from the current geometry and interpreted by a shared transport operator.**

Short internal form:

> **Persistent appearance, dynamic transport state, reusable transport rule.**

---

## 12. Explicit non-goals and uncommitted choices

The following are not yet architecture decisions:

- graph neural network;
- transformer or sparse attention;
- exact surface-anchor granularity;
- exact number of anchors;
- MLP layer count / width;
- latent dimensions;
- spherical harmonics vs learned directional basis;
- exact near-field neighbor count;
- exact far-field hierarchy;
- runtime cache structure;
- number of multi-bounce update iterations;
- RNA vs 8DNA as the final implementation substrate;
- explicit full BRDF/material inverse rendering.

Do not turn these into commitments without an experiment that distinguishes the alternatives.

---

## 13. Failure modes this architecture must avoid

- **Canonical GI leakage:** persistent state/shared weights memorize canonical indirect transport.
- **Dense relation explosion:** dynamic transport becomes all-to-all `O(N²)` interaction.
- **Full-scene re-encoding in disguise:** the dynamic update effectively rebuilds the whole current scene.
- **Weak multi-configuration identifiability:** insufficient transport excitation lets the persistent state retain most canonical GI.
- **Quality-only success:** quality improves but update cost offers no meaningful advantage over regeneration.

---

## 14. Evidence-to-architecture mapping

| Evidence | Architecture implication |
|---|---|
| stationary target radiance changes when another part moves | current local surface state alone is insufficient |
| direct transport contributes almost none of the canonical failure | direct visibility alone is insufficient |
| missing change is indirect-path occlusion + moved-part reflected radiance | dynamic state must represent current cross-surface transport |
| same-state 8DNA refit can recover T3 | stale state, rather than absolute representational incapacity, is a viable target |
| RNA and 8DNA share the failure through different implementations | avoid configuration-specific transport ownership in persistent learned state |
| full recomputation is expensive: neural regeneration 39 min – 3.2 h (WL27), physical GI ≈0.2–3 s/frame at 16–256 spp (WL26) | update locality and compact dynamic state are first-class design goals; the update must target the physical-recompute scale, not the neural-regeneration scale |
| WL28: frozen RNA + local-only residual cannot move at held-out T3 | the current state must be nonlocal (H1 insufficiency, now also empirical) |
| WL28: frozen RNA + sparse geometric relation probes (K = 32: hit, distance, remote normal, remote persistent feature, remote direct visibility) fits training configurations but does not generalise to held-out T3 | geometric relations alone are an insufficient current state; the state must carry **radiometric incident transport** (radiance arriving along each direction), not only which surface is hit |
| WL28 oracle: reference path-class radiance through the same shared operator, frozen RNA and training contract recovers T3 | the lifecycle split (persistent appearance + current state + shared operator) is viable when the state has the right content; keep it |

---

## 15. Immediate next research tasks

1. **Full-recomputation cost calibration — done**
   - physical: BMW27 / BMW Garage XL Cycles on the RTX 5080 (worklog 26);
   - neural: RNA and 8DNA refit pipelines on the teaset T3 state (worklog 27);
   - resulting working budget: an update of roughly ~1–10 ms per change inside a frame (worklog 26's estimate, not a validated threshold); see §10.

2. **Minimal dynamic-state prototype — first prototype done (worklog 28, CASE C)**
   - sidecar over frozen RNA with a sparse geometric relation state (O(Q·K), K = 32; ~16 ms state update, 0.26 ms decoder for 66 k queries): **does not** recover held-out T3; the oracle incident-transport state does;
   - the original task statement follows, kept for reference:
   - implementation may now begin as a **bounded architecture experiment**, not as the full end-to-end method;
   - use the canonical teaset T0/T3 failure first;
   - implement the smallest current directional/relational transport-state path that can express the measured missing transport while keeping persistent appearance/state fixed;
   - preserve explicit separation between persistent state and current transport state;
   - compare against the frozen baseline and the historical full T3 recompute/refit control;
   - treat update latency, changed-state fraction, recovery quality, and canonical-GI leakage as first-class measurements;
   - stop after this prototype if the current-state abstraction itself is insufficient; do not rescue it with parameter sweeps or dense all-to-all interaction.

3. **Next bounded question — current task (own batch and protocol)**
   - give the current state runtime-constructible **radiometric** incident-transport content (radiance arriving along each probe from the current neighbours), keeping the frozen persistent path, the shared operator, the local-only control, the T1/T3 hold-out and the historical rule;
   - separate the confounds worklog 28 left open: training-relation coverage (two changed training configurations, both weaker than T3) and probe angular resolution for near-mirror transport;
   - keep the update within the measured order (~16 ms for 66 k queries in the first prototype) and record changed-state fraction and canonical-GI leakage.

---

## 16. Completion question for the architecture phase

> **Can configuration-dependent global transport be updated from current geometry at substantially lower cost than full recomputation, while persistent neural appearance remains reusable and rendering quality remains comparable?**

Until that question can be tested, this document remains a working architecture contract rather than a final method specification.