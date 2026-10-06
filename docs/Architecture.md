# Architecture — Persistent Appearance + Dynamic Transport

**Status:** Working method architecture  
**Last updated:** 2026-10-06  
**Scope:** Central representation contract for the next method-design phase. This is not yet a frozen implementation architecture.

---

## 1. Why this document exists

The research-evidence phase established a controlled failure mode in which a persistent neural transport representation remains tied to the geometry configuration on which it was learned.

The strongest current evidence shows that, for a stationary query surface, the physically correct radiance can change because another part moves even when:

- the query surface identity is unchanged;
- its local geometry is unchanged;
- the view and light are unchanged;
- current direct-light visibility is still available.

In the accepted teaset case, the missing physical change is dominated by:

1. nonlocal occlusion of indirect incidence paths; and
2. radiance reflected from the moved part toward the query.

The code-level audit of RNA and 8DNA further showed that both methods, despite different implementations, learn inter-surface transport from one fixed internal configuration and do not provide the learned transport component with current internal-configuration state capable of updating that transport.

This document converts that diagnosis into a method-design contract.

---

## 2. Central design question

The method is not primarily asking:

> How can a larger MLP model dynamic global illumination?

It is asking:

> Which information should remain persistent when geometry changes, and which transport information must be updated from the current geometry configuration?

The architecture must preserve the compression and reuse advantages of neural transport without letting configuration-dependent inter-surface transport become stale persistent state.

---

## 3. Central structure

The current central structure is:

```text
Persistent Surface / Material State
            +
Current Configuration-Dependent GI State
            +
Shared Scattering / Transport Operator
            ↓
Current Outgoing Radiance
```

Conceptually, for surface anchor or query (i) at configuration (t):

[
L_o^t(i,omega_o)
=
F_	heta
left(
p_i,
g_i^t,
omega_o,
	ext{local geometry}
ight)
]

where:

- (p_i) is persistent across geometry configurations;
- (g_i^t) changes with the current geometry configuration;
- (F_	heta) is a shared reusable rule rather than an asset/configuration-specific GI memory.

The exact parameterization of (p_i), (g_i^t), and (F_	heta) remains open.

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

This state is intended to be learned offline and reused without full regeneration whenever possible.

Important boundary:

> Current cross-part arrangement must not be required to construct or refresh the persistent state during normal runtime updates.

The persistent state must not become a hidden store for canonical configuration-specific GI merely because that reduces training loss.

---

### 4.2 Current Configuration-Dependent GI State

The dynamic state represents transport information whose validity depends on the current geometry relation.

The evidence requires this state to be capable of representing at least:

- current nonlocal visibility of indirect incidence;
- current cross-surface occlusion relationships;
- radiance arriving from other surfaces;
- proximity/contact-dependent transport context;
- other configuration-sensitive inter-surface transport discovered later.

A useful current interpretation is:

> the dynamic state is a compact representation of the current incident-transport context seen by the surface.

This is intentionally weaker than full explicit inverse rendering. The method does not currently require exact decomposition into physically named variables such as BRDF, visibility, and incident radiance, but its information path must prevent configuration-dependent transport from being silently owned only by persistent state.

---

### 4.3 Shared Scattering / Transport Operator

The shared network should behave as a reusable computation rule, not primarily as a storage mechanism for one asset configuration.

Its intended role is approximately:

> Given persistent local appearance, current incident-transport state, local geometry, and view, predict current outgoing radiance.

The architecture should therefore avoid solving the problem merely by increasing MLP capacity.

Preferred property:

> Geometry configuration changes should be expressed mainly through the dynamic state, not by changing the shared operator weights.

---

## 5. Dynamic GI compression

A full directional incident-radiance field is expensive to store or recompute at every surface.

The current preferred direction is a compact directional representation:

[
L_i(omega)
approx
sum_{k=1}^{K}
c_{ik},B_k(omega)
]

where:

- (B_k) is a shared directional basis;
- (c_{ik}) is the current configuration-dependent coefficient state for surface/anchor (i).

Possible realizations include learned directional bases or other compact directional parameterizations.

This is a design direction, not yet a frozen choice.

The important contract is:

> Dynamic GI state should preserve transport-relevant directional information while remaining much smaller and cheaper to update than full scene transport recomputation.

---

## 6. Avoiding pairwise O(N²) transport

Naively relating every surface element to every other surface element recreates the pairwise cost that neural compression is meant to avoid.

The current architecture therefore separates current transport into two conceptual regimes.

### 6.1 Sparse near-field transport

Explicitly maintain only transport-relevant relations with a small neighborhood of surfaces likely to create strong configuration-sensitive effects, such as:

- near-contact surfaces;
- strong occluders;
- specularly coupled surfaces;
- cavity-forming geometry;
- nearby surfaces with strong reflected contribution.

If each anchor maintains (k) important relations with (k ll N), the target scaling is closer to:

[
O(Nk)
]

than (O(N^2)).

The exact neighbor-selection mechanism is not yet chosen.

### 6.2 Compressed far-field transport

The aggregate influence of the remaining scene should be represented coarsely rather than by explicit pairwise relations.

Candidate forms include:

- directional low-rank coefficients;
- hierarchical spatial aggregation;
- clustered transport summaries;
- another learned compact representation.

The method should not implement dense all-to-all attention over all surface elements unless evidence later justifies the cost.

---

## 7. Incremental update principle

Geometry change should not automatically invalidate the entire neural asset.

The target runtime behavior is:

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

A useful long-term target is a dirty-region update:

[
C_{	ext{incremental update}}
ll
C_{	ext{full transport recomputation}}
]

The exact invalidation and propagation strategy is not yet designed.

The project must not claim that runtime recomputation disappears completely. The goal is to replace expensive full-state regeneration or refitting with a much smaller current-state update.

---

## 8. Multi-bounce transport

Global illumination is recursive.

A future realization may update the dynamic transport state iteratively:

[
g_i^{(r+1)}
=
U_	heta
left(
g_i^{(r)},
p_i,
{g_j^{(r)}, r_{ij}}_{j in mathcal N_i},
g_i^{far}
ight)
]

where (r) is an update/transport iteration and (mathcal N_i) is a sparse current neighborhood.

This resembles message passing, but no graph neural network is selected yet.

The architectural principle is:

> A reusable transport rule may be persistent, while the relation topology/state over which transport propagates must reflect the current geometry.

---

## 9. Training contract

Multi-configuration training is necessary to make the desired ownership identifiable, but it is not itself the research contribution.

The same persistent surface identity should be shared across multiple geometry configurations:

[
p_i(G_0)=p_i(G_1)=cdots
]

while the dynamic state changes:

[
g_i(G_0)
eq g_i(G_1)
]

when transport relations change.

This forces configuration-dependent radiance differences to pass through the dynamic information path instead of being explainable only by a configuration-specific persistent representation.

### 9.1 Transport excitation, not motion magnitude

Training configurations should not be selected only by large geometric displacement.

A small motion can cause a large transport change if it opens or closes visibility, creates near-contact, forms a cavity, or changes strong interreflection.

Conversely, a large motion can be transport-irrelevant.

Therefore training diversity should target **transport-relation diversity**, including changes in:

- mutual visibility;
- occlusion topology;
- proximity/contact;
- relative direction/orientation;
- strong interreflection paths;
- creation or removal of important indirect paths.

This is an identifiability requirement.

---

## 10. What the method must beat

The architecture only has practical value if selective reuse is meaningfully better than rebuilding the current transport state.

Future comparisons must distinguish:

1. **Physical full recomputation**
   - current geometry → path-traced/current GI result;

2. **Neural full-state recomputation**
   - current geometry → full neural re-encoding/refit → render;

3. **Proposed incremental update**
   - persistent state reuse + affected current transport-state update → render.

Relevant metrics include:

- update latency;
- full regeneration/refit latency;
- render latency;
- effective frame rate;
- memory;
- training cost;
- quality;
- amount of state updated;
- scaling with scene and changed-region size.

A method that updates almost the whole scene at nearly full-regeneration cost does not satisfy the intended contribution.

---

## 11. Current method hypothesis

The current working hypothesis is:

> **Dynamic neural light transport can retain the compression and fidelity advantages of persistent neural assets if reusable surface/material information is stored persistently, while configuration-sensitive inter-surface transport is represented by a compact current state that can be updated locally from the current geometry and interpreted by a shared transport operator.**

A shorter internal shorthand is:

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

### Canonical GI leakage

Persistent state or shared weights silently memorize the canonical geometry's indirect transport.

### Dense relation explosion

Dynamic transport becomes all-to-all (O(N^2)) interaction.

### Full-scene re-encoding in disguise

The "dynamic update" is effectively a complete current-geometry encoder every frame.

### Weak multi-configuration identifiability

Training configurations do not sufficiently change transport relations, allowing the persistent state to retain most canonical GI and reducing the dynamic branch to a small correction.

### Quality-only success

The method improves quality but offers no meaningful update-cost advantage over full regeneration.

---

## 14. Evidence-to-architecture mapping

| Evidence | Architecture implication |
|---|---|
| stationary target radiance changes when another part moves | current local surface state alone is insufficient |
| direct transport contributes almost none of the canonical failure | direct visibility alone is insufficient |
| missing change is indirect-path occlusion + moved-part reflected radiance | dynamic state must represent current cross-surface transport |
| same-state 8DNA refit can recover T3 | stale state, rather than absolute representational incapacity, is a viable target |
| RNA and 8DNA share the failure through different implementations | avoid configuration-specific transport ownership in persistent learned state |
| large full recomputation may be expensive | update locality and compact dynamic state are first-class design goals |

---

## 15. Immediate next research tasks

The architecture is now sufficiently concrete to begin two bounded activities in parallel:

1. **Full-recomputation cost calibration**
   - use an official Cycles benchmark scene as a reproducible baseline;
   - begin with BMW27 and expand only if necessary;
   - measure the cost of obtaining current GI after geometry change on the RTX 5080;
   - use the result to define a meaningful runtime update budget.

2. **Minimal dynamic-state prototype design**
   - do not implement the full method yet;
   - decide the smallest directional/current transport representation that can express the already-measured teaset failure;
   - preserve explicit separation between persistent state and current transport state.

---

## 16. Completion question for the architecture phase

The architecture should eventually let the project answer:

> **Can configuration-dependent global transport be updated from current geometry at substantially lower cost than full recomputation, while persistent neural appearance remains reusable and rendering quality remains comparable?**

Until that question can be tested, this document remains a working architecture contract rather than a final method specification.
