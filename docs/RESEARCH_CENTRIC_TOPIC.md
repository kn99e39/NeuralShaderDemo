# Research Centric Topic — Dynamic Neural Shading / Neural Light Transport

## Status

This document defines the **central research topic and interpretation contract** for this project.

It is not a final method proposal, not a claim of novelty, and not an implementation specification.

All experiments, diagnostics, architecture discussions, and future method development should remain aligned with the research question defined here unless the project direction is explicitly changed.

---

# 1. Centric Research Topic

Recent high-quality Neural Shading / Neural Light Transport methods can compress complex appearance and transport effects very efficiently, including:

- interreflection,
- self-shadowing,
- multiple scattering,
- near-field transport,
- complex layered appearance,
- fine-scale geometry-dependent shading.

This ability is a major strength of neural transport representations.

However, the learned transport may also become dependent on the **geometry configuration present during training**.

When geometry changes substantially — especially under:

- non-rigid deformation,
- articulation,
- fold creation or disappearance,
- cross-part approach,
- self-contact,
- contact release,
- compound deformation,
- other open-ended geometry changes,

some of the previously learned transport may no longer remain physically valid.

The central research problem is therefore not merely:

> "Can neural compression be applied to deforming geometry?"

That question is too broad and overlaps with existing work.

The actual research interest is:

> **What geometry-dependent information is encoded inside a high-quality neural transport representation, and when geometry changes, what information should remain reusable, what should be updated, and what should be recomputed?**

---

# 2. Core Motivation

A static high-quality neural asset can be viewed conceptually as compressing some combination of:

- intrinsic material response,
- local geometric appearance,
- microstructure,
- self-visibility,
- self-shadowing,
- intra-object interreflection,
- higher-order multiple scattering,
- scene- or asset-specific transport.

For a fixed asset configuration, strongly compressing these effects together can be highly effective.

The architectural problem appears when geometry becomes mutable.

If a representation trained on geometry state:

\[
G_0
\]

implicitly stores transport associated with that state:

\[
T(G_0)
\]

then after deformation to:

\[
G_1
\]

the physically correct transport may instead be:

\[
T(G_1)
\]

and therefore:

\[
T(G_0) \neq T(G_1)
\]

for some regions or transport components.

The scientific question is not simply whether an image metric becomes worse.

The important question is:

> **Which parts of the learned representation remain semantically valid across the geometry change, and which parts have become stale because they encoded geometry-specific transport?**

---

# 3. Representation-Level View

The current research is fundamentally about **representation ownership**.

A useful conceptual distinction is:

## Persistent information

Information that may remain reusable when geometry changes.

Examples may include:

- material identity,
- local BRDF / BSDF characteristics,
- texture identity,
- certain microstructure statistics,
- transferable appearance priors.

## Geometry-dependent information

Information whose validity depends on the current geometry configuration.

Examples may include:

- self-visibility,
- contact shadowing,
- mutual occlusion,
- cavity structure,
- inter-part interreflection,
- geometry-conditioned multiple scattering,
- some near-field transport dependencies.

The exact boundary is **not yet known**.

Determining that boundary is part of the research.

A useful abstract target is:

\[
\text{Persistent Appearance}
+
\text{Current Geometry}
\rightarrow
\text{Current Light Transport}
\]

However, this equation is only a conceptual framing.

It does **not** imply that explicit factorization is already the chosen solution.

---

# 4. What Existing Approaches Already Cover

Do not assume that dynamic neural shading is an unexplored topic.

Existing research already includes approaches that:

1. recompute or predict transport from the current geometry,
2. generate a new transport state conditioned on pose or deformation,
3. adapt a neural representation online,
4. reuse local/object-level transport while dynamically recomputing other components,
5. generalize transport prediction across multiple geometry or scene configurations.

Therefore:

> **"Use neural compression for deformation" is not itself a novel research contribution.**

The unresolved territory of interest is more specific:

> **Can we understand and redesign the ownership of information inside high-quality neural transport representations so that the fidelity and compression advantages of static neural assets can survive substantially more general geometry changes?**

---

# 5. Current Working Hypothesis

The current working hypothesis is:

> A high-quality neural shading / neural light-transport representation trained on a static asset may entangle reusable material appearance with geometry-state-specific light transport.

If this is true, a frozen representation evaluated after intrinsic geometry deformation may retain transport biased toward the original geometry.

Possible observable symptoms include:

- a shadow that should disappear but remains partially encoded,
- a newly formed cavity whose indirect lighting is not represented correctly,
- stale interreflection after two surfaces move apart,
- missing interreflection after two surfaces move closer,
- incorrect shading around newly formed self-contact,
- appearance that remains biased toward the canonical pose despite correct current geometry inputs.

This hypothesis must be experimentally tested.

It must not be treated as established fact.

---

# 6. Important Distinction: Trivial Failure vs Research-Relevant Failure

A deformation experiment can fail for reasons unrelated to the central research question.

For example:

- the deformed surface queries the wrong spatial feature,
- canonical/deformed coordinates are mismatched,
- normals are not updated correctly,
- tangent frames are stale,
- the renderer uses inconsistent light/view coordinates,
- correspondence between canonical and deformed surfaces is broken.

These are implementation or representation-indexing failures.

They are **not sufficient evidence** for geometry-conditioned transport entanglement.

The target phenomenon is stronger:

> The same material/surface identity is preserved correctly, current geometry information is provided correctly, and yet the frozen neural representation still produces transport inconsistent with the new geometry configuration.

Experiments must explicitly separate these cases.

---

# 7. Priority Failure Cases

The most informative deformation families currently include:

## Fold Creation / Disappearance

A surface develops or loses a fold or cavity.

Useful for observing:

- new self-shadow,
- trapped-light regions,
- new or disappearing interreflection,
- higher-order cavity transport.

## Cross-Part Approach / Self-Contact

Two surface regions approach, nearly touch, touch, or separate.

Useful for observing:

- mutual visibility changes,
- contact shadows,
- near-field transport changes,
- nonlocal transport dependencies while local surface identity remains mostly unchanged.

## Compound / Open-Ended Deformation

Multiple deformation modes occur together, such as:

- bending + twisting,
- folding + self-contact,
- cross-part approach + twist.

Useful for testing whether the representation remains robust outside a simple low-dimensional pose family.

---

# 8. Research Sequence

The project should proceed in this order unless evidence motivates a change.

## Stage 1 — Existence

Determine whether high-quality static neural transport representations actually lose validity under meaningful geometry change.

## Stage 2 — Attribution

Determine what causes the failure.

Possible factors include:

- local geometry change,
- visibility,
- self-shadowing,
- interreflection,
- higher-order transport,
- representation indexing,
- insufficient architecture capacity.

## Stage 3 — Boundary

Determine where reuse remains valid and where it breaks.

Questions include:

- which deformation types matter,
- what deformation magnitude matters,
- whether failure correlates better with geometric displacement or transport change,
- whether some transport components remain reusable.

## Stage 4 — Representation Architecture

Only after Stages 1–3 provide evidence, investigate:

- what should remain persistent,
- what should be conditioned on current geometry,
- what should be invalidated,
- what should be updated,
- what should be recomputed.

## Stage 5 — Dynamic Generalization

Evaluate whether the redesigned representation can extend high-quality neural shading from static assets toward broader dynamic scenes.

## Stage 6 — Inverse Rendering Extension

Only later, if relevant, study how the representation can be recovered from real observations.

Inverse rendering is not required to establish the core representation problem.

---

# 9. Research Questions

The current primary research questions are:

### Q1 — Existence

Does a high-quality neural light-transport representation trained on a static asset systematically lose validity when the asset undergoes intrinsic geometry deformation?

### Q2 — Attribution

If failure occurs, which transport components or representation mechanisms are responsible?

### Q3 — Validity Boundary

Under what geometry changes does frozen transport remain reusable, and when does it become invalid?

### Q4 — Representation Ownership

What information should be geometry-invariant, and what information should depend on or be recomputed from current geometry?

### Q5 — Dynamic Reusability

Can a neural asset preserve the fidelity and compression advantages of static neural transport while remaining reusable under substantially more general dynamic geometry?

Q4 and Q5 must not be treated as solved before Q1–Q3 are established experimentally.

---

# 10. Current Experimental Philosophy

Experiments should prioritize **directly observable, high-quality failure cases**.

Synthetic scenes are useful for:

- debugging,
- correspondence validation,
- controlled attribution.

But synthetic success or failure alone is not enough to establish architectural relevance.

Important claims should ultimately be supported by:

- high-quality neural transport baselines,
- meaningful non-rigid deformation,
- path-traced references,
- quantitative accounting,
- localized error attribution,
- human-reviewable visual evidence.

A useful result is not merely:

> "PSNR decreased."

A stronger result is:

> "After surface identity and current local geometry were handled correctly, error became concentrated in newly formed cavity/contact regions where the physical transport changed, while unrelated regions remained stable."

---

# 11. What Must Not Be Assumed

Do not assume any of the following without evidence:

- that all modern neural shading methods fail under deformation,
- that deformation failure is always severe,
- that neural compression itself is the problem,
- that explicit factorization is necessarily the correct solution,
- that online adaptation is inferior,
- that canonical-space shading is sufficient,
- that geometry-conditioned MLPs solve the issue,
- that pose conditioning generalizes to open-ended deformation,
- that one RNA result generalizes to all neural transport representations,
- that a synthetic scene proves real high-quality asset behavior,
- that metric degradation automatically proves transport entanglement.

Negative results are scientifically valid.

---

# 12. Premature Solution Space — Do Not Commit Yet

Until the failure is sufficiently established and attributed, do not automatically adopt:

- dynamic latent fields,
- deformation networks,
- geometry-conditioned MLPs,
- temporal networks,
- online fine-tuning,
- transport residual prediction,
- explicit BRDF / transport factorization,
- canonical-space neural shaders,
- per-frame adaptation,
- graph neural networks,
- transformers,
- local transport caches.

These remain possible future approaches, not current commitments.

---

# 13. Intended Long-Term Direction

If the representation problem is real and can be addressed, the long-term goal is:

> **Extend high-quality static neural shading / neural transport representations into reusable dynamic representations that preserve strong compression and rendering fidelity while remaining physically valid under broader geometry changes.**

Potential downstream domains include:

- animated characters,
- cloth,
- hair and fibers,
- soft-body simulation,
- interactive game assets,
- deformable digital twins,
- destructible or editable assets,
- physically changing scenes.

The long-term vision may be broad, but early claims must remain bounded by actual evidence.

---

# 14. Agent Interpretation Contract

Any Agent working in this repository must keep the following distinction clear:

## Implementation success is not research success.

A script running, a scene rendering, or a metric changing does not establish the research hypothesis.

## Observation and interpretation must remain separate.

Reports should distinguish:

- **IMPLEMENTATION FACT**
- **MEASUREMENT**
- **OBSERVATION**
- **INTERPRETATION**
- **UNRESOLVED QUESTION**

## Do not optimize toward a desired failure.

Do not:

- tune deformation magnitude purely to maximize error,
- choose lighting only because it exaggerates a hypothesis,
- loosen controls because the expected failure is weak,
- silently modify the baseline representation,
- interpret bugs as scientific evidence.

## Preserve baselines.

When testing a new mechanism, keep the previous baseline reproducible.

## Stop conditions matter.

If the experiment does not support the hypothesis, report that result rather than adding heuristics until the expected behavior appears.

---

# 15. One-Sentence Project Definition

> **This project investigates how high-quality neural shading and neural light-transport representations should partition persistent appearance information and geometry-dependent transport so that their compression and fidelity advantages can remain useful under non-rigid and open-ended geometry change.**

---

# 16. Central Intent for Future Agents

When deciding whether a task is aligned with this project, ask:

> **Does this work help determine what information inside a high-quality neural transport representation remains valid across geometry change, what becomes invalid, and how that ownership should eventually be represented?**

If the answer is no, the task is likely peripheral unless explicitly approved.

