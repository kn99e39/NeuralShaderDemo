# Research Centric Topic — Surface-Owned Selective Neural Transport Recomputation

**Last updated:** 2026-10-14 (worklog 32)  
**Status:** Active problem; Surface-Owned Transport Units are the active representation hypothesis (Batch 1 of ≥ 5 architecture batches reported); full method not validated  
**Target:** SIGGRAPH 2027 (official Technical Papers deadline to be verified separately)  
**Superseded framing:** object-level-only selection (2026-10-12), archived at docs/history/ARCHITECTURE_OBJECT_LEVEL_SELECTION_2026-10-12.md

## 1. Fundamental Research Problem

High-quality neural shading / learned global illumination fitted to a fixed geometry configuration can retain stale interreflection, indirect-path occlusion, glossy and multi-bounce transport after a geometry edit. Re-evaluating the frozen shader with current local surface inputs is not necessarily sufficient.

**Fundamental question:** after a geometry edit, can learned transport be repaired by recomputing the **complete owned neural state of every invalidated transport unit**, while every still-valid unit is reused unchanged?

A single large object can contain regions whose learned transport has become invalid while other regions remain valid. The representation should therefore be able to divide an object's surface into multiple independently reconstructible ownership domains.

## 2. User Decisions in Force (2026-10-13, recorded in worklog 32)

1. RNA (Relightable Neural Assets, adobe-research @ `66b5b098`) is the **initial implementation baseline** — a research substrate, not a permanent constraint or the required final model.
2. The ownership architecture **may be finer than a scene object**.
3. **Surface-Owned Transport Units** are the active research hypothesis.
4. Each invalidated unit **may have its complete owned neural state recomputed** (every parameter it owns).
5. **Object-level full recomputation remains a valid coarse-grained comparison.**
6. At least **five meaningful architecture batches** come before a broad viability judgment.
7. A negative result for one candidate does **not** falsify the general ownership hypothesis.
8. Dynamic-GI research of roughly **2007–2013** is the priority historical design reference (section 5).
9. The 1997 line-space hierarchy (Drettakis & Sillion) is **not** the primary architectural starting point.
10. Full efficiency and scaling validation come **after representation feasibility**.

Never reinterpret "selective" as a mask over triplane cells, feature channels or MLP layers inside one jointly owned model (worklog 30's tested and negative hypothesis). Selection is over **owned units**, each of which is recomputed whole.

## 3. Transport Unit Contract

A unit is U_i = (S_i, Θ_i):
- S_i — a well-defined owned surface domain; every evaluated surface sample resolves to exactly one owner (or a declared explicit composition of owners);
- Θ_i — the full set of learned parameters owned by that domain; updating U_i never changes Θ_j (j ≠ i).

Unit ownership is **storage and update ownership, not physical locality**: a unit's targets are full-scene GI, including light from other objects. A shared trainable decoder that must change to recover one unit is a representation coupling to report, never to hide with optimizer masks.

Separate properties, always reported separately:
- **parameter preservation** (Θ_j bit-identical),
- **output preservation** (predictions read only from preserved units are unchanged),
- **radiometric validity** (a preserved unit's prediction is still correct for the current scene).
The first two are contracts of the representation; the third depends on the scene edit.

## 4. Geometry-Dirty vs Transport-Dirty

- D_G: units/objects whose geometry was directly edited.
- D_T: units whose **learned transport is invalid under G1**, including stationary receivers whose indirect light changed.

D_G ≠ D_T. Worklog 31 measured this on the canonical teaset: mover-only refit left the stationary receivers as stale as the frozen model; at object granularity every object was transport-dirty, so whole-object selection could save nothing there. Finer ownership is the motivation for the current hypothesis — whether D_T becomes usefully sparse at unit granularity is an open empirical question.

## 5. Prior Art and Historical Design References

Priority references (advisor-approved direction; design references only, nothing implemented automatically):

| Reference | What it contributes | Category |
|---|---|---|
| Gautron, Bouatouch, Pattanaik, *Temporal Radiance Caching*, IEEE TVCG 13(5), 2007 | reuses cached (ir)radiance records across animation frames with temporal gradients; glossy GI in animated scenes | radiance caching, temporal reuse / invalidation by age-gradient |
| Debattista, Dubla, Banterle, Santos, Chalmers, *Instant Caching for Interactive Global Illumination*, CGF 28(8), 2009 | irradiance cache + instant radiosity; updates only light paths invalidated by geometry changes | radiance caching, transport invalidation |
| Wang, Zhou, Pan, Bao, *An Efficient GPU-Based Approach for Interactive Global Illumination*, SIGGRAPH 2009 | clusters shading points (adaptive seeding + k-means), final gather at cluster centres, interpolation | sampling and scheduling (clustered evaluation) |
| Crassin, Neyret, Sainz, Green, Eisemann, *Interactive Indirect Illumination Using Voxel Cone Tracing*, CGF 30(7) (PG) 2011 | on-the-fly voxel octree of the scene, cone-traced visibility/incoming energy, 2 bounces incl. glossy | surface/volume representation rebuilt per frame |

Keep five concepts distinct: **surface representation**, **radiance caching**, **learned parameter ownership**, **transport invalidation**, **sampling/scheduling**. These papers mainly contribute caching, invalidation and scheduling ideas; none defines learned-parameter ownership, which is this project's representation question.

Also prior art (not the starting point): hierarchical/line-space dependency structures (Drettakis & Sillion 1997), Luksch et al. 2019, Enlighten workflows; neural per-object composition (NeLT 2023, Superposed Deformable Feature Fields 2024). Do **not** claim novelty for "a hierarchy", "update affected parts only", or "per-object/per-region neural representations" alone.

## 6. Established Evidence and Scope

- Worklogs 21–25: a changed relative placement within the canonical teaset produces nonlocal indirect transport changes at stationary surfaces; frozen 8DNA and RNA do not track it; indirect-path occlusion and moved-part reflection dominate.
- Worklogs 26–27: physical and historical whole-asset neural recomputation are expensive (different regimes).
- Worklogs 28–29: two compact relational-state candidates fail held-out near-mirror T3.
- Worklog 30: intra-object triplane-cell masking in one 8DNA model (shared flows frozen) fails — a valid negative result **for that feature-level hypothesis only**.
- Worklog 31: explicit (non-neural) per-part surface radiance units with no shared decoder compose to 8DNA-level T0 quality and, refitted, recover the T3 change (tracking gain ≈ 1); at object granularity D_T = all objects; no wall-clock saving in that implementation.
- Worklog 32 (Batch 1): RNA split into K complete RNA units (P1/P4/P16, matched and full-size). Parameter and output ownership hold exactly (REINIT: 15 frozen units bit-identical, every frozen-unit pixel bit-identical, re-initialised unit recovers its P16 error). Static fidelity of P16 is within ~0.2 dB / 1.03× MAE of original RNA, with a measurable excess discontinuity at intra-part seams. **But the predeclared static-credibility gate (original RNA ≥ 25 dB) failed in both configurations** (C1 17.97 dB; C2 24.92 dB), so the ownership-fidelity verdict is not adjudicated.

## 7. Immediate Scientific Frontier

Representation feasibility first (decision 10). Open questions, in order:
1. a credible static baseline under a predeclared protocol (worklog 32's stop condition);
2. unit-level reconstruction after a **physically grounded** geometry edit (radiometric validity, not just ownership semantics);
3. whether D_T is sparse at unit granularity on a scene where that is physically plausible;
4. seam/composition quality and capacity cost of ownership;
5. only then dependency discovery, hierarchy, efficiency and scaling.

## 8. Research Conduct / Stop Conditions

- Preserve historical scripts, worklogs 21–32 and archived architecture snapshots; never reinterpret worklog 30 as a test of unit-level or object-level ownership.
- Predeclare scenes, quality gates and thresholds; never move them after seeing results.
- A stop condition ends the current experimental route, not the research direction; attribute a negative result (baseline limitation, candidate limitation, ownership isolation, optimisation/capacity, partition, supervision/correspondence, compositing) before interpreting it.
- No automatic advance to the next batch; each batch needs a user decision.

## 9. One-Line Research Definition

> **Keep high-quality learned GI valid under geometry edits by owning learned transport in independently reconstructible surface units, recomputing the complete state of only the units whose transport became invalid, and reusing every unit that remains valid.**
