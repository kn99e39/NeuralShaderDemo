# Worklog — Rain Attribution Coverage and Runtime Block (2026-09-27)

## Implementation fact

The historical F0--F3 files remain untouched.  A separate canonical Rain
scene copy adds a stable surface-provenance AOV (scarf red, top green), while
the established canonical-position AOV remains in place.  Separate
provenance-instrumented F0/F1 diagnostic states and an independently locked
R0--R4 production-rig trajectory were created under ignored results paths.

The new trajectory uses only `RIG-rain` local-X FK controls.  R1 is the local
bend control (`FK-Scarf1=-45`, `FK-Scarf2=0`, `FK-Scarf3=60` degrees); R4
holds the first and third controls and changes only `FK-Scarf2` to `90`
degrees.  Before any RNA render, the `DEF-Scarf1`--`DEF-Scarf3` geometry
descriptor was 0.16 at R1 and 0.12 at R4 (cross-segment centroid distance),
with source/evaluated topology preserved at 1,285 vertices and 1,217 polygons.
This locks a lower-interaction local control and a stronger-interaction state
without selecting from frozen-RNA error.

`state_contract.json` now records the actual historical FK-Scarf2/FK-Scarf3
implementation and the R trajectory; no unmeasured fold-depth values remain.
`cloth.json` now records the completed static gate and validation-best Rain
checkpoint.  Its manifest validator passes.

## Observation

The F1 cyan artifact is a real high-error region in the historical render.
The historical F0 direct-visibility buffer was not retained.  A new
provenance-instrumented F0 diagnostic render was attempted twice in a fresh
output directory, including after a clean `wsl --shutdown`; both attempts
ended before a trace/output completion marker and restarted WSL.  No
historical render was overwritten.

## Measurement

The preserved F1 reference, frozen RNA output, canonical-position AOV, and
200-view static training H5 were analysed offline.  The deterministic cyan
selector (cyan RNA chroma plus valid-pixel error at or above p95) selected
1,198 pixels.  Canonical nearest-surface mapping classified 950 (79.3%) as
scarf and 248 (20.7%) as top.  At a 0.005 world-unit support radius, every
selected point had at least one canonical training view; the median support
was 37 views and mean support was 63.68 views.  Median nearest training-sample
distance was 0.00177 and p95 was 0.00347.

The coverage visualization and its JSON are in
`results/batch1_hq_dynamic_failure/cloth/attribution/legacy_coverage/` and
the PNG is staged in `results/evaluation/` with provenance hash.

## Interpretation

The dominant historical F1 cyan artifact is **not coverage-limited** under
the measured canonical-support criterion: it is predominantly scarf rather
than newly exposed top, and it is observed across many canonical training
views.  Its current narrower classification is
**FEATURE-CONTAMINATION / SPATIAL-REPRESENTATION-LIMITED**, pending the new
trajectory's transport comparison.  It must not be called stale transport:
the required F0--F1 direct-visibility delta was not measurable because F0's
buffer was absent and the separate diagnostic render is runtime-blocked.

## Remaining limit and hypothesis status

The locked R0--R4 no-disocclusion trajectory has not produced GT/Frozen RNA
renders, transport passes, or ROI metrics.  The WSL DXG GPU bridge again
restarted on every frozen diagnostic launch, so this batch cannot yet test
whether error follows interaction rather than local deformation.  The central
hypothesis is therefore **NOT ATTRIBUTABLE** at this point, not supported or
weakened by the new trajectory.  No refit and no Einar work was attempted.
