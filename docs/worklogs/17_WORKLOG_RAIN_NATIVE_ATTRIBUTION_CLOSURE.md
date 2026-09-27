# Worklog — Rain Native Attribution Closure (2026-09-27)

## Implementation fact

The Rain attribution run moved from the unstable local WSL DXG path to
`LabServer63` (RTX 3080 Ti) without editing the Adobe RNA source, which stayed
at commit `66b5b098e2013db32214e44a439a0cd66d331e4d`.  The server used a
separate Python 3.10 environment, BPy 3.5.1, Torch 2.8.0+cu128, and the
lockfile-compatible Lightning 2.5.5.  The local static checkpoint, H5, camera
contract, and state files were SHA-256 checked after transfer.

The historical F0/F1 states and the earlier R states remain untouched.  Their
new native diagnostics store current position/normal/camera, canonical
position, stable scarf/top provenance, and direct-light AOVs separately.

GT-only validation found that the original R1--R4 path did not produce a
useful interaction progression.  Before any frozen-RNA R render, it was
redesigned exactly once into opposing `FK-Scarf2`/`FK-Scarf3` local-X bends:
R0 `(0,0)`, R1 `(45,-45)`, R2 `(90,-90)`, R3 `(135,-135)`, R4 `(180,-180)`.
The redesigned states live separately under `locked_states_redesign1`.

## Measurement

The native F1 cyan selector chose 1,197 pixels.  Its same-surface training
support was 96.74% at radius 0.005 (median 36 supporting views); this matches
the previously measured historical result.  The local-input novelty diagnostic
also reproduced: artifact normal and camera-direction nearest-support medians
were 155.02 and 109.39 degrees, versus 0.22 and 1.52 degrees for the
same-surface low-error control.  This is a diagnostic comparison only.

For the F1 cyan region, AOV-constrained F1-to-F0 nearest-canonical matching
found 381 matches within radius 0.005 and 816 unmatched points.  Among the
matched set, direct visibility had 0 visible-to-occluded transitions (353
visible-to-visible, 27 occluded-to-visible, 1 occluded-to-occluded).

The redesigned R trajectory preserved the 1,285-vertex/1,217-polygon topology.
Its Scarf2/Scarf3 centroid distance declined from 0.09914 (R0) through
0.09547, 0.07855, 0.04795, to 0.01994 (R4); R4 minimum distance was 0.00060.
In the actual frozen-renderer direct AOV, matched scarf visible-to-occluded
counts were R1 11, R2 80, R3 460, R4 398.

The frozen checkpoint was unchanged and no refit was run.  Full-frame
reference/RNA PSNR for R0--R4 was 34.51, 31.55, 31.38, 31.26, and 25.89 dB;
MAE was 0.173, 0.213, 0.216, 0.218, and 0.326.  For R4, the 398 matched scarf
visible-to-occluded pixels had mean MAE 0.170, while an equal-count
same-surface visible-to-visible control had mean MAE 0.300 (ratio 0.568).

## Observation

The native renderer completed all F0/F1 and redesigned R0--R4 reference and
frozen-RNA outputs.  The historical cyan feature and its coverage/novelty
measurements were reproducible on the stable environment.  The redesigned R
path did create measurable direct-visibility transitions and a large R4
full-frame degradation, but its direct-transition subset was not the high-error
subset relative to the matched control.

Review artifacts are staged in `results/evaluation/`: the R0--R4 GT/RNA GIF,
R4 transition/control map, F0--F1 direct-visibility map, and native F1
same-surface coverage map.  Their hashes are recorded in the merged evaluation
manifest.

## Interpretation

The historical F1 cyan artifact is not coverage-limited under the measured
same-surface canonical-support criterion.  The native F0--F1 diagnostic does
not support a visible-to-occluded transport explanation for that artifact.
The redesigned R experiment establishes that stronger interaction can increase
overall frozen-RNA error, but the direct visibility-transition pixels themselves
were lower-error than the matched visible control.  Therefore a direct
visibility/stale-transport cause is **not supported** by this batch.

The defensible final classification is **FEATURE-CONTAMINATION /
SPATIAL-REPRESENTATION-LIMITED**, with the local-input novelty result as a
supporting diagnostic rather than a causal proof.  No claim is made that this
isolates a unique neural mechanism; nearest-canonical AOV matching is not
triangle/UV correspondence.  No Einar work or checkpoint refit was performed.
