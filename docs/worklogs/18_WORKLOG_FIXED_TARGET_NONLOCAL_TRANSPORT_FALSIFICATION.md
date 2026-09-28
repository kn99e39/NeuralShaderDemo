# Worklog — Fixed-Target Nonlocal-Transport Falsification (2026-09-28)

## Implementation fact

This batch preserved historical F0--F3 and R0--R4 evidence and used a separate N0--N4 Rain trajectory on LabServer63 native Linux (RTX 3080 Ti). The target was the `GEO-rain_scarf` patch with `DEF-Scarf1` weight at least 0.99; only descendant `FK-Scarf3` moved. N0--N4 were declared before frozen-RNA evaluation at 0, 90, 135, 165, and 180 degrees local X.

The requested triangle color AOV could not be emitted by the server BPy wheel because it lacks writable CORNER-domain color attributes. The final correspondence used the permitted stable-UV alternative: `GEO-rain_scarf` provenance plus `UVMap`, with 94 target source polygons and no duplicate target-internal UV polygon coordinate set. It is not nearest-canonical XYZ correspondence.

GT-only Cycles/AOV renders recorded beauty, UV, canonical position, provenance, normal, and diffuse-direct for N0, an N0 repeat, and N1--N4. The unchanged official Adobe RNA source and frozen Rain checkpoint were then evaluated on these locked states.

## Measurement

Target position maximum delta was zero for every state. Target-to-moving-neighbor minimum distance declined from 0.117545 at N0 to 0.096016, 0.062965, 0.030869, and 0.024644 at N1--N4. N1 moved the distal segment by mean 0.075893 world units; N4 moved it by 0.107328.

Reference repeatability was RGB MAE 5.243e-06 and luminance MAE 4.895e-06. Exact UV/provenance target pixels with unchanged diffuse-direct branch numbered 2,904 for GT and 1,125--1,126 for frozen RNA; direct-visibility branch flips were zero throughout. Target GT MAE relative to N0 was zero at N2--N4 and 3.977e-06 at N1. Frozen-RNA delta MAE was zero at every state.

## Observation

The trajectory created substantial distal-neighbor motion and monotonic approach while keeping the target fixed, but did not induce a target radiance change above reference noise. Full-frame GT changed; it is not target-patch evidence and was not used for the decision.

## Interpretation

**PHYSICAL EFFECT TOO WEAK.** This trajectory does not stress fixed-target nonlocal transport above render noise, so frozen-RNA behavior is not interpreted as transport failure or success. No refit, Einar work, capacity change, lighting tuning, or new architecture was performed.

Worklog 17's `FEATURE-CONTAMINATION / SPATIAL-REPRESENTATION-LIMITED` interpretation remains **unresolved**. This batch neither supports nor refutes it because the required fixed-target physical signal was absent; its historical nearest-canonical diagnostics remain diagnostic only.