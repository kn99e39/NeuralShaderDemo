"""Focused tests of the reference-only transport decomposition (worklog 24).

1. partition   : per sample, the class contributions sum to the plain total
                 (float32 accumulation tolerance).
2. estimator   : on a pixel lattice covering the whole image (every 8th pixel
                 in x and y), the diagnostic total matches Mitsuba's C++ path
                 on the same scene in both regimes: image mean within 1%, and
                 display MAE to path seed A no larger than 1.5x path's own
                 seed-A-vs-B MAE.
3. part identity: in every state each part is found at its protocol position
                 (teaset_parts.part_shape_ids probes the part's own translated
                 vertices); the ids are distinct; probing the T3 scene with T0
                 positions fails for the mover (negative control).
4. state identity: the decomposition's scene for each state is built from the
                 locked protocol translations (equality check), and >= 99% of
                 the interaction ROI's first hits are on teapot4 in every state.
Repeat noise is measured by the analysis itself (seeds A and B).
"""

from __future__ import annotations

import json

import numpy as np

import ednalib as L
import teaset_parts as T
import transport_decomposition as D


def main() -> int:
    mi, dr = L.init_upstream()
    rec = {}
    locked = json.loads(open(L.EXPERIMENT / "protocol/teaset_frozen_locked.json", encoding="utf-8").read())

    # 1 + 2. partition and estimator vs C++ path
    rec["estimator"] = {}
    yy, xx = np.meshgrid(np.arange(4, 512, 8), np.arange(4, 512, 8), indexing="ij")
    pix = (yy * 512 + xx).reshape(-1)
    for regime in ("common_light", "w21_envmap"):
        scene, tr, ids, _, _ = D.regime_setup(regime, "T3")
        spp = 1024
        r = D.render_pixels(regime, "T3", pix, spp, 9001)
        path = mi.load_dict({"type": "path", "max_depth": -1, "rr_depth": 5})
        # chunked like the canonical references: one 512^2 x 1024 spp wavefront
        # launch would exhaust GPU memory
        pa = np.asarray(L.render_chunked(scene, path, spp, 32, 11)[0], np.float32).reshape(-1, 3)[pix]
        pb = np.asarray(L.render_chunked(scene, path, spp, 32, 12)[0], np.float32).reshape(-1, 3)[pix]
        mae_d = float(np.abs(L.tonemap(r["total"]) - L.tonemap(pa)).mean())
        mae_p = float(np.abs(L.tonemap(pb) - L.tonemap(pa)).mean())
        ratio = float(r["total"].mean() / (0.5 * (pa + pb)).mean())
        rec["estimator"][regime] = {"partition_max_abs_err": r["per_sample_partition_max_abs_err"],
                                    "mean_ratio_diag_over_path": ratio, "display_mae_diag_vs_pathA": mae_d,
                                    "display_mae_pathB_vs_pathA": mae_p,
                                    "pass": r["per_sample_partition_max_abs_err"] < 1e-3 and abs(ratio - 1) < 0.01 and mae_d <= 1.5 * mae_p}

    # 3 + 4. part and state identity
    rec["identity"] = {}
    for regime in ("common_light", "w21_envmap"):
        for s in locked["states"]:
            scene, tr, ids, roi, _ = D.regime_setup(regime, s)
            pixr = np.flatnonzero(np.load(roi)["mask_interaction"].reshape(-1))
            sensor = scene.sensors()[0]
            si = scene.ray_intersect(D.pixel_rays(sensor, 512, pixr, np.full((len(pixr), 2), 0.5)))
            on_target = float(np.mean(np.array(T.shape_ids(si)) == ids["teapot4"]))
            rec["identity"][f"{regime}/{s}"] = {"ids": ids, "distinct": len(set(ids.values())) == 4,
                                                "translations_match_protocol": tr == locked["states"][s],
                                                "roi_first_hit_on_teapot4": on_target,
                                                "pass": len(set(ids.values())) == 4 and tr == locked["states"][s] and on_target >= 0.99}
    scene3 = D.regime_setup("common_light", "T3")[0]
    try:
        T.part_shape_ids(scene3, {})
        neg = False
    except RuntimeError as exc:
        neg = "teapot2" in str(exc) or "no probe ray" in str(exc) or "ambiguous" in str(exc)
    rec["identity"]["negative_control_T3_probed_with_T0"] = {"probe_fails": neg, "pass": neg}

    rec["pass"] = all(v["pass"] for grp in ("estimator", "identity") for v in rec[grp].values())
    out = L.RESULTS / D.OUT / "tests.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    L.write_json(out, rec)
    print(json.dumps(rec, indent=1, default=str))
    return 0 if rec["pass"] else 1


if __name__ == "__main__":
    import os
    import sys

    rc = main()
    sys.stdout.flush()
    os._exit(rc)
