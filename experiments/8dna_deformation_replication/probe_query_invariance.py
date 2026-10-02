"""Diagnostic probe (worklog 25): do the learned components' runtime inputs change
between T0 and T3 at the stationary interaction ROI?

Read-only. Uses the worklog-22 RNA feature buffers (rna_teaset/features/T0.npz,
T3.npz; common-light regime, 16 sub-pixel samples per pixel, 16 area-light
samples per hit) and the accepted interaction ROI. Those buffers compute the
canonical position exactly as the 8DNA 'attached' adapter does (subtract the
hit part's translation; teaset_parts.PartRigidAsset / rna_bridge.features), so
the same comparison answers both backbones' query-input question:

  RNA canonical mode : position (training-AABB normalised), normal, camera_dir,
                       light_dir (constant), per-light-sample direct visibility.
  RNA current mode   : as above but position normalised by the *current* scene
                       AABB (official renderer behaviour).
  8DNA attached mode : xi = x1 - t_part, wi = camera direction, envelope =
                       canonical bbox + t_part, all in the asset frame.

No production code is changed; nothing is rendered.
"""

from __future__ import annotations

import json

import numpy as np

import ednalib as L


def main() -> int:
    L.init_upstream()  # environment_record reads the Mitsuba version
    R = L.RESULTS
    f0 = np.load(R / "rna_teaset/features/T0.npz")
    f3 = np.load(R / "rna_teaset/features/T3.npz")
    roi = np.load(R / "gt_design/common_light/rois_T3.npz")
    pix = np.flatnonzero(roi["mask_interaction"].reshape(-1))
    k = int(f0["samples"])
    lanes = (pix[:, None] * k + np.arange(k)[None]).reshape(-1)
    names = [str(n) for n in f0["part_names"]]
    rec = {"roi_pixels": int(len(pix)), "lanes": int(len(lanes))}

    for z, tag in ((f0, "T0"), (f3, "T3")):
        hit = z["hit"][lanes].astype(bool)
        rec[f"{tag}_hit_fraction"] = float(hit.mean())
        parts = z["part"][lanes][hit]
        rec[f"{tag}_first_hit_parts"] = {names[i]: float(np.mean(parts == i)) for i in range(len(names))}
    both = f0["hit"][lanes].astype(bool) & f3["hit"][lanes].astype(bool)
    ln = lanes[both]
    same_part = f0["part"][ln] == f3["part"][ln]
    rec["same_first_hit_part"] = float(same_part.mean())

    def diff(key):
        a, b = f0[key][ln].astype(np.float64), f3[key][ln].astype(np.float64)
        d = np.abs(a - b).max(-1)
        return {"max_abs": float(d.max()), "fraction_changed": float(np.mean(d > 0))}

    for key in ("canonical_position", "position", "normal", "camera_dir"):
        rec[f"delta_{key}"] = diff(key)

    # per-light-sample direct visibility (stored for hits in hit order)
    def vis(z, lanes_):
        order = np.flatnonzero(z["hit"].astype(bool))
        pos = np.searchsorted(order, lanes_)
        return z["light_vis"][pos]

    v0, v3 = vis(f0, ln), vis(f3, ln)
    rec["direct_visibility"] = {"lit_fraction_T0": float(v0.mean()), "lit_fraction_T3": float(v3.mean()),
                                "samples_changed": float(np.mean(v0 != v3)),
                                "lanes_with_any_change": float(np.mean((v0 != v3).any(1)))}

    # current-mode normalisation (official RNA renderer): AABB of the current scene
    lo0, hi0 = f0["current_aabb_min"], f0["current_aabb_max"]
    lo3, hi3 = f3["current_aabb_min"], f3["current_aabb_max"]
    p = f0["position"][ln].astype(np.float64)
    n0 = (p - lo0) / (hi0 - lo0)
    n3 = (p - lo3) / (hi3 - lo3)
    rec["current_mode_aabb"] = {"T0": [lo0.tolist(), hi0.tolist()], "T3": [lo3.tolist(), hi3.tolist()],
                                "normalised_coordinate_shift_max": float(np.abs(n0 - n3).max()),
                                "shift_in_triplane_texels_512": float(np.abs(n0 - n3).max() * 512)}
    rec["verdict"] = {
        "query_geometry_identical": rec["delta_canonical_position"]["max_abs"] == 0.0
        and rec["delta_normal"]["max_abs"] == 0.0 and rec["delta_camera_dir"]["max_abs"] == 0.0
        and rec["same_first_hit_part"] == 1.0,
        "only_changing_runtime_input_canonical": "per-light-sample direct visibility",
    }
    out = R / "code_audit" / "probe_query_invariance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    rec["environment"] = L.environment_record()
    L.write_json(out, rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "environment"}, indent=1))
    return 0


if __name__ == "__main__":
    import os
    import sys

    rc = main()
    sys.stdout.flush()
    os._exit(rc)  # skip the DrJit+torch teardown crash (outputs are written)
