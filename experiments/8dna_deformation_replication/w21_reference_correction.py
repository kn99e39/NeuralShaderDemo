"""Worklog-21 reference correction.

Worklog 21's path-traced references were rendered with DrJit LoopRecord on,
which on this GPU loses 1-5% of the radiance and does not reproduce
(ednalib.render_reference).  This script re-renders the locked references in
wavefront mode with the locked seeds and chunking, and re-evaluates the
unchanged historical 8DNA renders against them.  Worklog-21 outputs are read,
never written; results go to results/8dna_replication/<out>/.

1. regression : evaluate() on the historical references reproduces the
                historical frozen_eval.json exactly;
2. corrected  : references re-rendered; metrics, physical-signal gate and the
                locked worklog-21 decision rule recomputed.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import ednalib as L
import teaset_parts as T
from teaset_frozen_eval import evaluate, load_rois


CORRECTED_INTEGRATOR = {"type": "path", "max_depth": -1, "rr_depth": 5}
CORRECTED_CHUNK = 32


def interaction_rule(ev: dict, mode: str = "attached") -> dict:
    """Worklog-21 locked decision rule on the interaction ROI."""
    st = ev["states"]
    row = {}
    for s in st:
        r = st[s]["modes"][mode]["interaction"]
        e = r["model_error_8DNA_vs_GT"]["display_mae"]
        e0 = r["model_error_at_T0_same_surface"]["display_mae"]
        row[s] = {"error": e, "error_T0": e0, "rise": e / e0 - 1, "gain": r["response_linear"]["gain_projection"],
                  "far_rise": st[s]["modes"][mode]["far"]["model_error_8DNA_vs_GT"]["display_mae"]
                  / st[s]["modes"][mode]["far"]["model_error_at_T0_same_surface"]["display_mae"] - 1}
    fail = [s for s in ("T1b", "T2", "T3") if row[s]["rise"] >= 0.25 and (row[s]["gain"] or 0) < 0.5]
    control = row["T1"]["rise"] < 0.25
    verdict = ("CROSS-MODEL GEOMETRY-CONFIGURATION FAILURE OBSERVED" if fail and control
               else "NO MEANINGFUL FAILURE UNDER TESTED REGIME" if not fail else "FAILURE WITH GENERIC-MOVEMENT CONFOUND")
    return {"per_state": row, "failing_states": fail, "control_T1_ok": control, "classification": verdict}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="protocol/teaset_frozen_locked.json")
    ap.add_argument("--out", default="w21_reference_correction")
    args = ap.parse_args()
    proto = json.loads(open(L.EXPERIMENT / args.protocol, encoding="utf-8").read())
    mi, dr = L.init_upstream()
    hist = L.RESULTS / proto["frozen_output"]
    out = L.RESULTS / args.out
    states = proto["states"]
    rois = load_rois(L.RESULTS / proto["gt_output"], states)
    neural = {(s, m): L.load_exr(hist / s / f"8dna_{m}.exr") for s in states for m in proto["modes"][s]}
    neural_b = L.load_exr(hist / "T0" / "8dna_upstream_seedB.exr")
    rec = {"protocol": proto, "environment": L.environment_record(),
           "corrected_reference": {"integrator": CORRECTED_INTEGRATOR, "chunk": CORRECTED_CHUNK, "mode": "wavefront (LoopRecord off)"},
           "historical_neural_sha256": {f"{s}/{m}": L.sha256(hist / s / f"8dna_{m}.exr") for s, m in neural}}

    # 1. regression against the historical evaluation
    old_gt = {s: (L.load_exr(hist / s / "gt_A.exr"), L.load_exr(hist / s / "gt_B.exr")) for s in states}
    old = json.loads(open(hist / "frozen_eval.json", encoding="utf-8").read())
    redo = evaluate(proto, old_gt, neural, neural_b, rois)
    diffs = [abs(redo["states"][s]["modes"][m][k]["model_error_8DNA_vs_GT"]["display_mae"]
                 - old["states"][s]["modes"][m][k]["model_error_8DNA_vs_GT"]["display_mae"])
             for s in states for m in proto["modes"][s] for k in ("interaction", "tray", "far", "mover")]
    rec["regression_max_abs_diff"] = max(diffs)
    rec["historical_rule"] = interaction_rule(old)

    # 2. corrected references: wavefront, locked spp and seeds; C++ path instead of prb (identical means in
    #    wavefront mode, probe_loop_record.py; wavefront prb is ~35x slower here), chunk 32 to bound memory
    gt = {}
    for s, tr in states.items():
        scene = mi.load_dict(T.scene_dict(proto["res"], tr, None, CORRECTED_INTEGRATOR))
        a, ta = L.render_reference(scene, proto["gt"]["spp"], CORRECTED_CHUNK, proto["gt"]["seed_A"])
        b, _ = L.render_reference(scene, proto["gt"]["spp"], CORRECTED_CHUNK, proto["gt"]["seed_B"])
        gt[s] = (a, b)
        L.save_exr(out / s / "gt_A.exr", a)
        L.save_exr(out / s / "gt_B.exr", b)
        rec.setdefault("reference_seconds", {})[s] = ta
        rec.setdefault("historical_vs_corrected_mean_ratio", {})[s] = float(old_gt[s][0].mean() / a.mean())
        print(s, f"reference {ta:.0f}s", flush=True)
    ev = evaluate(proto, gt, neural, neural_b, rois)
    rec["corrected"] = ev
    rec["corrected_rule"] = interaction_rule(ev)

    # physical-signal gate on the corrected references (worklog-21 definition: seed set A vs T0, A vs B noise)
    flat = lambda img: img.reshape(-1, 3)
    gate = {}
    for s in states:
        m, i0 = rois[s]["interaction"]
        chg = np.abs(L.tonemap(flat(gt[s][0])[m]) - L.tonemap(flat(gt["T0"][0])[i0])).mean()
        noise = np.abs(L.tonemap(flat(gt[s][0])[m]) - L.tonemap(flat(gt[s][1])[m])).mean()
        gate[s] = {"interaction_change": float(chg), "repeat_noise": float(noise), "ratio": float(chg / noise)}
    rec["corrected_physical_gate"] = gate
    L.write_json(out / "w21_reference_correction.json", rec)
    print("regression", rec["regression_max_abs_diff"])
    print("historical", rec["historical_rule"]["classification"], "corrected", rec["corrected_rule"]["classification"])
    print(json.dumps({s: {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()} for s, r in rec["corrected_rule"]["per_state"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
