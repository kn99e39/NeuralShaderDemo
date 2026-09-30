"""Cross-backbone evaluation and refit controls (protocol/teaset_cross_backbone_locked.json).

Common-light regime: RNA and 8DNA are scored against one reference (the
references re-rendered by the locked 8DNA common-light run) with the same
paired metrics (teaset_frozen_eval.evaluate).  Each model is judged against
its own T0 by the locked decision rule; raw errors are never ranked across
models.  Refits: 8DNA T0-retrain/T3-refit in both regimes (worklog-21 envmap
with the corrected references, and common light); RNA T3 refit in common light.
"""

from __future__ import annotations

import json

import numpy as np

import ednalib as L
from teaset_frozen_eval import evaluate, load_rois

RELATION_STATES = ("T1b", "T2", "T3")


def rule(ev: dict, label: str, mode: str) -> dict:
    e_key = f"model_error_{label}_vs_GT"
    row = {}
    for s, st in ev["states"].items():
        r = st["modes"][mode]
        per = {}
        for k in ("interaction", "tray", "far", "mover"):
            e, e0 = r[k][e_key]["display_mae"], r[k]["model_error_at_T0_same_surface"]["display_mae"]
            per[k] = {"error": e, "error_T0": e0, "rise": e / e0 - 1, "gain": r[k]["response_linear"]["gain_projection"],
                      "dG": r[k]["physical_change_GT_vs_GT0"]["display_mae"],
                      "dN": r[k][f"model_change_{label}_vs_{label}0"]["display_mae"]}
        row[s] = per
    # noise precondition (protocol decision_rule.noise_precondition): a state is
    # interpreted only if its interaction dG exceeds 3x the model's own T0 seed
    # repeat and 3x the reference repeat in that ROI.
    noise = ev["noise"]["interaction"]
    ref_noise = noise["gt_repeat_A_vs_B"]["display_mae"]
    model_noise = noise.get("neural_seed_repeat_T0", {}).get("display_mae")
    for s in row:
        dg = row[s]["interaction"]["dG"]
        row[s]["interaction"]["noise_precondition"] = {
            "dG_over_reference_noise": dg / ref_noise,
            "dG_over_model_noise": None if model_noise is None else dg / model_noise,
            "pass": model_noise is not None and dg > 3 * model_noise and dg > 3 * ref_noise}
    meets = [s for s in RELATION_STATES if row[s]["interaction"]["rise"] >= 0.25 and (row[s]["interaction"]["gain"] or 0) < 0.5]
    fail = [s for s in meets if row[s]["interaction"]["noise_precondition"]["pass"]]
    supporting = [s for s in meets if s not in fail]
    interpretable = [s for s in RELATION_STATES if row[s]["interaction"]["noise_precondition"]["pass"]]
    control = row["T1"]["interaction"]["rise"] < 0.25
    if not interpretable:
        verdict = "NOT INTERPRETABLE (no relation state above the noise floors)"
    else:
        verdict = ("FROZEN FAILURE" if fail and control else "NO MEANINGFUL FAILURE" if not fail else "FAILURE, CONTROL CONFOUNDED")
    return {"mode": mode, "per_state": row, "noise_floors": {"reference": ref_noise, "model": model_noise},
            "interpretable_states": interpretable, "failing_states": fail,
            "meets_failure_criterion_below_noise_precondition": supporting,
            "control_T1_ok": control, "classification": verdict}


def paired_error(img, gt_img, m, idx=None):
    flat = lambda a: a.reshape(-1, 3)
    a, g = flat(img), flat(gt_img)
    return L.pixel_metrics(a[m] if idx is None else a[idx], g[m] if idx is None else g[idx])


def refit_block(gt, rois, n_t0, n_t3, released_t0_err=None) -> dict:
    """Recovery rule on the interaction ROI: refit(T3) vs same-pipeline T0 model."""
    flat = lambda a: a.reshape(-1, 3)
    g0 = flat(0.5 * (gt["T0"][0] + gt["T0"][1]))
    g3 = flat(0.5 * (gt["T3"][0] + gt["T3"][1]))
    m, i0 = rois["T3"]["interaction"]
    m0 = rois["T0"]["interaction"][0]
    e3 = L.pixel_metrics(flat(n_t3)[m], g3[m])["display_mae"]
    e0_same = L.pixel_metrics(flat(n_t0)[i0], g0[i0])["display_mae"]
    e0_roi = L.pixel_metrics(flat(n_t0)[m0], g0[m0])["display_mae"]
    dn, dg = (flat(n_t3)[m] - flat(n_t0)[i0]).reshape(-1), (g3[m] - g0[i0]).reshape(-1)
    gain = float(dn @ dg / (dg @ dg))
    out = {"refit_T3_interaction_error": e3, "T0_model_error_same_surface": e0_same, "T0_model_error_roi": e0_roi,
           "error_ratio": e3 / e0_same, "gain": gain,
           "recovers": bool(e3 <= 1.25 * e0_same and gain >= 0.5)}
    if released_t0_err is not None:
        out["released_T0_error_roi"] = released_t0_err
        out["pipeline_control_ok"] = bool(abs(e0_roi / released_t0_err - 1) <= 0.25)
    return out


def main() -> int:
    proto = json.loads(open(L.EXPERIMENT / "protocol/teaset_cross_backbone_locked.json", encoding="utf-8").read())
    w21 = json.loads(open(L.EXPERIMENT / "protocol/teaset_frozen_locked.json", encoding="utf-8").read())
    L.init_upstream()
    states = proto["states"]
    fdir = L.RESULTS / proto["frozen_output"]
    rdir = L.RESULTS / proto["rna_output"]
    rois = load_rois(L.RESULTS / proto["gt_output"], states)
    gt = {s: (L.load_exr(fdir / s / "gt_A.exr"), L.load_exr(fdir / s / "gt_B.exr")) for s in states}
    rec = {"protocol": proto, "environment": L.environment_record()}

    # 8DNA (common light): metrics from the locked frozen run
    ev8 = json.loads(open(fdir / "frozen_eval.json", encoding="utf-8").read())
    rec["8dna"] = {"rule_attached": rule(ev8, "8DNA", "attached"), "rule_fixed": rule(ev8, "8DNA", "fixed"),
                   "rule_upstream": rule(ev8, "8DNA", "upstream")}

    # RNA (common light)
    rna_modes = ("canonical", "current")
    rna = {(s, m): np.load(rdir / f"{s}_{m}.npy").astype(np.float32) for s in states for m in rna_modes}
    rna_b = np.load(rdir / "T0_canonical_B.npy").astype(np.float32)  # independent light samples: RNA's own noise
    proto_rna = dict(proto, modes={s: list(rna_modes) for s in states})
    evr = evaluate(proto_rna, gt, rna, rna_b, rois, model_label="RNA")
    rec["rna_eval"] = evr
    rec["rna"] = {"rule_canonical": rule(evr, "RNA", "canonical"), "rule_current": rule(evr, "RNA", "current")}

    # static gates (G1): model T0 interaction error vs the reference's T3 interaction change
    gate_ref = json.loads(open(L.RESULTS / proto["gt_output"] / "gt_states.json", encoding="utf-8").read())
    signal = gate_ref["states"]["T3"]["regions"]["interaction"]["gt_change_display_mae_A"]
    e_rna0 = evr["states"]["T0"]["modes"]["canonical"]["interaction"]["model_error_RNA_vs_GT"]["display_mae"]
    e_8d0 = ev8["states"]["T0"]["modes"]["attached"]["interaction"]["model_error_8DNA_vs_GT"]["display_mae"]
    rec["static_gate_G1"] = {"reference_T3_interaction_change": signal,
                             "rna_T0_interaction_error": e_rna0, "rna_pass": e_rna0 < signal,
                             "8dna_T0_interaction_error": e_8d0, "8dna_pass": e_8d0 < signal}

    # refits
    rr = L.RESULTS / "refit" / "renders"
    rec["refit_8dna_common_light"] = refit_block(
        gt, rois, L.load_exr(rr / "common_light" / "T0_retrain_at_T0.exr"), L.load_exr(rr / "common_light" / "T3_refit_at_T3.exr"),
        released_t0_err=ev8["states"]["T0"]["modes"]["upstream"]["interaction"]["model_error_8DNA_vs_GT"]["display_mae"])
    cdir = L.RESULTS / "w21_reference_correction"
    corr = json.loads(open(cdir / "w21_reference_correction.json", encoding="utf-8").read())
    gt_w = {s: (L.load_exr(cdir / s / "gt_A.exr"), L.load_exr(cdir / s / "gt_B.exr")) for s in w21["states"]}
    rois_w = load_rois(L.RESULTS / w21["gt_output"], w21["states"])
    rec["refit_8dna_w21_envmap_corrected"] = refit_block(
        gt_w, rois_w, L.load_exr(rr / "w21_envmap" / "T0_retrain_at_T0.exr"), L.load_exr(rr / "w21_envmap" / "T3_refit_at_T3.exr"),
        released_t0_err=corr["corrected"]["states"]["T0"]["modes"]["upstream"]["interaction"]["model_error_8DNA_vs_GT"]["display_mae"])
    rna_refit = rdir / "refit_T3_current.npy"
    if rna_refit.exists():
        rec["refit_rna_common_light"] = refit_block(gt, rois, rna[("T0", "canonical")], np.load(rna_refit).astype(np.float32))
    else:
        rec["refit_rna_common_light"] = "not run"

    L.write_json(L.RESULTS / "cross_backbone" / "cross_backbone.json", rec)
    print(json.dumps({"8dna": rec["8dna"]["rule_attached"]["classification"], "rna": rec["rna"]["rule_canonical"]["classification"],
                      "gate": rec["static_gate_G1"]}, indent=1))
    for s in states:
        a = rec["8dna"]["rule_attached"]["per_state"][s]["interaction"]
        b = rec["rna"]["rule_canonical"]["per_state"][s]["interaction"]
        print(f"{s:4s} dG {a['dG']:.4f} | 8DNA rise {a['rise']:+.0%} gain {a['gain'] if a['gain'] is None else round(a['gain'], 2)} | "
              f"RNA rise {b['rise']:+.0%} gain {b['gain'] if b['gain'] is None else round(b['gain'], 2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
