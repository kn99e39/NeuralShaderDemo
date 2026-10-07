"""Evaluate worklog-29 branches (Windows) -- stage 4.

    windows/run.ps1 ../radiometric_relation_state_prototype/rrs_eval.py --run <run dir> [--worklog 29]

Branches: A = worklog-28 geometric relational (its stored predictions), B zero, C shuffled,
D real, and the exact-direction oracle if it was run.  Candidate image = historical frozen
RNA render + the branch's residual on the stable ROI pixels (worklog 28).

Primary metric (protocol "primary_metric"): on stable T3 pixels paired with T0,
    E_delta = mean |(tm(N_T3) - tm(N_T0)) - (tm(G_T3) - tm(G_T0))|,  R_delta = E_delta / mean |tm(G_T3) - tm(G_T0)|.
Secondary: absolute errors, historical refit_block (own and frozen T0), frozen-response rule.
Also: the predeclared success test (criteria 1-3 numeric; 4 is the reviewer's), the T1
control, state diagnostics of the proxy, cost accounting, and reviewer exports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[0]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXP / "relational_residual_prototype"))
sys.path.insert(0, str(EXP / "8dna_deformation_replication"))
sys.path.insert(0, str(EXP / "neural_recompute_cost"))

import numpy as np  # noqa: E402

import rrp_common as C28  # noqa: E402
import rrs_common as C  # noqa: E402
from rrp_eval import compose  # noqa: E402

STATES = ("T0", "T1", "T1b", "T2", "T3")


def paired(data, s):
    p0 = data["T0"]["roi_pixels"][data["T0"]["stable_pixels"]]
    ps = data[s]["roi_pixels"][data[s]["stable_pixels"]]
    return np.intersect1d(p0, ps, return_indices=True)


def change_metrics(L, data, pred, s):
    pix, i0, i1 = paired(data, s)
    tm = L.tonemap
    g0, gs = data["T0"]["gt_pixels"][i0], data[s]["gt_pixels"][i1]
    n0 = data["T0"]["frozen_pixels"][i0] + pred["T0"][i0]
    ns = data[s]["frozen_pixels"][i1] + pred[s][i1]
    dref = tm(gs) - tm(g0)
    e = np.abs((tm(ns) - tm(n0)) - dref).mean()
    el = np.abs((ns - n0) - (gs - g0)).mean()
    return {"pixels": int(len(pix)), "E_delta_display": float(e), "R_delta": float(e / np.abs(dref).mean()),
            "E_delta_linear": float(el), "R_delta_linear": float(el / np.abs(gs - g0).mean()),
            "ref_change_display": float(np.abs(dref).mean()),
            "static_T0_error_display": float(np.abs(tm(n0) - tm(g0)).mean()), "state_error_display": float(np.abs(tm(ns) - tm(gs)).mean())}


def rank(a):
    r = np.empty(len(a))
    r[np.argsort(a, kind="stable")] = np.arange(len(a))
    return r


def state_diagnostics(L, run, wl28, data, proto):
    names = C28.protocol()["parts_order"]
    mover = names.index(C28.protocol()["moved_part_for_reporting_only"])
    pr = {s: C.encode(np.load(run / "proxy" / f"{s}.npy")) for s in ("T0", "T1", "T1b", "T2", "T3")}
    pz = {s: np.load(wl28 / "probes" / f"{s}.npz") for s in pr}
    out = {}
    for s in pr:
        hit = pz[s]["hit"]
        lum = np.load(run / "proxy" / f"{s}.npy")[hit].mean(-1)
        mv = (pz[s]["remote_part"] == mover).sum(1)
        out[s] = {"probe_hit_fraction": float(hit.mean()),
                  "moved_part_hits_per_query": {"mean": float(mv.mean()), "p50": float(np.median(mv)), "p90": float(np.percentile(mv, 90)), "max": int(mv.max())},
                  "proxy_luminance_hits": {q: float(np.percentile(lum, v)) for q, v in (("p10", 10), ("p50", 50), ("p90", 90), ("p99", 99))} | {"mean": float(lum.mean())}}
    k = 16
    for s in ("T1b", "T2", "T3"):
        pix, i0, i1 = paired(data, s)
        q0 = (i0[:, None] * k + np.arange(k)).reshape(-1)
        q1 = (i1[:, None] * k + np.arange(k)).reshape(-1)
        a, b = pr["T0"][q0], pr[s][q1]
        ch = np.abs(a - b).max(-1) > 1e-6
        norm_q = np.sqrt(((a - b) ** 2).sum((1, 2)))
        norm_px = norm_q.reshape(-1, k).mean(1)
        dg = np.abs(L.tonemap(data[s]["gt_pixels"][i1]) - L.tonemap(data["T0"]["gt_pixels"][i0])).mean(-1)
        out[s]["vs_T0"] = {"probes_proxy_changed_fraction": float(ch.mean()), "query_change_norm_mean": float(norm_q.mean()),
                           "pearson_pixel_change_norm_vs_gt_change": float(np.corrcoef(norm_px, dg)[0, 1]),
                           "spearman_pixel_change_norm_vs_gt_change": float(np.corrcoef(rank(norm_px), rank(dg))[0, 1])}
    # T3 extrapolation: radiometric values vs geometry
    tr = [s for s in C.protocol()["split"]["train_states"]]
    hv = np.concatenate([pr[s][pz[s]["hit"]] for s in tr])
    lo, hi = hv.min(0), hv.max(0)
    v3 = pr["T3"][pz["T3"]["hit"]]
    out["T3_extrapolation"] = {"hit_proxy_values_outside_training_range_per_channel": ((v3 < lo) | (v3 > hi)).mean(0).tolist()}
    # per query (same queries): summed proxy and moved-part count vs each query's training-state range
    pix3 = pz["T3"]["pixels"][pz["T3"]["stable"]]
    sums, counts = {}, {}
    for s in tr + ["T3"]:
        ps = pz[s]["pixels"][pz[s]["stable"]]
        cm, i3, js = np.intersect1d(pix3, ps, return_indices=True)
        ssum = np.full((len(pix3), k), np.nan)
        cnt = np.full((len(pix3), k), np.nan)
        qs = (js[:, None] * k + np.arange(k)).reshape(-1)
        ssum[i3] = pr[s][qs].sum((1, 2)).reshape(-1, k)
        cnt[i3] = (pz[s]["remote_part"][qs] == mover).sum(1).reshape(-1, k)
        sums[s], counts[s] = ssum, cnt
    smin = np.nanmin(np.stack([sums[s] for s in tr]), 0)
    smax = np.nanmax(np.stack([sums[s] for s in tr]), 0)
    cmax = np.nanmax(np.stack([counts[s] for s in tr]), 0)
    ok = ~np.isnan(smin)
    rad_out = (sums["T3"] < smin) | (sums["T3"] > smax)
    geo_out = counts["T3"] > cmax
    out["T3_extrapolation"].update({
        "queries_summed_proxy_outside_training_range": float(rad_out[ok].mean()),
        "queries_moved_part_count_above_training_max": float(geo_out[ok].mean()),
        "both": float((rad_out & geo_out)[ok].mean()), "radiometric_only": float((rad_out & ~geo_out)[ok].mean()),
        "geometry_only": float((~rad_out & geo_out)[ok].mean())})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--worklog", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    run = Path(args.run).resolve()
    proto, p28 = C.protocol(), C28.protocol()
    import ednalib as L
    import nrc_records as R

    git = R.git_state(L.ROOT)
    R.require_clean(git, args.allow_dirty)
    L.init_upstream()
    from cross_backbone_eval import refit_block, rule
    from teaset_frozen_eval import evaluate, load_rois

    cb = json.loads((L.EXPERIMENT / p28["base_protocol"]).read_text(encoding="utf-8"))
    wl28 = L.ROOT / proto["worklog28_reference"]["run"]
    fdir, rdir = L.RESULTS / cb["frozen_output"], L.RESULTS / cb["rna_output"]
    rois = load_rois(L.RESULTS / cb["gt_output"], cb["states"])
    gt = {s: (L.load_exr(fdir / s / "gt_A.exr"), L.load_exr(fdir / s / "gt_B.exr")) for s in STATES}
    frozen = {s: np.load(rdir / f"{s}_canonical.npy").astype(np.float32) for s in STATES}
    frozen_b = np.load(rdir / "T0_canonical_B.npy").astype(np.float32)
    data = {s: dict(np.load(wl28 / "dataset" / f"{s}.npz")) for s in p28["evaluate_states"]}
    tr = json.loads((run / "models" / "training.json").read_text())
    orc_f = run / "models" / "oracle_training.json"
    preds = {}
    for seed in p28["training"]["seeds"]:
        preds[f"A_geometric_s{seed}"] = dict(np.load(wl28 / "models" / f"pred_relational_s{seed}.npz"))
        for v, lab in (("zero", "B_zero"), ("shuffled", "C_shuffled"), ("real", "D_real")):
            preds[f"{lab}_s{seed}"] = dict(np.load(run / "models" / f"pred_{v}_s{seed}.npz"))
        if orc_f.exists():
            preds[f"O_oracle_s{seed}"] = dict(np.load(run / "models" / f"pred_oracle_s{seed}.npz"))
    proto_m = dict(cb, modes={s: ["m"] for s in cb["states"]})
    rec = {"git": git, "environment": L.environment_record(), "protocol_sha256": hashlib.sha256(C.PROTOCOL.read_bytes()).hexdigest(),
           "branches": {}, "training": tr.get("wl28_reproduction")}
    composed = {}
    fz = {s: np.zeros_like(data[s]["frozen_pixels"]) for s in p28["evaluate_states"]}
    for tag, pred in [("frozen", fz)] + list(preds.items()):
        imgs = {s: compose(frozen[s], data[s]["roi_pixels"], data[s]["stable_pixels"], pred[s]) for s in STATES}
        img_b = compose(frozen_b, data["T0_B"]["roi_pixels"], data["T0_B"]["stable_pixels"], pred["T0_B"])
        composed[tag] = imgs
        ev = evaluate(proto_m, gt, {(s, "m"): imgs[s] for s in STATES}, img_b, rois, model_label="RNA")
        rl = rule(ev, "RNA", "m")
        # T1 control: correction difference T0 -> T1 over T1 ROI pixels paired to T0 (idx0)
        m1, i01 = rois["T1"]["interaction"]
        tm = L.tonemap
        f1, f0 = imgs["T1"].reshape(-1, 3), imgs["T0"].reshape(-1, 3)
        z1, z0 = frozen["T1"].reshape(-1, 3), frozen["T0"].reshape(-1, 3)
        g1, g0 = (0.5 * (gt["T1"][0] + gt["T1"][1])).reshape(-1, 3), (0.5 * (gt["T0"][0] + gt["T0"][1])).reshape(-1, 3)
        corr1 = tm(f1[m1]) - tm(z1[m1])
        corr0 = tm(f0[i01]) - tm(z0[i01])
        rec["branches"][tag] = {
            "T3_change": change_metrics(L, data, pred, "T3"),
            "T1b_change": change_metrics(L, data, pred, "T1b"), "T2_change": change_metrics(L, data, pred, "T2"),
            "abs_interaction_error": {s: rl["per_state"][s]["interaction"]["error"] for s in STATES},
            "refit_rule_own_T0": refit_block(gt, rois, imgs["T0"], imgs["T3"]),
            "refit_rule_frozen_T0": refit_block(gt, rois, frozen["T0"], imgs["T3"]),
            "frozen_rule_class": rl["classification"],
            "T1": {"abs_error": rl["per_state"]["T1"]["interaction"]["error"], "own_T0_error": rl["per_state"]["T1"]["interaction"]["error_T0"],
                   "rise": rl["per_state"]["T1"]["interaction"]["rise"], "gain": rl["per_state"]["T1"]["interaction"]["gain"],
                   "gt_change_display": float(np.abs(tm(g1[m1]) - tm(g0[i01])).mean()),
                   "correction_norm_T1_display": float(np.abs(corr1).mean()), "correction_norm_T0_display": float(np.abs(corr0).mean()),
                   "correction_difference_T0_to_T1_display": float(np.abs(corr1 - corr0).mean())}}
        if tag != "frozen":
            c = rec["branches"][tag]["T3_change"]
            print(f"{tag:16s} R_delta {c['R_delta']:.3f}  E_delta {c['E_delta_display']:.4f}  static {c['static_T0_error_display']:.4f}  "
                  f"T3 abs {c['state_error_display']:.4f}  gain {rec['branches'][tag]['refit_rule_own_T0']['gain']:.3f}", flush=True)
    # predeclared success test (criteria 1-3; 4 is the reviewer's qualitative judgement)
    seeds = p28["training"]["seeds"]
    Rd = lambda lab, s: rec["branches"][f"{lab}_s{s}"]["T3_change"]["R_delta"]
    c1 = all(Rd("D_real", s) < Rd("B_zero", s) for s in seeds)
    c2 = all(Rd("D_real", s) < Rd("C_shuffled", s) for s in seeds)
    c3 = sum(Rd("D_real", s) <= 0.50 for s in seeds) >= 2
    rec["success_test"] = {"criterion_1_below_zero_every_seed": c1, "criterion_2_below_shuffled_every_seed": c2,
                           "criterion_3_two_of_three_R_delta_le_0.5": c3,
                           "criterion_4": "reviewer judgement of the signed change maps, recorded in the worklog",
                           "numeric_pass": bool(c1 and c2 and c3),
                           "R_delta": {lab: [Rd(lab, s) for s in seeds] for lab in ("A_geometric", "B_zero", "C_shuffled", "D_real")
                                       + (("O_oracle",) if orc_f.exists() else ())}}
    if orc_f.exists():
        rec["oracle_case"] = "O1" if sum(Rd("O_oracle", s) <= 0.50 for s in seeds) >= 2 else "O2"
        rec["oracle_noise"] = json.loads((run / "oracle" / "oracle.json").read_text(encoding="utf-8"))["states"]
    print(json.dumps(rec["success_test"]), flush=True)
    rec["state_diagnostics"] = state_diagnostics(L, run, wl28, data, proto)
    # cost accounting (T3)
    rl_ = json.loads((run / "remote_light" / "remote_light.json").read_text(encoding="utf-8"))["states"]["T3"]
    px = json.loads((run / "proxy" / "proxy.json").read_text())["states"]["T3"]["gpu_timing"]
    p28j = json.loads((wl28 / "probes" / "probes.json").read_text(encoding="utf-8"))["states"]["T3"]
    f28 = json.loads((wl28 / "dataset" / "features.json").read_text())["states"]["T3"]
    tim = tr["timing_T3_real"]
    lat = {"probe_rays_gpu_wl28": p28j["probe_gpu_timing"]["seconds_median"],
           "remote_light_sampling": rl_["light_sampling_timing"]["seconds_median"],
           "proxy_rna_evaluation_gpu": px["proxy"]["seconds_median"],
           "persistent_feature_lookup_wl28": f28["remote_triplane_only_gpu_s"],
           "encoder_aggregation": tim["state_s_median"], "decoder": tim["decoder_s_median"]}
    lat["dynamic_state_update_total"] = sum(lat[k] for k in ("probe_rays_gpu_wl28", "remote_light_sampling", "proxy_rna_evaluation_gpu",
                                                             "persistent_feature_lookup_wl28", "encoder_aggregation"))
    lat["frozen_rna_roi_render_gpu"] = px["frozen_rna_roi_render"]["seconds_median"]
    q = tim["queries"]
    rec["cost_T3"] = {"Q": q, "K": p28["probes"]["K"], "QK": q * p28["probes"]["K"], "hits": rl_["hits"],
                      "hit_fraction": rl_["hits"] / (q * p28["probes"]["K"]), "proxy_rna_queries": px["proxy"]["rna_queries"],
                      "shadow_rays_for_proxy": rl_["light_sampling_timing"]["shadow_rays"],
                      "dynamic_state_bytes": tim["state_bytes_float32"], "raw_descriptor_bytes": q * p28["probes"]["K"] * len(C.PROBE_FEATURES) * 4,
                      "params": {t: v["params"] for t, v in tr["runs"].items() if t.endswith("_s0")},
                      "training_s": {t: v["train_s"] for t, v in tr["runs"].items()}, "peak_gpu_mib": {t: v["peak_gpu_mib"] for t, v in tr["runs"].items()},
                      "latency_s": lat,
                      "note": "stages run in separate processes (Mitsuba on Windows, torch in WSL) and are summed; one-hop RNA evaluation at the hits, no path tracing"}
    h28 = json.loads((wl28 / "models" / "training.json").read_text())["runs"]
    rec["params_by_branch"] = {"frozen": 0, "A_geometric": h28["relational_s0"]["params"], "B_zero": tr["runs"]["zero_s0"]["params"],
                               "C_shuffled": tr["runs"]["shuffled_s0"]["params"], "D_real": tr["runs"]["real_s0"]["params"]}
    if orc_f.exists():
        rec["params_by_branch"]["O_oracle"] = json.loads(orc_f.read_text())["runs"]["oracle_s0"]["params"]
    R.write_json(run / "eval" / "rrs_eval.json", rec)
    if args.worklog:
        export(run, rec, composed, frozen, gt, rois, L, args.worklog, orc_f.exists())
    return 0


def export(run, rec, composed, frozen, gt, rois, L, worklog, oracle):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colormaps

    from cross_backbone_exports import crop_box, upscale

    out = L.ROOT / "results" / "evaluation" / worklog
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    tm = L.tonemap
    g = {s: 0.5 * (gt[s][0] + gt[s][1]) for s in STATES}

    def signed_d(dd, scale=0.1):
        return (colormaps["RdBu_r"](np.clip(dd.mean(-1) / scale * 0.5 + 0.5, 0, 1))[..., :3] * 255).astype(np.uint8)

    def roi(s, pad=6):
        m = np.load(L.RESULTS / f"gt_design/common_light/rois_{s}.npz")["mask_interaction"]
        return crop_box(m, pad), m

    def masked(rgb, m, ys, xs):
        r = rgb.copy()
        r[~m] = 235
        return upscale(r[ys, xs], 512)

    tags = ["A_geometric_s0", "B_zero_s0", "C_shuffled_s0", "D_real_s0"] + (["O_oracle_s0"] if oracle else [])
    (ys, xs), m3 = roi("T3")
    dref = tm(g["T3"]) - tm(g["T0"])
    cols = [("GT change T0->T3", dref), ("frozen RNA change", tm(frozen["T3"]) - tm(frozen["T0"]))] + \
           [(f"{t} change", tm(composed[t]["T3"]) - tm(composed[t]["T0"])) for t in tags]
    r1 = [L.label(masked(signed_d(dd), m3, ys, xs), lab) for lab, dd in cols]
    r2 = [L.label(masked(L.error_map(np.abs(dd - dref).mean(-1), 0.2), m3, ys, xs), "|change - GT change| 0..0.2") for lab, dd in cols]
    name = "01_T3_signed_change_maps.png"
    L.save_png(out / name, np.concatenate([np.concatenate(r1, 1), np.concatenate(r2, 1)], 0))
    files[name] = {"note": "interaction ROI (grey outside); signed display change T0->T3, RdBu +-0.1; row 2 |branch change - GT change|"}
    # all seeds of the real branch and its controls
    rows = []
    for lab in ("D_real", "B_zero", "C_shuffled") + (("O_oracle",) if oracle else ()):
        rows.append(np.concatenate([L.label(masked(signed_d(tm(composed[f"{lab}_s{s}"]["T3"]) - tm(composed[f"{lab}_s{s}"]["T0"])), m3, ys, xs),
                                            f"{lab} s{s} change") for s in (0, 1, 2)], 1))
    name = "02_T3_change_maps_all_seeds.png"
    L.save_png(out / name, np.concatenate(rows, 0))
    files[name] = {"note": "signed display change T0->T3, RdBu +-0.1, interaction ROI"}
    # absolute T3 ROI, error, correction, full frame
    z = np.load(L.RESULTS / "gt_design/common_light/rois_T3.npz")
    ysw, xsw = crop_box(z["mask_interaction"] | z["mask_mover"] | np.load(L.RESULTS / "gt_design/common_light/rois_T0.npz")["mask_mover"])
    imgs = [("GT T3", g["T3"]), ("frozen RNA", frozen["T3"])] + [(t, composed[t]["T3"]) for t in tags]
    ldr = lambda im: L.to_u8(tm(im))
    full = [L.label(ldr(im), lab) for lab, im in imgs]
    crop = [L.label(upscale(ldr(im)[ysw, xsw], 512), lab + " (crop)") for lab, im in imgs]
    errm = [L.label(upscale(L.error_map(np.abs(tm(im) - tm(g["T3"])).mean(-1), 0.2)[ysw, xsw], 512), "|diff| vs GT 0..0.2") for lab, im in imgs]
    corr = [L.label(upscale(signed_d(tm(im) - tm(frozen["T3"]))[ysw, xsw], 512), lab + " - frozen") for lab, im in imgs]
    name = "03_T3_absolute_error_correction.png"
    L.save_png(out / name, np.concatenate([np.concatenate(r, 1) for r in (full, crop, errm, corr)], 0))
    files[name] = {"note": "row 1 full frame, row 2 crop, row 3 |display diff| vs GT (inferno 0..0.2), row 4 signed correction vs frozen (RdBu +-0.1)"}
    # T1: GT change (paired by idx0) and branch changes / corrections on the T1 ROI
    (y1, x1), m1 = roi("T1")
    mflat, idx0 = rois["T1"]["interaction"]
    def change_T1(img1, img0):
        d = np.zeros_like(img1).reshape(-1, 3)
        d[mflat] = tm(img1.reshape(-1, 3)[mflat]) - tm(img0.reshape(-1, 3)[idx0])
        return d.reshape(img1.shape)
    t1tags = ["D_real_s0", "B_zero_s0", "C_shuffled_s0"]
    c1 = [("GT change T0->T1", change_T1(g["T1"], g["T0"]))] + [(f"{t} change", change_T1(composed[t]["T1"], composed[t]["T0"])) for t in t1tags]
    k1 = [(f"{t} correction at T1", tm(composed[t]["T1"]) - tm(frozen["T1"])) for t in t1tags]
    name = "04_T1_change_and_correction.png"
    L.save_png(out / name, np.concatenate([np.concatenate([L.label(masked(signed_d(dd), m1, y1, x1), lab) for lab, dd in c1], 1),
                                           np.concatenate([np.full((512, 512, 3), 255, np.uint8)] +
                                                          [L.label(masked(signed_d(dd), m1, y1, x1), lab) for lab, dd in k1], 1)], 0))
    files[name] = {"note": "T1 ROI; row 1 signed display change T0->T1 (T1 pixels paired to T0 by idx0), row 2 signed correction vs frozen at T1"}
    shutil.copy2(run / "eval" / "rrs_eval.json", out / "05_rrs_eval.json")
    files["05_rrs_eval.json"] = {"note": "machine-readable metrics, success test, diagnostics, cost"}
    shutil.copy2(C.PROTOCOL, out / "06_protocol_rrs_v1.json")
    files["06_protocol_rrs_v1.json"] = {"note": "protocol"}
    lines = ["| branch | seed | R_delta (T3) | E_delta | T0 static err | T3 abs err | ratio (own T0) | gain | T1 corr. diff | params |", "|---|---|---|---|---|---|---|---|---|---|"]
    for tag, b in rec["branches"].items():
        c = b["T3_change"]
        lab, _, sd = tag.rpartition("_s") if tag != "frozen" else ("frozen", "", "-")
        p = rec["params_by_branch"].get(lab, "-")
        lines.append(f"| {lab} | {sd} | {c['R_delta']:.3f} | {c['E_delta_display']:.4f} | {c['static_T0_error_display']:.4f} | {c['state_error_display']:.4f} | "
                     f"{b['refit_rule_own_T0']['error_ratio']:.3f} | {b['refit_rule_own_T0']['gain']:.3f} | {b['T1']['correction_difference_T0_to_T1_display']:.4f} | {p} |")
    (out / "07_results_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    files["07_results_table.md"] = {"note": "from 05_rrs_eval.json"}
    lat = rec["cost_T3"]["latency_s"]
    tl = ["| quantity | value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in rec["cost_T3"].items() if k not in ("latency_s", "training_s", "peak_gpu_mib", "params")] + \
         [f"| latency {k} | {v * 1000:.2f} ms |" for k, v in lat.items()]
    (out / "08_cost_table.md").write_text("\n".join(tl) + "\n", encoding="utf-8")
    files["08_cost_table.md"] = {"note": "from 05_rrs_eval.json"}
    for nm in files:
        files[nm]["sha256"] = hashlib.sha256((out / nm).read_bytes()).hexdigest()
    L.write_json(out / "manifest.json", {"worklog": worklog, "run": L.rel(run), "project_commit": rec["git"]["commit"],
                                        "worklog28_run": C.protocol()["worklog28_reference"]["run"], "display": "clamp(x^(1/2.2),0,1)",
                                        "sources": {"frozen": "results/8dna_replication/rna_teaset/frozen/<state>_canonical.npy",
                                                    "references": "results/8dna_replication/frozen/teaset_common_light_8dna/<state>/gt_A|gt_B.exr",
                                                    "A_geometric": f"{C.protocol()['worklog28_reference']['run']}/models/pred_relational_s<seed>.npz",
                                                    "B/C/D/oracle": f"{L.rel(run)}/models/pred_<variant>_s<seed>.npz"},
                                        "files": files})


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
