"""Evaluate the relational candidate and the local-only control with the historical definitions (Windows).

    windows/run.ps1 ../relational_residual_prototype/rrp_eval.py --run <run dir> [--worklog 28]

Candidate image of a state = historical frozen RNA render + the branch's predicted
residual on the stable ROI pixels (everything else is the frozen render).  Then:
  * frozen-response metrics and rule: teaset_frozen_eval.evaluate + cross_backbone_eval.rule
    (worklog 22), each branch judged against its own T0, its T0_B image as noise repeat;
  * recovery rule: cross_backbone_eval.refit_block(gt, rois, N(T0), N(T3)) with the
    branch's own T0 (primary) and with the frozen T0 (secondary);
  * state accounting, latency and descriptor-change statistics;
  * reviewer panels and a manifest under results/evaluation/<worklog>/ (with --worklog).
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "8dna_deformation_replication"))
sys.path.insert(0, str(HERE.parents[0] / "neural_recompute_cost"))

import numpy as np  # noqa: E402

import rrp_common as C  # noqa: E402

STATES = ("T0", "T1", "T1b", "T2", "T3")


def compose(frozen: np.ndarray, roi_pix: np.ndarray, stable_idx: np.ndarray, residual: np.ndarray) -> np.ndarray:
    out = frozen.reshape(-1, 3).copy()
    out[roi_pix[stable_idx]] += residual.astype(np.float32)
    return out.reshape(frozen.shape)


def descriptor_change(d0: dict, ds: dict) -> dict:
    """T0 vs state on the stable pixels both share (identity-pixel states only)."""
    p0 = d0["roi_pixels"][d0["stable_pixels"]]
    ps = ds["roi_pixels"][ds["stable_pixels"]]
    common, i0, i1 = np.intersect1d(p0, ps, return_indices=True)
    k = int(d0["samples_per_pixel"])
    q0 = (i0[:, None] * k + np.arange(k)).reshape(-1)
    q1 = (i1[:, None] * k + np.arange(k)).reshape(-1)
    a, b = d0["probes"][q0], ds["probes"][q1]
    la, lb = d0["local"][q0], ds["local"][q1]
    names = list(C.PROBE_FEATURES)
    hit_a, hit_b = a[..., names.index("hit")] > 0, b[..., names.index("hit")] > 0
    changed = np.abs(a - b).max(-1) > 0
    feat_idx = [names.index(f"remote_feat_{c}") for c in range(8)]
    remote_feat_changed = np.abs(a[..., feat_idx] - b[..., feat_idx]).max(-1) > 0
    lnames = list(C.LOCAL_FEATURES)
    persistent = [lnames.index(f"query_feat_{c}") for c in range(8)] + [lnames.index(n) for n in ("view_s", "view_n", "view_t", "light_s", "light_n", "light_t")]
    return {"common_stable_pixels": int(len(common)), "queries": int(len(q0)), "probes": int(a.shape[0] * a.shape[1]),
            "probe_changed_fraction": float(changed.mean()), "query_with_any_probe_change_fraction": float(changed.any(1).mean()),
            "hit_status_changed_fraction": float((hit_a != hit_b).mean()),
            "remote_feature_changed_fraction_of_both_hit": float(remote_feat_changed[hit_a & hit_b].mean()) if (hit_a & hit_b).any() else None,
            "mean_abs_descriptor_change": float(np.abs(a - b).mean()),
            "persistent_query_inputs_max_abs_change": float(np.abs(la[:, persistent] - lb[:, persistent]).max()),
            "direct_vis_fraction_changed_queries": float(np.mean(la[:, lnames.index("direct_vis_fraction")] != lb[:, lnames.index("direct_vis_fraction")]))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--worklog", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    run = Path(args.run).resolve()
    proto = C.protocol()
    import ednalib as L
    import nrc_records as R

    git = R.git_state(L.ROOT)
    R.require_clean(git, args.allow_dirty)
    L.init_upstream()
    from cross_backbone_eval import refit_block, rule
    from teaset_frozen_eval import evaluate, load_rois

    cb = json.loads((L.EXPERIMENT / proto["base_protocol"]).read_text(encoding="utf-8"))
    fdir, rdir = L.RESULTS / cb["frozen_output"], L.RESULTS / cb["rna_output"]
    rois = load_rois(L.RESULTS / cb["gt_output"], cb["states"])
    gt = {s: (L.load_exr(fdir / s / "gt_A.exr"), L.load_exr(fdir / s / "gt_B.exr")) for s in STATES}
    frozen = {s: np.load(rdir / f"{s}_canonical.npy").astype(np.float32) for s in STATES}
    frozen_b = np.load(rdir / "T0_canonical_B.npy").astype(np.float32)
    data = {s: dict(np.load(run / "dataset" / f"{s}.npz")) for s in proto["evaluate_states"]}
    training = json.loads((run / "models" / "training.json").read_text())
    oracle_f = run / "models" / "oracle_training.json"  # predeclared oracle diagnostic, if it was run (never a method)
    if oracle_f.exists():
        training["runs"].update(json.loads(oracle_f.read_text())["runs"])
    feats = json.loads((run / "dataset" / "features.json").read_text())
    probes = json.loads((run / "probes" / "probes.json").read_text(encoding="utf-8"))
    proto_m = dict(cb, modes={s: ["m"] for s in cb["states"]})

    def score(images: dict, image_b: np.ndarray) -> dict:
        ev = evaluate(proto_m, gt, {(s, "m"): images[s] for s in STATES}, image_b, rois, model_label="RNA")
        rl = rule(ev, "RNA", "m")
        return {"rule": rl, "abs_interaction_error": {s: rl["per_state"][s]["interaction"]["error"] for s in STATES},
                "refit_rule_own_T0": refit_block(gt, rois, images["T0"], images["T3"]),
                "refit_rule_frozen_T0": refit_block(gt, rois, frozen["T0"], images["T3"])}

    rec = {"git": git, "environment": L.environment_record(), "protocol_sha256": hashlib.sha256(C.PROTOCOL.read_bytes()).hexdigest(),
           "branches": {}, "baselines": {}}
    rec["baselines"]["frozen_rna"] = score(frozen, frozen_b)
    for tag, path in (("wl27_rebuild_val_best", "results/neural_recompute_cost/v1_d2d6432/eval/rna/official_epoch=117-val_psnr=18.02dB.npy"),
                      ("wl22_refit", "results/8dna_replication/rna_teaset/frozen/refit_T3_current.npy")):
        img = np.load(L.ROOT / path).astype(np.float32)
        rec["baselines"][tag] = {"render": path, "refit_rule_frozen_T0": refit_block(gt, rois, frozen["T0"], img),
                                 "abs_interaction_error_T3": float(L.pixel_metrics(img.reshape(-1, 3)[rois["T3"]["interaction"][0]],
                                                                                    (0.5 * (gt["T3"][0] + gt["T3"][1])).reshape(-1, 3)[rois["T3"]["interaction"][0]])["display_mae"])}
    composed = {}
    for tag in training["runs"]:
        pred = dict(np.load(run / "models" / f"pred_{tag}.npz"))
        imgs = {s: compose(frozen[s], data[s]["roi_pixels"], data[s]["stable_pixels"], pred[s]) for s in STATES}
        img_b = compose(frozen_b, data["T0_B"]["roi_pixels"], data["T0_B"]["stable_pixels"], pred["T0_B"])
        composed[tag] = imgs
        sc = score(imgs, img_b)
        extra = {}
        for s in STATES:
            d = data[s]
            y = d["gt_pixels"] - d["frozen_pixels"]
            p = pred[s]
            disp = np.abs(L.tonemap(d["frozen_pixels"] + p) - L.tonemap(d["frozen_pixels"])).mean(-1)
            extra[s] = {"stable_pixels": int(len(p)), "residual_tracking_mae_linear": float(np.abs(p - y).mean()),
                        "residual_target_mae_linear": float(np.abs(y).mean()), "correction_mae_linear": float(np.abs(p).mean()),
                        "correction_display_mae": float(disp.mean()),
                        "meaningful_correction_fraction": float(np.mean(disp > proto["evaluation"]["meaningful_correction_display_threshold"]))}
        sc["extra"] = extra
        sc["training"] = {k: v for k, v in training["runs"][tag].items() if k != "curve"}
        rec["branches"][tag] = sc
        r = sc["refit_rule_own_T0"]
        print(tag, f"T3 ratio {r['error_ratio']:.3f} gain {r['gain']:.3f} recovers {r['recovers']} | frozen rule "
                   f"{sc['rule']['classification']} | T1 rise {sc['rule']['per_state']['T1']['interaction']['rise']:+.1%} "
                   f"gain {sc['rule']['per_state']['T1']['interaction']['gain']:.2f}", flush=True)
    rec["descriptor_change_vs_T0"] = {s: descriptor_change(data["T0"], data[s]) for s in ("T1b", "T2", "T3", "T0_B")}
    t3 = probes["states"]["T3"]
    tim = training["timing_T3"]
    q = t3["queries"]
    rec["state_accounting_T3"] = {
        "Q_queries": q, "K": proto["probes"]["K"], "QK_probes": t3["probes"], "probe_hit_fraction": t3["probe_hit_fraction"],
        "probe_hits_by_part": t3["probe_hits_by_part"], "state_dim": proto["model"]["state_dim"],
        "state_bytes_float32": tim["relational"]["state_bytes_float32"],
        "raw_descriptor_bytes_float32": int(q * proto["probes"]["K"] * len(C.PROBE_FEATURES) * 4),
        "params": {b: training["runs"][f"{b}_s0"]["params"] for b in ("relational", "local_only")},
        "latency_s": {"probe_rays_gpu": t3["probe_gpu_timing"]["seconds_median"],
                      "remote_feature_lookup": feats["states"]["T3"]["remote_triplane_only_gpu_s"],
                      "remote_feature_lookup_via_full_rna_forward": feats["states"]["T3"]["remote_feature_lookup_gpu_resident_s"],
                      "remote_feature_lookup_incl_host_copies": feats["states"]["T3"]["remote_feature_lookup_s"],
                      "state_encoder_aggregation": tim["relational"]["state_s_median"],
                      "decoder": tim["relational"]["decoder_s_median"],
                      "local_only_total": tim["local_only"]["state_plus_decoder_s_median"]},
        "structure": "O(Q*K) probe rays and per-probe encoder work for fixed K; no all-pairs interaction"}
    lat = rec["state_accounting_T3"]["latency_s"]
    lat["relational_update_total"] = lat["probe_rays_gpu"] + lat["remote_feature_lookup"] + lat["state_encoder_aggregation"]
    rec["state_accounting_T3"]["latency_note"] = ("stages run in two processes (Mitsuba on Windows, torch in WSL) and are summed; "
                                                  "the remote-feature figure is RNA's triplane module alone on GPU-resident positions (checked identical to the hooked features); "
                                                  "frozen RNA rendering and its G-buffer are not included")
    lat["relational_update_plus_inference"] = lat["relational_update_total"] + lat["decoder"]
    rec["features_check"] = {k: v for k, v in feats.items() if k != "states"} | {
        "base_vs_historical_frozen_max_rel": {s: v["base_vs_historical_frozen_max_rel"] for s, v in feats["states"].items()}}
    rec["training_counts"] = training["counts"]
    R.write_json(run / "eval" / "rrp_eval.json", rec)
    if args.worklog:
        export(run, rec, composed, frozen, gt, rois, data, L, args.worklog)
    return 0


def export(run, rec, composed, frozen, gt, rois, data, L, worklog):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colormaps

    from cross_backbone_exports import crop_box, upscale

    out = L.ROOT / "results" / "evaluation" / worklog
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    ldr = lambda im: L.to_u8(L.tonemap(im))
    g = {s: 0.5 * (gt[s][0] + gt[s][1]) for s in STATES}
    err = lambda a, s: L.error_map(np.abs(L.tonemap(a) - L.tonemap(g[s])).mean(-1), 0.2)

    def signed(a, b, scale=0.1):
        d = (L.tonemap(a) - L.tonemap(b)).mean(-1)
        rgb = colormaps["RdBu_r"](np.clip(d / scale * 0.5 + 0.5, 0, 1))[..., :3]
        return (rgb * 255).astype(np.uint8)

    z = np.load(L.RESULTS / "gt_design/common_light/rois_T3.npz")
    ys, xs = crop_box(z["mask_interaction"] | z["mask_mover"] | np.load(L.RESULTS / "gt_design/common_light/rois_T0.npz")["mask_mover"])
    wl27 = np.load(L.ROOT / rec["baselines"]["wl27_rebuild_val_best"]["render"]).astype(np.float32)
    rel, loc = composed["relational_s0"], composed["local_only_s0"]
    for s, cols in (("T3", [("GT T3", g["T3"]), ("frozen RNA", frozen["T3"]), ("frozen + local-only", loc["T3"]),
                            ("frozen + relational", rel["T3"]), ("WL27 full T3 rebuild", wl27)]),
                    ("T1", [("GT T1", g["T1"]), ("frozen RNA", frozen["T1"]), ("frozen + local-only", loc["T1"]),
                            ("frozen + relational", rel["T1"])])):
        rows = [np.concatenate([L.label(ldr(im), lab) for lab, im in cols], 1),
                np.concatenate([L.label(upscale(ldr(im)[ys, xs], 512), lab + " (ROI crop)") for lab, im in cols], 1),
                np.concatenate([L.label(upscale(err(im, s)[ys, xs], 512), "|display diff| vs GT 0..0.2") for lab, im in cols], 1)]
        corr = [np.full((512, 512, 3), 255, np.uint8), upscale(signed(g[s], frozen[s])[ys, xs], 512)]
        corr[0] = L.label(corr[0], "signed maps, display units, -0.1 (blue) .. +0.1 (red)")
        corr[1] = L.label(corr[1], f"GT - frozen ({s})")
        for lab, im in cols[2:4]:
            corr.append(L.label(upscale(signed(im, frozen[s])[ys, xs], 512), f"{lab} - frozen"))
        while len(corr) < len(cols):
            corr.append(np.full((512, 512, 3), 255, np.uint8))
        rows.append(np.concatenate(corr, 1))
        name = f"0{1 if s == 'T3' else 2}_review_{s}.png"
        L.save_png(out / name, np.concatenate(rows, 0))
        files[name] = {"sources": [f"{L.rel(run)}/models/pred_relational_s0.npz", f"{L.rel(run)}/models/pred_local_only_s0.npz",
                                   f"results/8dna_replication/frozen/teaset_common_light_8dna/{s}/gt_A.exr",
                                   f"results/8dna_replication/rna_teaset/frozen/{s}_canonical.npy"]
                       + ([rec["baselines"]["wl27_rebuild_val_best"]["render"]] if s == "T3" else [])}
        frames = [L.label(upscale(ldr(im)[ys, xs], 512), lab) for lab, im in cols]
        L.save_gif(out / f"0{1 if s == 'T3' else 2}_review_{s}_crop_cycle.gif", frames, ms=900)
        files[f"0{1 if s == 'T3' else 2}_review_{s}_crop_cycle.gif"] = {"sources": files[name]["sources"]}
    # ROI-tight change maps: does a branch's T0 -> T3 change follow the reference change?
    def roi_box(s, pad=6):
        m = np.load(L.RESULTS / f"gt_design/common_light/rois_{s}.npz")["mask_interaction"]
        return crop_box(m, pad), m

    def masked(rgb, m, ys_, xs_):
        r = rgb.copy()
        r[~m] = 235
        return upscale(r[ys_, xs_], 512)

    (ys3, xs3), m3 = roi_box("T3")
    tags = [t for t in ("local_only_s0", "relational_s0", "relational_s1", "relational_s2", "oracle_s0") if t in composed]
    ch = [("reference change GT(T3)-GT(T0)", g["T3"], g["T0"]), ("frozen RNA change", frozen["T3"], frozen["T0"])] + \
         [(f"{t} change", composed[t]["T3"], composed[t]["T0"]) for t in tags]
    row1 = [L.label(masked(signed(a, b), m3, ys3, xs3), lab) for lab, a, b in ch]
    dgd = L.tonemap(g["T3"]) - L.tonemap(g["T0"])
    row2 = [L.label(masked(L.error_map(np.abs((L.tonemap(a) - L.tonemap(b)) - dgd).mean(-1), 0.2), m3, ys3, xs3),
                    "|change - reference change| 0..0.2") for lab, a, b in ch]
    name = "07_T3_change_maps_roi.png"
    L.save_png(out / name, np.concatenate([np.concatenate(row1, 1), np.concatenate(row2, 1)], 0))
    files[name] = {"sources": [f"{L.rel(run)}/models/pred_{t}.npz" for t in tags] + ["results/8dna_replication/rna_teaset/frozen/T3_canonical.npy",
                                                                                    "results/8dna_replication/rna_teaset/frozen/T0_canonical.npy"],
                   "note": "interaction ROI only (grey outside), signed display-space change T0->T3, RdBu +-0.1; row 2 |branch change - reference change|"}
    (ys1, xs1), m1 = roi_box("T1")
    res1 = [("target residual GT(T1)-frozen(T1)", g["T1"])] + [(f"{t} residual", composed[t]["T1"]) for t in tags]
    row = [L.label(masked(signed(a, frozen["T1"]), m1, ys1, xs1), lab) for lab, a in res1]
    res0 = [("target residual GT(T0)-frozen(T0)", g["T0"])] + [(f"{t} residual", composed[t]["T0"]) for t in tags]
    (ys0, xs0), m0 = roi_box("T0")
    row0 = [L.label(masked(signed(a, frozen["T0"]), m0, ys0, xs0), lab + " (T0)") for lab, a in res0]
    name = "08_T1_vs_T0_residuals_roi.png"
    L.save_png(out / name, np.concatenate([np.concatenate(row0, 1), np.concatenate(row, 1)], 0))
    files[name] = {"sources": [f"{L.rel(run)}/models/pred_{t}.npz" for t in tags],
                   "note": "row 1 T0 (training), row 2 T1 (held-out relation-preserving control); each state's own ROI; signed residual vs frozen RNA, RdBu +-0.1"}
    shutil.copy2(run / "eval" / "rrp_eval.json", out / "03_rrp_eval.json")
    files["03_rrp_eval.json"] = {"sources": [f"{L.rel(run)}/eval/rrp_eval.json"]}
    shutil.copy2(C.PROTOCOL, out / "04_protocol_rrp_v1.json")
    files["04_protocol_rrp_v1.json"] = {"sources": [L.rel(C.PROTOCOL)]}
    lines = ["| branch | seed | T3 ratio (own T0) | T3 gain | recovers | frozen-rule class | T1 rise | T1 gain | T3 abs err | params |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    fr = rec["baselines"]["frozen_rna"]
    lines.append(f"| frozen RNA | - | {fr['refit_rule_own_T0']['error_ratio']:.3f} | {fr['refit_rule_own_T0']['gain']:.3f} | "
                 f"{fr['refit_rule_own_T0']['recovers']} | {fr['rule']['classification']} | {fr['rule']['per_state']['T1']['interaction']['rise']:+.1%} | "
                 f"{fr['rule']['per_state']['T1']['interaction']['gain']:.2f} | {fr['abs_interaction_error']['T3']:.4f} | 0 |")
    for tag, b in rec["branches"].items():
        r = b["refit_rule_own_T0"]
        lines.append(f"| {b['training']['branch']} | {b['training']['seed']} | {r['error_ratio']:.3f} | {r['gain']:.3f} | {r['recovers']} | "
                     f"{b['rule']['classification']} | {b['rule']['per_state']['T1']['interaction']['rise']:+.1%} | "
                     f"{b['rule']['per_state']['T1']['interaction']['gain']:.2f} | {b['abs_interaction_error']['T3']:.4f} | {b['training']['params']} |")
    for tag in ("wl27_rebuild_val_best", "wl22_refit"):
        r = rec["baselines"][tag]["refit_rule_frozen_T0"]
        lines.append(f"| {tag} (vs frozen T0, historical semantics) | - | {r['error_ratio']:.3f} | {r['gain']:.3f} | {r['recovers']} | - | - | - | "
                     f"{rec['baselines'][tag]['abs_interaction_error_T3']:.4f} | full RNA |")
    (out / "05_results_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    files["05_results_table.md"] = {"sources": [f"{L.rel(run)}/eval/rrp_eval.json"]}
    acc = rec["state_accounting_T3"]
    tl = ["| quantity | value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in acc.items() if k != "latency_s"] + \
         [f"| latency {k} | {v * 1000:.2f} ms |" for k, v in acc["latency_s"].items()]
    (out / "06_state_and_timing.md").write_text("\n".join(tl) + "\n", encoding="utf-8")
    files["06_state_and_timing.md"] = {"sources": [f"{L.rel(run)}/eval/rrp_eval.json"]}
    for name in files:
        files[name]["sha256"] = hashlib.sha256((out / name).read_bytes()).hexdigest()
    L.write_json(out / "manifest.json", {"worklog": worklog, "run": L.rel(run), "project_commit": rec["git"]["commit"],
                                        "protocol": L.rel(C.PROTOCOL), "display": "clamp(x^(1/2.2),0,1)",
                                        "panels": "row 1 full frame, row 2 interaction-ROI crop, row 3 |display diff| vs GT (inferno 0..0.2), "
                                                  "row 4 signed display-space maps (RdBu, +-0.1): GT - frozen and each branch - frozen",
                                        "files": files})


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
