"""Metrics, timing and gates for worklog 31 (protocol sor_v1.json `evaluation`, `gates`).

    windows/run.ps1 ../surface_owned_radiance/sor_metrics.py --root <batch dir>

Reads arm outputs, the oracle, held-out records, references and regions; writes <root>/metrics.json.
Total image of an arm at a state = worklog-24 class images v0 + v1 of that state (physical direct,
current geometry) + the arm's learned >= 2-vertex image.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import sor_common as S

HIST = S.ROOT / "results" / "8dna_replication"
REG = S.ROOT / "results" / "selective_refit_oracle" / "v1_bc085ea" / "regions" / "regions.npz"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--include-smoke", action="store_true", help="pipeline check only; never evidence")
    args = ap.parse_args()
    import nrc_records as R
    import ednalib as L

    git = R.git_state(L.ROOT)
    R.require_clean(git, args.allow_dirty)
    mi, dr = S.init()
    import numpy as np
    import torch
    from cross_backbone_eval import refit_block
    from sro_metrics import change_tracking
    from teaset_frozen_eval import load_rois
    from sor_targets import camera_queries, scene_for

    root = Path(args.root)
    proto = S.protocol()
    disp = L.tonemap
    gt = {s: (L.load_exr(HIST / "w21_reference_correction" / s / "gt_A.exr"), L.load_exr(HIST / "w21_reference_correction" / s / "gt_B.exr")) for s in ("T0", "T3")}
    g0, g3 = 0.5 * (gt["T0"][0] + gt["T0"][1]), 0.5 * (gt["T3"][0] + gt["T3"][1])
    cls = {s: np.load(HIST / "transport_decomposition" / "w21_envmap" / f"image_{s}.npz")["classes"] for s in ("T0", "T3")}
    direct = {s: cls[s][:, :, 0] + cls[s][:, :, 1] for s in cls}
    ref_ind = {s: cls[s][:, :, 2:9].sum(2) for s in cls}
    reg = dict(np.load(REG))
    rois = load_rois(HIST / "gt_design" / "v2", ("T0", "T3"))
    masks = {k: reg[k] for k in ("R_aff", "R_aff_stat", "R_mover", "R_unaff", "R_buffer", "R_unstable")}
    masks["asset"] = ~reg["background"]
    masks["roi_interaction"] = rois["T3"]["interaction"][0].reshape(512, 512)
    for i, p in enumerate(S.PARTS):
        masks[f"obj_{p}"] = reg["part_T3"] == p
        masks[f"obj_{p}_stable"] = (reg["part_T3"] == p) & reg["stable_stationary"]

    def score(ind, state="T3"):
        tot = direct[state] + ind
        g = g3 if state == "T3" else g0
        d = np.abs(disp(tot) - disp(g)).mean(-1)
        row = {k: float(d[m].mean()) for k, m in masks.items() if m.any()}
        row["indirect_linear_mae"] = {k: float(np.abs(ind - ref_ind[state])[masks[k]].mean())
                                      for k in ("R_aff_stat", "R_unaff", "R_mover", "asset")}
        return row, tot

    arms_dir = root / "arms"
    t0_arm = arms_dir / f"T0_s{proto['fit_contract']['pretrain_T0']['seed']}"
    t0_snaps = np.load(t0_arm / "snapshots.npz")
    t0_last = sorted(t0_snaps.files)[-1]
    ind_T0 = t0_snaps[t0_last]
    t0_row, n0 = score(ind_T0, "T0")
    ref8 = L.load_exr(HIST / "frozen" / "teaset_locked" / "T0" / "8dna_upstream.exr")
    m_int0 = rois["T0"]["interaction"][0].reshape(512, 512)
    static = {
        "T0_total_display_mae": {k: t0_row[k] for k in ("roi_interaction", "R_aff_stat", "R_unaff", "asset") if k in t0_row},
        "T0_interaction_same_surface": refit_block(gt, rois, n0, n0)["T0_model_error_same_surface"],
        "8dna_T0_interaction": float(np.abs(disp(ref8) - disp(g0)).mean(-1)[m_int0].mean()),
        "reference_change_interaction": float(np.abs(disp(g3).reshape(-1, 3)[rois["T3"]["interaction"][0]] - disp(g0).reshape(-1, 3)[rois["T3"]["interaction"][1]]).mean()),
        "reference_change_R_aff_stat": float(np.abs(disp(g3) - disp(g0)).mean(-1)[reg["R_aff_stat"]].mean()),
        "T0_same_surface_R_aff_stat": float(np.abs(disp(n0) - disp(g0)).mean(-1)[reg["R_aff_stat"]].mean()),
        "T0_heldout": json.loads((t0_arm / "arm.json").read_text())["heldout"][-1],
        "T0_indirect_linear_mae": t0_row["indirect_linear_mae"],
    }
    gates = {"G0_O0_static": bool(static["T0_interaction_same_surface"] < static["reference_change_interaction"]
                                  and static["T0_same_surface_R_aff_stat"] < static["reference_change_R_aff_stat"]
                                  and static["T0_interaction_same_surface"] <= 1.25 * static["8dna_T0_interaction"])}

    # camera stencils at T3 for pixel-identity checks
    units = S.load_units(root / "units" / "units.npz")
    model = S.SurfaceRadiance(units, proto["representation"]["res"])
    sc, ids, tr = scene_for("T3", 512)
    cam = camera_queries(sc, ids, tr, 512, 4, seed=5)
    from sor_fit import camera_stencil_units
    cam_units = camera_stencil_units(model, cam)

    out = {"schema": "sor_metrics/v1", "git": git, "static": static, "gates": gates, "arms": {}}
    frozen_img = None
    for d in sorted(arms_dir.iterdir()):
        if d.name.startswith("T0_") or not (d / "arm.json").exists():
            continue
        arm = json.loads((d / "arm.json").read_text())
        if arm.get("smoke") and not args.include_smoke:
            continue
        snaps = np.load(d / "snapshots.npz")
        ev = R.load_events(d / "events.jsonl")
        ivs = R.intervals(ev)
        t_created = next(e for e in ev if e["event"] == "process_created")["t_created"]
        evals = [r for r in ivs if r["phase"] == "evaluation" and r["seconds"] is not None]
        ready = {int(e["step"]): e for e in ev if e["event"] == "snapshot_ready"}
        done = {int(e["step"]): e for e in ev if e["event"] == "snapshot_evaluated"}
        trace = []
        for k in sorted(snaps.files):
            step = int(k.split("_")[1])
            ind = snaps[k]
            row, tot = score(ind)
            blk = refit_block(gt, rois, n0, tot)
            row["rule"] = {kk: blk[kk] for kk in ("error_ratio", "gain", "recovers", "refit_T3_interaction_error")}
            row["tracking"] = {kk: change_tracking(tot, n0, g3, g0, reg[kk]) for kk in ("R_aff_stat", "R_unaff")}
            e = ready[step]
            prior_eval = sum(r["seconds"] for r in evals if r["end"] <= e["t"] + 1e-6)
            prior_eval += sum(float(done[s]["eval_s"]) for s in done if s < step)
            row.update({"step": step, "adaptation_latency_s": e["t"] - t_created - prior_eval,
                        "trace_s": e["trace_s"], "opt_s": e["opt_s"], "samples_accepted": e["accepted"]})
            trace.append(row)
            if arm["arm"] == "C" and step == 0 and frozen_img is None:
                frozen_img = ind
        out["arms"][d.name] = {"arm": arm["arm"], "seed": arm["seed"], "objects_trainable": arm["objects_trainable"],
                               "trainable_units": arm["trainable_units"], "support_fraction": arm["support_fraction"],
                               "batch": arm["batch"], "identity": arm["identity"], "heldout": arm["heldout"],
                               "fit": arm["fit"], "peak_mem_gib": arm["peak_mem_gib"], "trace": trace,
                               "final_table": str(d / "table.pt")}

    # A (frozen) = C's step-0 render; pixel identity of frozen-object pixels for selective arms
    out["A_frozen"], _ = score(frozen_img) if frozen_img is not None else (None, None)
    if frozen_img is not None:
        rA, totA = score(frozen_img)
        out["A_frozen"]["rule"] = {kk: v for kk, v in refit_block(gt, rois, n0, totA).items() if kk in ("error_ratio", "gain", "recovers")}
        out["A_frozen"]["tracking"] = {kk: change_tracking(totA, n0, g3, g0, reg[kk]) for kk in ("R_aff_stat", "R_unaff")}
    for name, a in out["arms"].items():
        if a["arm"] in ("DG", "E", "E5", "U") and frozen_img is not None:
            d = arms_dir / name
            tab = torch.load(d / "table.pt", map_location="cpu")
            t0tab = torch.load(t0_arm / "table.pt", map_location="cpu")
            R2 = model.res ** 2
            changed_units = (tab.reshape(-1, R2, 3) != t0tab.reshape(-1, R2, 3)).reshape(model.n_units, -1).any(1).numpy()
            reads_changed = np.where(cam_units >= 0, changed_units[np.clip(cam_units, 0, None)], False).any(1)
            pix_changed = reads_changed.reshape(512 * 512, 16).any(1).reshape(512, 512)
            snaps = np.load(d / "snapshots.npz")
            last = snaps[sorted(snaps.files)[-1]]
            diff = np.abs(last - frozen_img).max(-1)
            fz = ~pix_changed & masks["asset"]
            a["pixel_identity"] = {"pixels_reading_only_frozen_units": int(fz.sum()), "changed": int((diff[fz] > 0).sum()),
                                   "max_abs_diff": float(diff[fz].max()) if fz.any() else 0.0,
                                   "units_changed": int(changed_units.sum())}

    def fin(name):
        a = out["arms"].get(name)
        return a["trace"][-1] if a else None

    C, E = fin("C_s0"), fin("E_s0")
    gl = None
    if C:
        gl = bool(C["rule"]["recovers"] and C["tracking"]["R_aff_stat"]["R_delta"] <= 0.5)
    gates["G_local"] = gl
    orc = json.loads((root / "oracle" / "oracle.json").read_text())
    screen = {}
    for sel_arm, sd in (("E", 0), ("E", 1), ("E5", 0), ("DG", 0), ("U", 0)):
        C, E = fin(f"C_s{sd}"), fin(f"{sel_arm}_s{sd}")
        if not (C and E):
            continue
        q_star = 1.10 * C["R_aff"]
        first = lambda tr_: next((r for r in tr_ if r["R_aff"] <= q_star), None)
        tC, tE = first(out["arms"][f"C_s{sd}"]["trace"]), first(out["arms"][f"{sel_arm}_s{sd}"]["trace"])
        objs = out["arms"][f"{sel_arm}_s{sd}"]["objects_trainable"]
        screen[f"{sel_arm}_seed{sd}"] = {
            "objects_refit": objs,
            "quality": {"sel": E["R_aff"], "C": C["R_aff"], "ratio": E["R_aff"] / C["R_aff"], "pass": bool(E["R_aff"] <= 1.10 * C["R_aff"]),
                        "R_aff_stat": [E["R_aff_stat"], C["R_aff_stat"]], "R_mover": [E["R_mover"], C["R_mover"]],
                        "per_object": {p: [E.get(f"obj_{p}"), C.get(f"obj_{p}")] for p in S.PARTS}},
            "nontrivial": bool(objs is not None and len(objs) < len(S.PARTS)),
            "preservation": out["arms"][f"{sel_arm}_s{sd}"].get("identity"),
            "pixel_identity": out["arms"][f"{sel_arm}_s{sd}"].get("pixel_identity"),
            "cost": {"Q_star": q_star, "t_C": tC["adaptation_latency_s"] if tC else None, "t_sel": tE["adaptation_latency_s"] if tE else None,
                     "speedup": (tC["adaptation_latency_s"] / tE["adaptation_latency_s"]) if (tC and tE) else None,
                     "final_latency": [E["adaptation_latency_s"], C["adaptation_latency_s"]],
                     "samples": [E["samples_accepted"], C["samples_accepted"]]},
        }
    out["screening"] = screen
    out["oracle"] = {k: orc[k] for k in ("object_change_share", "object_relative_change", "dirty_objects", "dirty_objects_sensitivity", "masks", "timing")}
    S.write_json(root / "metrics.json", out)
    print(json.dumps({"static": static, "gates": gates, "screening": screen}, indent=1, default=str)[:5000], flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(rc)
