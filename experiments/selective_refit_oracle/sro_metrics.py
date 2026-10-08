"""Metrics, timing and gates for the oracle selective-refit batch (protocol `evaluation`, `cost_accounting`, `gates`).

    windows/run.ps1 ../selective_refit_oracle/sro_metrics.py --root <batch results dir> [--allow-dirty]

Reads (never writes) training runs, renders, the oracle mask, regions, references and the
historical worklog-21/27 renders; writes <root>/metrics.json.  Region and gate definitions are
the protocol's; nothing here is tuned on the results.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import sro_common as S

REGION_KEYS = ("R_aff", "R_aff_stat", "R_mover", "R_unaff", "R_buffer", "R_unstable")
HIST = S.ROOT / "results" / "8dna_replication"
WL27 = S.ROOT / "results" / "neural_recompute_cost" / "v1_d2d6432"


def disp(x):
    import ednalib as L

    return L.tonemap(x)


def region_errors(img, g3, regions, rois):
    import numpy as np

    d = np.abs(disp(img) - disp(g3)).mean(-1)
    out = {k: float(d[regions[k]].mean()) for k in REGION_KEYS}
    asset = ~regions["background"]
    out["asset"] = float(d[asset].mean())
    out["full_image"] = float(d.mean())
    for k in ("interaction", "tray", "far", "mover"):
        m = rois["T3"][k][0].reshape(d.shape)
        out[f"roi_{k}"] = float(d[m].mean())
    return out


def change_tracking(img3, n0, g3, g0, mask):
    """Stable pixels only (same surface at the same pixel in T0 and T3)."""
    import numpy as np

    dN, dG = (img3 - n0)[mask].reshape(-1), (g3 - g0)[mask].reshape(-1)
    gg = float(dG @ dG)
    dNd = (disp(img3) - disp(n0))[mask]
    dGd = (disp(g3) - disp(g0))[mask]
    e_delta = float(np.abs(dNd - dGd).mean())
    mag = float(np.abs(dGd).mean())
    return {"gain": float(dN @ dG / gg) if gg > 0 else None, "E_delta": e_delta, "dG_mean_abs": mag,
            "R_delta": e_delta / mag if mag > 0 else None}


def updated_cells(ckpt, released_sd, N):
    import torch

    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    a, b = sd[S.TRIPLANE_KEY], released_sd[S.TRIPLANE_KEY]
    return (a != b).any(1).reshape(-1).numpy(), sd


def pixel_stencil_cells(x1_sub, N):
    """(P, 25, 3) sub-pixel first hits -> (P, 25*12) flat cell indices (-1 where no hit)."""
    import numpy as np
    import torch

    P = x1_sub.shape[0]
    x = torch.from_numpy(np.nan_to_num(x1_sub.reshape(-1, 3), nan=0.0))
    idx, w = S.stencil(x, N)
    idx = idx.reshape(P, 25, 12).numpy()
    valid = ~np.isnan(x1_sub[..., 0])
    idx[~valid] = -1
    return idx.reshape(P, -1)


def adaptation_times(events_path):
    """Per snapshot step: adaptation latency (protocol cost_accounting) and phase breakdown."""
    import nrc_records as R

    ev = R.load_events(events_path)
    ivs = R.intervals(ev)
    t_created = next(e for e in ev if e["event"] == "process_created")["t_created"]
    snaps = [r for r in ivs if r["phase"] == "snapshot_write"]
    rows = {}
    for s in snaps:
        step = int(s["key"])
        before = lambda ph: sum(r["seconds"] for r in ivs if r["phase"] == ph and r["end"] is not None and r["end"] <= s["begin"] + 1e-6)
        val_before = before("validation")
        snap_before = before("snapshot_write")
        elapsed_written = s["end"] - t_created
        rows[step] = {
            "elapsed_s": elapsed_written,
            "adaptation_latency_s": elapsed_written - val_before - snap_before,
            "optimization_s": before("optimization"),
            "path_generation_s": before("path_generation"),
            "resample_s": before("resample"),
            "dataset_init_s": before("dataset_init"),
            "model_load_s": before("model_load"),
            "mask_load_s": before("mask_load"),
            "optimizer_init_s": before("optimizer_init"),
            "validation_excluded_s": val_before,
            "snapshot_write_excluded_s": snap_before,
            "snapshot_write_s": s["seconds"],
        }
        rows[step]["process_setup_and_other_s"] = rows[step]["adaptation_latency_s"] - sum(
            rows[step][k] for k in ("optimization_s", "path_generation_s", "resample_s", "dataset_init_s",
                                    "model_load_s", "mask_load_s", "optimizer_init_s", "snapshot_write_s"))
    val = {int(r["key"]): r["end_event"] for r in ivs if r["phase"] == "validation" and r["key"] not in (None, "build")}
    return rows, val


def first_reaching(trace, key, target):
    for r in trace:
        if r[key] is not None and r[key] <= target:
            return r
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    import nrc_records as R
    import ednalib as L

    git = R.git_state(L.ROOT)
    R.require_clean(git, args.allow_dirty)
    S.init()
    import numpy as np
    import torch
    from cross_backbone_eval import refit_block
    from teaset_frozen_eval import load_rois

    root = Path(args.root)
    proto = S.protocol()
    regions = dict(np.load(root / "regions" / "regions.npz"))
    for k in list(regions):
        if regions[k].dtype == bool:
            regions[k] = regions[k].reshape(512, 512)
    oracle = np.load(root / "oracle" / "oracle_cells.npz")
    ref = HIST / "w21_reference_correction"
    gt = {s: (L.load_exr(ref / s / "gt_A.exr"), L.load_exr(ref / s / "gt_B.exr")) for s in ("T0", "T3")}
    g0, g3 = 0.5 * (gt["T0"][0] + gt["T0"][1]), 0.5 * (gt["T3"][0] + gt["T3"][1])
    rois = load_rois(HIST / "gt_design" / "v2", ("T0", "T3"))
    frozen = HIST / "frozen" / "teaset_locked"
    n0 = L.load_exr(frozen / "T0" / "8dna_upstream.exr")          # released at T0 = every warm arm's own T0 model
    n0b = L.load_exr(frozen / "T0" / "8dna_upstream_seedB.exr")
    a_att = L.load_exr(frozen / "T3" / "8dna_attached.exr")
    a_up = L.load_exr(frozen / "T3" / "8dna_upstream.exr")
    stable = regions["stable_stationary"]
    released_sd = torch.load(S.released_checkpoint(), map_location="cpu", weights_only=False)["state_dict"]
    N = 64

    # --- noise floors and static gates
    dd = lambda a, b, m: float(np.abs(disp(a) - disp(b)).mean(-1)[m].mean())
    noise = {}
    for k in ("R_aff", "R_aff_stat", "R_mover", "R_unaff"):
        m = regions[k]
        noise[k] = {"reference_repeat": dd(gt["T3"][0], gt["T3"][1], m), "neural_seed_repeat_T0": dd(n0, n0b, m)}
    mi_ = rois["T3"]["interaction"]
    noise["roi_interaction"] = {"reference_repeat": dd(gt["T0"][0], gt["T0"][1], rois["T0"]["interaction"][0].reshape(512, 512)),
                                "neural_seed_repeat_T0": dd(n0, n0b, rois["T0"]["interaction"][0].reshape(512, 512))}
    static = {
        "R_aff_stat": {"T0_model_error_same_surface": dd(n0, g0, regions["R_aff_stat"]),
                       "reference_change": dd(g3, g0, regions["R_aff_stat"])},
        "roi_interaction": refit_block(gt, rois, n0, a_att),
        "full_image_psnr_T0": L.metrics(n0, g0)["display_psnr_db"],
    }
    static["roi_interaction"]["reference_change"] = float(np.abs(disp(g3).reshape(-1, 3)[mi_[0]] - disp(g0).reshape(-1, 3)[mi_[1]]).mean())
    gates = {
        "G0_static_baseline": bool(static["R_aff_stat"]["T0_model_error_same_surface"] < static["R_aff_stat"]["reference_change"]
                                   and static["roi_interaction"]["T0_model_error_same_surface"] < static["roi_interaction"]["reference_change"]),
        "G_signal": bool(all(static[k]["reference_change"] > 3 * noise[k]["reference_repeat"] and
                             static[k]["reference_change"] > 3 * noise[k]["neural_seed_repeat_T0"]
                             for k in ("R_aff_stat", "roi_interaction"))),
    }

    def score(img):
        row = region_errors(img, g3, regions, rois)
        blk = refit_block(gt, rois, n0, img)
        row["rule"] = {k: blk[k] for k in ("error_ratio", "gain", "recovers", "refit_T3_interaction_error")}
        row["tracking"] = {k: change_tracking(img, n0, g3, g0, regions[k]) for k in ("R_aff_stat", "R_unaff")}
        return row

    out = {"schema": "sro_metrics/v1", "git": git, "noise": noise, "static": static, "gates": gates, "arms": {}}
    out["A_frozen_attached"] = score(a_att)
    out["A_step0_upstream"] = score(a_up)

    # --- historical scratch rebuild (WL27): every epoch render, instrumentation-excluded usable time
    w27 = json.loads((WL27 / "report" / "timing_8dna_envmap_primary.json").read_text())
    trace_b = []
    for q in w27["quality_trace"]:
        p = S.ROOT / q["render"]
        if p.exists():
            r = score(L.load_exr(p))
            trace_b.append({"epoch": q["epoch"], "steps": (q["epoch"] + 1) * 8192,
                            "adaptation_latency_s": q["usable_elapsed_excl_instrumentation_s"], **r})
    out["B_historical"] = {"source": "worklog 27 v1_d2d6432 (Lightning upstream train.py, seed 9)", "trace": trace_b}

    # --- this batch's runs
    pix_cells = pixel_stencil_cells(regions["x1_T3_sub"], N)
    step0_render = a_up  # released checkpoint, upstream mode at T3 (worklog 21); checked against this batch's step 0 below
    out["step0_reproduction"] = {}
    for run in sorted((root / "runs").iterdir()):
        tj = run / "train.json"
        if not tj.exists():
            continue
        tr = json.loads(tj.read_text())
        if tr.get("smoke"):
            continue
        times, val = adaptation_times(run / "events.jsonl")
        trace = []
        for step in tr["snapshot_steps"]:
            rp = run / "renders" / f"step_{step:06d}.exr"
            if not rp.exists():
                continue
            img = L.load_exr(rp)
            if step == 0 and tr["arm"] != "B":
                out["step0_reproduction"][run.name] = {"max_abs_diff_vs_wl21_upstream_T3": float(np.abs(img - a_up).max())}
            row = {"step": step, **score(img), **times.get(step, {})}
            ve = val.get(step)
            if ve:
                row["validation"] = {k: ve.get(k) for k in ("loss", "loss_touching_mask", "loss_not_touching_mask")}
            ck = run / "snapshots" / f"step_{step:06d}.ckpt"
            upd, _ = updated_cells(ck, released_sd, N)
            row["updated_cells"] = int(upd.sum())
            if tr["arm"] in ("E", "E_mask", "S", "D") and step > 0:
                touched = np.zeros(len(pix_cells), bool)
                ok = pix_cells >= 0
                touched_any = np.where(ok, upd[np.clip(pix_cells, 0, None)], False).any(1)
                frozen_pix = (~touched_any).reshape(512, 512) & ~regions["background"]
                diff = np.abs(img - step0_render).max(-1)
                row["frozen_stencil_pixels"] = {"pixels": int(frozen_pix.sum()),
                                                "changed": int((diff[frozen_pix] > 0).sum()),
                                                "max_abs_diff": float(diff[frozen_pix].max()) if frozen_pix.any() else 0.0,
                                                "R_unaff_share_frozen": float((frozen_pix & regions["R_unaff"]).sum() / max(regions["R_unaff"].sum(), 1))}
            trace.append(row)
        out["arms"][f"{tr['arm']}_s{tr['seed']}"] = {"arm": tr["arm"], "seed": tr["seed"], "train": {k: tr[k] for k in (
            "trainable_parameters", "total_parameters", "mask", "restricted_fraction_of_valid", "frozen_identity_ok",
            "parameter_changes", "torch_memory_mib", "gpu_poll", "skipped_steps", "empty_restricted_steps", "stream")},
            "trace": trace}

    # --- gates on seed-0 (and seed-1 replicate) finals
    def final(name):
        a = out["arms"].get(name)
        return a["trace"][-1] if a and a["trace"] else None

    def screening(seed):
        D, E, C = final(f"D_s{seed}"), final(f"E_s{seed}"), final(f"C_s{seed}")
        if not (D and E and C):
            return None
        A = out["A_frozen_attached"]
        red_D, red_C = A["R_aff"] - D["R_aff"], A["R_aff"] - C["R_aff"]
        g_d = bool(red_D > 3 * noise["R_aff"]["neural_seed_repeat_T0"] and red_D >= 0.5 * red_C)
        q_star = 1.10 * D["R_aff"]
        tD = first_reaching(out["arms"][f"D_s{seed}"]["trace"], "R_aff", q_star)
        tE = first_reaching(out["arms"][f"E_s{seed}"]["trace"], "R_aff", q_star)
        step0_unaff = out["A_step0_upstream"]["R_unaff"]
        n_cells = int(oracle["M95"].sum())
        fs = E.get("frozen_stencil_pixels", {})
        res = {"G_D_meaningful": g_d, "stale_reduction_D": red_D, "stale_reduction_C": red_C,
               "S1_quality": {"E": E["R_aff"], "D": D["R_aff"], "ratio": E["R_aff"] / D["R_aff"],
                              "pass": bool(E["R_aff"] <= 1.10 * D["R_aff"]) if g_d else None},
               "S2_preservation": {"E_R_unaff": E["R_unaff"], "step0_R_unaff": step0_unaff,
                                   "ratio": E["R_unaff"] / step0_unaff, "frozen_stencil_changed_pixels": fs.get("changed"),
                                   "pass": bool(E["R_unaff"] <= 1.05 * step0_unaff and fs.get("changed", 1) == 0)},
               "S3_frozen_subset": {"cells": n_cells, "fraction": n_cells / (3 * N * N), "pass": bool(n_cells <= 0.5 * 3 * N * N)},
               "S4_latency": {"Q_star": q_star,
                              "t_D": tD["adaptation_latency_s"] if tD else None, "step_D": tD["step"] if tD else None,
                              "t_E": tE["adaptation_latency_s"] if tE else None, "step_E": tE["step"] if tE else None}}
        s4 = res["S4_latency"]
        s4["speedup"] = (s4["t_D"] / s4["t_E"]) if (s4["t_D"] and s4["t_E"]) else None
        s4["pass"] = bool(s4["speedup"] is not None and s4["speedup"] >= 2.0)
        return res

    out["screening"] = {f"seed{s}": screening(s) for s in (0, 1)}
    S.write_json(root / "metrics.json", out)
    print(json.dumps({"gates": gates, "screening": out["screening"]}, indent=1, default=str)[:4000], flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(rc)
