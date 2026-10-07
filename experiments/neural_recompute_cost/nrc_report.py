"""Timing records, quality-vs-time traces and reviewer exports for the neural full-recompute batch.

    windows/run.ps1 ../neural_recompute_cost/nrc_report.py --run <run dir> [--worklog 27]

Analysis only: reads the chain/phase event logs and the evaluation records,
never re-times anything.  Writes one schema-checked timing record per track
(nrc_records.RECORD_FIELDS) and a summary under <run>/report/, and, with
--worklog, the reviewer exports with a SHA-256 manifest under
results/evaluation/<worklog>/.  Without --worklog (smoke runs) the exports go
to <run>/report/preview/.
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
sys.path.insert(0, str(HERE.parent / "8dna_deformation_replication"))

import nrc_records as R  # noqa: E402

TOP_8DNA = ("model_init", "dataset_init", "sanity_validation", "path_generation", "resample", "optimization", "validation",
            "checkpoint_write", "snapshot_write", "export")
TOP_RNA_TRAIN = ("data_load", "sanity_validation", "optimization", "validation", "checkpoint_write", "snapshot_write")
WL26_CONTEXT = {"scene": "BMW27, 1920x1080, Cycles OptiX, persistent data, after a rigid change (worklog 26)",
                "frame_s": {"16spp": 0.2188, "64spp": 0.763, "256spp": 2.9774},
                "label": "different scene and resolution, not quality matched: cost context only"}
COLORS = {"s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df",
          "surface": "#fcfcfb", "neutral": "#9a9893"}


def point(events, name, stage=None):
    for e in events:
        if e["event"] == name and e["kind"] == "point" and (stage is None or e["stage"] == stage):
            return e
    return None


def stage_accounting(ivs, events, stage, top, launch_t, exit_t, offset=0.0):
    """Tile a stage's wall time [launch, exit] with its top-level leaf phases.

    offset converts the process clock to the chain clock (t_chain = t_proc - offset).
    Returns phase -> seconds with spawn (launch -> interpreter start),
    process_setup, the leaves, and other, which sum to the stage wall time.
    """
    created = point(events, "process_created", stage)
    imports = point(events, "imports_done", stage)
    t_created = created["t_created"] - offset
    leaves = [r for r in ivs if r["stage"] == stage and r["phase"] in top and r["seconds"] is not None]
    leaves.sort(key=lambda r: r["begin"])
    for a, b in zip(leaves, leaves[1:]):
        if b["begin"] < a["end"] - 1e-3:
            raise ValueError(f"{stage}: top-level phases overlap: {a['phase']} {a['key']} / {b['phase']} {b['key']}")
    out = {"spawn": t_created - launch_t, "process_setup": imports["t"] - created["t_created"]}
    for p in top:
        out[p] = sum(r["seconds"] for r in leaves if r["phase"] == p)
    wall = exit_t - launch_t
    out["other"] = wall - sum(out.values())
    return out, wall


def epoch_rows(ivs, events, stage):
    rows = R.epoch_table(ivs, stage, events)
    R.check_epoch_accounting(rows)
    for r in rows:
        if r.get("checkpoint_available_t") is None and not r.get("incomplete"):
            r["checkpoint_available_t"] = r["end"]  # no write recorded in this epoch: fall back to its end
    return rows


def snapshot_cum(ivs, stage, until_t, offset=0.0):
    return sum(r["seconds"] for r in ivs if r["stage"] == stage and r["phase"] == "snapshot_write"
               and r["seconds"] is not None and r["end"] - offset <= until_t + 1e-6)


def memory(events, stage):
    m = point(events, "memory", stage)
    return {k: v for k, v in (m or {}).items() if k not in ("t", "pc", "stage", "event", "kind")}


def track_8dna(run, chain, st, proto, ev_eval, inf, git):
    events = R.load_events(run / "8dna" / "events.jsonl")
    ivs = R.intervals(events)
    s = st["8dna_train"]
    t0, t_exit = s["launch_t"], s["exit_t"]
    phases, wall = stage_accounting(ivs, events, "8dna_train", TOP_8DNA, t0, t_exit)
    rows = epoch_rows(ivs, events, "8dna_train")
    export = next(r for r in ivs if r["phase"] == "export")
    ckmap = {c["epoch"]: c for c in R.map_checkpoints(rows, t0, export_s=export["seconds"])}
    envt = point(events, "environment", "8dna_train")
    out = {}
    for regime, tkey in (("w21_envmap", "8dna_envmap_primary"), ("common_light", "8dna_common_light_secondary")):
        er = ev_eval["regimes"][regime]["rows"]
        trace = []
        for r in sorted(er, key=lambda x: x["epoch"]):
            c = ckmap[r["epoch"]]
            row_e = next(x for x in rows if x["epoch"] == r["epoch"])
            vm = (row_e.get("val_metrics") or {}).get("metrics", {})
            trace.append({"epoch": r["epoch"], "global_step": (row_e.get("val_metrics") or {}).get("global_step"),
                          "checkpoint_available_elapsed_s": c["checkpoint_available_elapsed_s"],
                          "usable_elapsed_s": c["usable_elapsed_s"],
                          "usable_elapsed_excl_instrumentation_s": c["usable_elapsed_s"] - snapshot_cum(ivs, "8dna_train", t0 + c["checkpoint_available_elapsed_s"]),
                          "error_ratio": r["error_ratio"], "refit_T3_interaction_error": r["refit_T3_interaction_error"],
                          "T0_model_error_same_surface": r["T0_model_error_same_surface"], "gain": r["gain"],
                          "validation_metric": {"val/loss": vm.get("val/loss"), "train/loss": vm.get("train/loss")},
                          "recovers": r["recovers"], "render": r["render"], "eval_render_s": r["render_s"]})
        fr = R.first_recovery(trace)
        final = trace[-1]
        total = t_exit - t0
        sel = export["end"] - t0
        inst_total = snapshot_cum(ivs, "8dna_train", t_exit)
        rec = {
            "schema": R.SCHEMA, "method": "8DNA", "track": tkey,
            "upstream_commit": envt["environment"]["upstream_commit"], "project_commit": git["commit"],
            "project_dirty": envt["git"]["dirty"],
            "configuration": {"launcher_argv": envt["launcher_argv"], "config": "upstream configs/default.yaml (no overrides)",
                              "evaluation_regime": regime, "evaluation": ev_eval["regimes"][regime]["neural"],
                              "protocol": "experiments/neural_recompute_cost/protocol/nrc_t3_recompute_v1.json"},
            "initialization": "REPRESENTATION_REBUILD (from scratch, seed 9), the historical worklog-22 path",
            "geometry_state": {"state": "T3", "translations": {"teapot2": [0.16, 0.0, 0.0]}},
            "host": chain["host"], "environment": envt["environment"],
            "start_timestamp": t0,
            "phase_durations_s": phases,
            "stages": [dict(s, stage="8dna_train")],
            "epochs": [{k: v for k, v in r.items() if k not in ("val_metrics",)} | {"val_metrics": (r.get("val_metrics") or {}).get("metrics")} for r in rows],
            "cumulative": R.cumulative(t0, {"process_created": point(events, "process_created")["t_created"],
                                            "imports_done": point(events, "imports_done")["t"],
                                            "fit_begin": next(r["begin"] for r in ivs if r["phase"] == "fit"),
                                            "fit_end": next(r["end"] for r in ivs if r["phase"] == "fit"),
                                            "export_end": export["end"], "stage_exit": t_exit}),
            "checkpoints": list(ckmap.values()),
            "quality_trace": trace,
            "first_recovery": fr,
            "time_to_first_recovery_s": None if fr is None else fr["usable_elapsed_s"],
            "time_to_historical_selection_s": sel,
            "historical_selection": "last.ckpt at the end of the 30-epoch schedule, exported",
            "total_schedule_s": total,
            "instrumentation_overhead_s": {"snapshot_write_total": inst_total},
            "total_schedule_excl_instrumentation_s": total - inst_total,
            "inference": inf["8dna"][regime],
            "memory": memory(events, "8dna_train"),
            "recovery_verdict": {
                "historical_selection_recovers": final["recovers"], "historical_selection_error_ratio": final["error_ratio"],
                "historical_selection_gain": final["gain"],
                "historical_values_worklog22": ev_eval["regimes"][regime]["historical"],
                "first_recovered_epoch": None if fr is None else fr["epoch"],
                "checkpoints_recovering": sum(t["recovers"] for t in trace), "checkpoints_evaluated": len(trace),
                "reproduction_gate": ("PASS: the rebuild's last.ckpt satisfies the historical rule" if final["recovers"] else
                                      "FAIL: the rebuild's last.ckpt does not satisfy the historical rule")
                if regime == "w21_envmap" else "n/a (secondary regime; historical common-light refit was case B by the rule)"},
            "mapping_check": ev_eval["mapping"],
        }
        rec["budget"] = budget_block(rec)
        R.validate_record(rec, smoke=chain["smoke"])
        out[tkey] = rec
    return out


def budget_block(rec):
    def b(x):
        return None if x is None else {"seconds": x, "category": R.latency_category(x), **R.frame_budget_ratios(x)}

    return {"time_to_first_recovery": b(rec["time_to_first_recovery_s"]),
            "time_to_historical_selection": b(rec["time_to_historical_selection_s"]),
            "total_schedule": b(rec["total_schedule_s"])}


def track_rna(run, chain, chain_ev, st, proto, ev_eval, inf, git):
    h5 = R.load_events(run / "rna" / "events.jsonl")
    tr = R.load_events(run / "rna" / "events_train.jsonl")
    offs = [e for e in chain_ev if e["event"] == "clock_offset" and e.get("ok")]
    offset = sum(e["offset_s"] for e in offs) / len(offs) if offs else 0.0
    ivs_h5, ivs_tr = R.intervals(h5), R.intervals(tr)
    t0 = st["rna_h5_train"]["launch_t"]
    stages, phases = [], {}
    for name in ("rna_h5_train", "rna_h5_val"):
        s = st[name]
        ph, wall = stage_accounting(ivs_h5, h5, name, ("view_render",), s["launch_t"], s["exit_t"])
        gen = next(r for r in ivs_h5 if r["stage"] == name and r["phase"] == "generate_h5")
        first_view = min(r["begin"] for r in ivs_h5 if r["stage"] == name and r["phase"] == "view_render")
        ph["scene_prep_derived"] = first_view - gen["begin"]
        ph["other"] -= ph["scene_prep_derived"]
        views = [r["seconds"] for r in ivs_h5 if r["stage"] == name and r["phase"] == "view_render"]
        stages.append(dict(s, stage=name, phases=ph, views=len(views), view_render_s_mean=sum(views) / len(views),
                           view_render_s_min=min(views), view_render_s_max=max(views)))
        for k, v in ph.items():
            phases[f"{name}.{k}"] = v
    s = st["rna_train"]
    done = point(tr, "process_done")["t"] - offset
    ph, wall = stage_accounting(ivs_tr, tr, "rna_train", TOP_RNA_TRAIN, s["launch_t"], done, offset)
    ph["status_poll_and_teardown"] = s["exit_t"] - done
    stages.append(dict(s, stage="rna_train", phases=ph, clock_offset_wsl_minus_windows_s=offset,
                       clock_offset_samples=[{k: e[k] for k in ("when", "offset_s", "round_trip_s")} for e in offs]))
    for k, v in ph.items():
        phases[f"rna_train.{k}"] = v
    gaps = {f"gap_before_{x['stage']}": x["gap_before_s"] for x in stages if x.get("gap_before_s") is not None}
    phases.update(gaps)
    rows = epoch_rows(ivs_tr, tr, "rna_train")
    ckmap = {c["epoch"]: c for c in R.map_checkpoints(rows, t0, offset_s=-offset)}
    trace = []
    for r in sorted(ev_eval["rows"], key=lambda x: (x["epoch"], x["kind"])):
        c = ckmap.get(r["epoch"])
        row_e = next((x for x in rows if x["epoch"] == r["epoch"]), {})
        vm = (row_e.get("val_metrics") or {}).get("metrics", {})
        trace.append({"epoch": r["epoch"], "kind": r["kind"], "tag": r["tag"], "global_step": r["global_step"],
                      "checkpoint_available_elapsed_s": None if c is None else c["checkpoint_available_elapsed_s"],
                      "usable_elapsed_s": None if c is None else c["usable_elapsed_s"],
                      "usable_elapsed_excl_instrumentation_s": None if c is None else
                      c["usable_elapsed_s"] - snapshot_cum(ivs_tr, "rna_train", t0 + c["checkpoint_available_elapsed_s"], offset),
                      "error_ratio": r["error_ratio"], "refit_T3_interaction_error": r["refit_T3_interaction_error"],
                      "T0_model_error_same_surface": r["T0_model_error_same_surface"], "gain": r["gain"],
                      "validation_metric": {"val_psnr": vm.get("val_psnr"), "val_loss": vm.get("val_loss")},
                      "recovers": r["recovers"], "is_validation_best": r["is_validation_best"], "render": r["render"],
                      "rna_infer_call_s": r["rna_infer_call_s"]})
    fr = R.first_recovery(trace)
    best = next(t for t in trace if t["is_validation_best"])
    total = st["rna_train"]["exit_t"] - t0
    sel = done - t0  # validation-best is known only once the schedule has ended
    inst = snapshot_cum(ivs_tr, "rna_train", st["rna_train"]["exit_t"], offset)
    envt = point(tr, "environment")
    envh = point(h5, "environment")
    rec = {
        "schema": R.SCHEMA, "method": "RNA", "track": "rna_common_light",
        "upstream_commit": envt.get("rna_commit"), "project_commit": git["commit"], "project_dirty": envh["git"]["dirty"],
        "configuration": {"train_argv": envt["train_argv"], "dataset_argv": envh["argv"],
                          "protocol": "experiments/neural_recompute_cost/protocol/nrc_t3_recompute_v1.json",
                          "derived_config_diff": next(e for e in chain_ev if e["event"] == "rna_derived_inputs")["config_diff"]},
        "initialization": "REPRESENTATION_REBUILD (from scratch, seed 0), the historical worklog-22 path",
        "geometry_state": {"state": "T3", "translations": {"teapot2": [0.16, 0.0, 0.0]}},
        "host": chain["host"], "environment": {"dataset": envh["environment"], "training": {k: envt[k] for k in envt if k not in ("t", "pc", "stage", "event", "kind")}},
        "start_timestamp": t0, "phase_durations_s": phases, "stages": stages,
        "epochs": [{k: v for k, v in r.items() if k != "val_metrics"} | {"val_metrics": (r.get("val_metrics") or {}).get("metrics")} for r in rows],
        "cumulative": R.cumulative(t0, {"h5_train_exit": st["rna_h5_train"]["exit_t"], "h5_val_exit": st["rna_h5_val"]["exit_t"],
                                        "train_launch": st["rna_train"]["launch_t"], "train_process_done": done,
                                        "stage_exit": st["rna_train"]["exit_t"]}),
        "checkpoints": list(ckmap.values()), "quality_trace": trace, "first_recovery": fr,
        "time_to_first_recovery_s": None if fr is None else fr["usable_elapsed_s"],
        "time_to_historical_selection_s": sel,
        "historical_selection": f"validation-best val_psnr checkpoint ({best['tag']}), known when the 250-epoch schedule ends",
        "total_schedule_s": total,
        "instrumentation_overhead_s": {"snapshot_write_total": inst},
        "total_schedule_excl_instrumentation_s": total - inst,
        "inference": inf["rna"],
        "memory": {"rna_h5_train": memory(h5, "rna_h5_train"), "rna_h5_val": memory(h5, "rna_h5_val"), "rna_train": memory(tr, "rna_train")},
        "recovery_verdict": {"historical_selection_recovers": best["recovers"], "historical_selection_error_ratio": best["error_ratio"],
                             "historical_selection_gain": best["gain"], "historical_values_worklog22": ev_eval["historical"],
                             "first_recovered_epoch": None if fr is None else fr["epoch"],
                             "checkpoints_recovering": sum(bool(t["recovers"]) for t in trace), "checkpoints_evaluated": len(trace),
                             "statement": ("recovered checkpoint within the fixed schedule" if fr else
                                           "no recovered checkpoint within the fixed schedule")},
        "static_quality_caveat": "RNA's static reconstruction of this near-mirror material is weak (worklog 22: G1 margin 4%; worklogs 23-24); its recovery reading is supporting, not co-equal, evidence.",
    }
    rec["budget"] = budget_block(rec)
    R.validate_record(rec, smoke=chain["smoke"])
    return rec


def dataset_compare(run, L):
    """Regenerated RNA H5 vs the historical worklog-22 files, channel by channel."""
    import h5py
    import numpy as np

    out = {}
    for split in ("train", "val"):
        a_p = run / "rna" / "datasets" / f"teaset_T3_{split}.h5"
        b_p = L.RESULTS / "rna_teaset" / "datasets" / f"teaset_T3_{split}.h5"
        res = {"new": L.rel(a_p), "historical": L.rel(b_p), "channels": {}}
        with h5py.File(a_p, "r") as a, h5py.File(b_p, "r") as b:
            for k in sorted(set(a) | set(b)):
                if k not in a or k not in b or a[k].shape != b[k].shape:
                    res["channels"][k] = {"comparable": False}
                    continue
                n = a[k].shape[0]
                maxd, ndiff, tot, ident_views = 0.0, 0, 0, 0
                for v in range(0, n, 8):
                    x = a[k][v:v + 8].astype(np.float32)
                    y = b[k][v:v + 8].astype(np.float32)
                    d = np.abs(x - y)
                    maxd = max(maxd, float(np.nanmax(d)) if d.size else 0.0)
                    ndiff += int((d > 0).sum())
                    tot += d.size
                    ident_views += int(sum(np.array_equal(x[i], y[i]) for i in range(len(x))))
                res["channels"][k] = {"max_abs_diff": maxd, "fraction_values_differing": ndiff / tot, "identical_views": ident_views,
                                      "views": n}
            res["attrs_equal_except_commit"] = {k: str(a.attrs[k]) == str(b.attrs[k]) for k in a.attrs if k != "project_commit"}
        out[split] = res
    return out


# --- figures -----------------------------------------------------------------

def fig_quality_time(path, tracks, plt):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True, facecolor=COLORS["surface"])
    series = [("8DNA envmap (primary)", tracks["8dna_envmap_primary"], COLORS["s1"]),
              ("8DNA common light (secondary)", tracks["8dna_common_light_secondary"], COLORS["s2"]),
              ("RNA common light", tracks["rna_common_light"], COLORS["s3"])]
    for ax, key, thr, lab in ((axes[0], "error_ratio", 1.25, "error ratio  (refit T3 error / T0-model error)"),
                              (axes[1], "gain", 0.5, "tracking gain  <dN, dG> / <dG, dG>")):
        ax.set_facecolor(COLORS["surface"])
        for name, rec, col in series:
            pts = sorted((t["usable_elapsed_s"] / 3600, t[key], t["recovers"]) for t in rec["quality_trace"] if t["usable_elapsed_s"] is not None)
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            ax.plot(xs, ys, color=col, lw=2, label=name, zorder=3)
            ax.scatter([p[0] for p in pts if p[2]], [p[1] for p in pts if p[2]], s=64, color=col, edgecolor=COLORS["ink"],
                       linewidth=1.0, zorder=5)
            ax.scatter([p[0] for p in pts if not p[2]], [p[1] for p in pts if not p[2]], s=36, facecolor=COLORS["surface"],
                       edgecolor=col, linewidth=1.5, zorder=4)
        ax.axhline(thr, color=COLORS["ink2"], lw=1, ls=(0, (4, 3)), zorder=2)
        ax.text(0.80, thr, f"historical rule: {'<=' if key == 'error_ratio' else '>='} {thr}", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", color=COLORS["ink2"], fontsize=9,
                bbox={"facecolor": COLORS["surface"], "edgecolor": "none", "pad": 1.5})
        ax.set_ylabel(lab, color=COLORS["ink2"], fontsize=9.5)
        ax.grid(True, color=COLORS["grid"], lw=0.8, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(COLORS["neutral"])
        ax.tick_params(colors=COLORS["ink2"], labelsize=9)
    ymax = max(t["error_ratio"] for _, r, _ in series for t in r["quality_trace"])
    axes[0].set_yscale("log")
    axes[0].set_ylim(0.8, max(2.0, 1.15 * ymax))
    from matplotlib.ticker import FixedLocator, FuncFormatter

    top = max(2.0, 1.15 * ymax)
    axes[0].yaxis.set_major_locator(FixedLocator([v for v in (1, 1.25, 1.5, 2, 3, 5, 10, 20, 50) if v <= top]))
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0].yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    axes[1].set_xlabel("elapsed wall time since T3 geometry handed to the pipeline (h)", color=COLORS["ink2"], fontsize=9.5)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    axes[0].set_title("Quality vs wall-clock time of the neural rebuild on T3  (filled, outlined marker = historical rule satisfied;\n"
                      "vertical line = end of the fixed schedule)",
                      color=COLORS["ink"], fontsize=11, loc="left")
    for name, rec, col in series:
        for ax in axes:
            ax.axvline(rec["total_schedule_s"] / 3600, color=col, lw=1, alpha=0.5, zorder=1)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=COLORS["surface"])
    plt.close(fig)


def fig_breakdown(path, tracks, plt):
    t8 = tracks["8dna_envmap_primary"]["phase_durations_s"]
    tr = tracks["rna_common_light"]["phase_durations_s"]
    groups = [
        ("8DNA rebuild", [("setup + scene", t8["spawn"] + t8["process_setup"] + t8["model_init"] + t8["dataset_init"]),
                          ("path / target generation", t8["path_generation"] + t8["resample"]),  # 8DNA: online path samples + their shuffle
                          ("optimisation", t8["optimization"]),
                          ("validation", t8["validation"] + t8["sanity_validation"]),
                          ("checkpoint + export", t8["checkpoint_write"] + t8["export"]),
                          ("other (incl. instrumentation)", t8["snapshot_write"] + t8["other"])]),
        ("RNA rebuild", [("setup + scene", sum(v for k, v in tr.items() if k.endswith((".spawn", ".process_setup", ".scene_prep_derived"))) +
                          tr.get("rna_train.data_load", 0)),
                         ("path / target generation", tr["rna_h5_train.view_render"] + tr["rna_h5_val.view_render"]),
                         ("optimisation", tr["rna_train.optimization"]),
                         ("validation", tr["rna_train.validation"] + tr["rna_train.sanity_validation"]),
                         ("checkpoint + export", tr["rna_train.checkpoint_write"]),
                         ("other (incl. instrumentation)", sum(v for k, v in tr.items() if k.endswith((".other", "snapshot_write", "status_poll_and_teardown")) or k.startswith("gap_")))]),
    ]
    cols = [COLORS["neutral"], COLORS["s2"], COLORS["s1"], COLORS["s3"], "#eda100", "#c9c7c1"]
    fig, ax = plt.subplots(figsize=(10, 3.4), facecolor=COLORS["surface"])
    ax.set_facecolor(COLORS["surface"])
    for yi, (name, parts) in enumerate(groups):
        left = 0.0
        for (lab, v), c in zip(parts, cols):
            v_h = v / 3600
            ax.barh(yi, v_h, left=left, color=c, edgecolor=COLORS["surface"], linewidth=2, height=0.55, label=lab if yi == 0 else None)
            if v_h > 0.08:
                ax.text(left + v_h / 2, yi, f"{v / 60:.0f} min", ha="center", va="center", fontsize=8.5, color=COLORS["ink"])
            left += v_h
        ax.text(left, yi, f"  {left:.2f} h", va="center", fontsize=9, color=COLORS["ink"])
    ax.set_yticks(range(len(groups)), [g[0] for g in groups], color=COLORS["ink"])
    ax.set_xlabel("wall time (h), T3 handed to pipeline -> end of the fixed schedule", color=COLORS["ink2"], fontsize=9.5)
    ax.invert_yaxis()
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=COLORS["ink2"], labelsize=9)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    ax.set_title("Where the rebuild time goes (fixed historical schedules)", color=COLORS["ink"], fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=COLORS["surface"])
    plt.close(fig)


def fig_historical(path, tracks, hist, plt):
    """Error ratio by epoch: this rebuild vs the worklog-22 run's surviving checkpoints (same evaluation path)."""
    # the historical 8DNA last.ckpt holds epoch-28 weights (step 237568; the rescore labels it 29): a duplicate, dropped
    h8 = lambda rows: [r for r in rows if not r["checkpoint"].endswith("last.ckpt")]
    panels = [("8DNA envmap", tracks["8dna_envmap_primary"]["quality_trace"], h8(hist["8dna"]["w21_envmap"]["rows"])),
              ("8DNA common light", tracks["8dna_common_light_secondary"]["quality_trace"], h8(hist["8dna"]["common_light"]["rows"])),
              ("RNA common light", [t for t in tracks["rna_common_light"]["quality_trace"] if t["kind"] == "snapshot"]
               + [t for t in tracks["rna_common_light"]["quality_trace"] if t["kind"] == "official"],
               [r for r in hist["rna"]["rows"] if r["epoch"] is not None])]  # last.ckpt's epoch is not in its name; omitted
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True, facecolor=COLORS["surface"])
    for ax, (title, new, old) in zip(axes, panels):
        ax.set_facecolor(COLORS["surface"])
        for rows, col, lab in ((new, COLORS["s1"], "this rebuild (RTX 5080)"),
                               (old, COLORS["s2"], "worklog-22 run, surviving checkpoints")):
            pts = sorted({(r["epoch"], r["error_ratio"], r["recovers"]) for r in rows})
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=1.5, label=lab, zorder=3)
            ax.scatter([p[0] for p in pts if p[2]], [p[1] for p in pts if p[2]], s=40, color=col, edgecolor=COLORS["ink"], lw=0.8, zorder=5)
            ax.scatter([p[0] for p in pts if not p[2]], [p[1] for p in pts if not p[2]], s=40, facecolor=COLORS["surface"],
                       edgecolor=col, lw=1.5, zorder=4)
        ax.axhline(1.25, color=COLORS["ink2"], lw=1, ls=(0, (4, 3)))
        ax.set_title(title, color=COLORS["ink"], fontsize=10, loc="left")
        ax.set_xlabel("epoch", color=COLORS["ink2"], fontsize=9)
        ax.grid(True, color=COLORS["grid"], lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=COLORS["ink2"], labelsize=8.5)
    axes[0].set_ylabel("error ratio (rule: <= 1.25, dashed)", color=COLORS["ink2"], fontsize=9)
    axes[0].set_ylim(0.95, 2.2)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle("Same rule, same evaluation path: this rebuild vs the worklog-22 refit run (filled = rule satisfied incl. gain >= 0.5)",
                 color=COLORS["ink"], fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=COLORS["surface"])
    plt.close(fig)


def review_panel(path, gif_path, L, gt, frozen, first, final, labels, mask, np):
    from cross_backbone_exports import crop_box, upscale

    ys, xs = crop_box(mask)
    ldr = lambda im: L.to_u8(L.tonemap(im))
    err = lambda a: L.error_map(np.abs(L.tonemap(a) - L.tonemap(gt)).mean(-1), 0.2)
    imgs = [gt, frozen, first, final]
    top, crops, errs = [], [], []
    for im, lab in zip(imgs, labels):
        if im is None:
            blank = np.full((512, 512, 3), 40, np.uint8)
            top.append(L.label(blank, lab))
            crops.append(L.label(upscale(blank[ys, xs], 512), lab))
            errs.append(L.label(upscale(blank[ys, xs], 512), "n/a"))
            continue
        top.append(L.label(ldr(im), lab))
        crops.append(L.label(upscale(ldr(im)[ys, xs], 512), lab + " (ROI crop)"))
        errs.append(L.label(upscale(err(im)[ys, xs], 512), "|display diff| vs GT, 0..0.2"))
    w = top[0].shape[1]
    row2 = np.concatenate(crops, 1)
    row3 = np.concatenate(errs, 1)
    pad = lambda r: np.pad(r, ((0, 0), (0, max(0, 4 * w - r.shape[1])), (0, 0)), constant_values=255)[:, :4 * w]
    L.save_png(path, np.concatenate([np.concatenate(top, 1), pad(row2), pad(row3)], 0))
    L.save_gif(gif_path, [c for c, im in zip(crops, imgs) if im is not None], ms=900)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--worklog", default=None)
    args = ap.parse_args()
    run = Path(args.run).resolve()
    import numpy as np

    import ednalib as L

    L.init_upstream()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    proto = json.loads((HERE / "protocol" / "nrc_t3_recompute_v1.json").read_text(encoding="utf-8"))
    chain_ev = R.load_events(run / "chain_events.jsonl")
    chain = point(chain_ev, "chain_start")
    st = {s["stage"]: s for s in R.stage_table(chain_ev)}
    git = chain["git"]
    ev8 = json.loads((run / "eval" / "8dna_eval.json").read_text(encoding="utf-8"))
    evr = json.loads((run / "eval" / "rna_eval.json").read_text(encoding="utf-8"))
    inf = json.loads((run / "eval" / "inference_timing.json").read_text(encoding="utf-8"))
    tracks = track_8dna(run, chain, st, proto, ev8, inf, git)
    tracks["rna_common_light"] = track_rna(run, chain, chain_ev, st, proto, evr, inf, git)
    dc_path = run / "eval" / "dataset_compare.json"
    if not dc_path.exists():
        R.write_json(dc_path, dataset_compare(run, L))
    dcmp = json.loads(dc_path.read_text(encoding="utf-8"))
    rep = run / "report"
    for k, rec in tracks.items():
        R.write_json(rep / f"timing_{k}.json", rec)
    summary = {
        "run": L.rel(run), "project_commit": git["commit"], "smoke": chain["smoke"],
        "tracks": {k: {"time_to_first_recovery_s": r["time_to_first_recovery_s"],
                       "first_recovered_epoch": r["recovery_verdict"]["first_recovered_epoch"],
                       "time_to_historical_selection_s": r["time_to_historical_selection_s"],
                       "historical_selection_recovers": r["recovery_verdict"]["historical_selection_recovers"],
                       "total_schedule_s": r["total_schedule_s"],
                       "total_schedule_excl_instrumentation_s": r["total_schedule_excl_instrumentation_s"],
                       "budget": r["budget"], "phases_s": r["phase_durations_s"]} for k, r in tracks.items()},
        "inference": inf, "dataset_reproducibility": dcmp, "worklog26_context": WL26_CONTEXT,
    }
    R.write_json(rep / "summary.json", summary)

    out = (L.ROOT / "results" / "evaluation" / args.worklog) if args.worklog else rep / "preview"
    out.mkdir(parents=True, exist_ok=True)
    files = {}

    def reg(name, sources):
        files[name] = {"sources": [s if isinstance(s, str) else L.rel(s) for s in sources]}

    fig_quality_time(out / "01_quality_vs_wall_time.png", tracks, plt)
    reg("01_quality_vs_wall_time.png", [rep / f"timing_{k}.json" for k in tracks])
    fig_breakdown(out / "02_phase_breakdown.png", tracks, plt)
    reg("02_phase_breakdown.png", [rep / "timing_8dna_envmap_primary.json", rep / "timing_rna_common_light.json"])
    diag = sorted((run / "diag").glob("historical_rescore_*/historical_rescore.json"))
    if diag:
        fig_historical(out / "09_historical_vs_rebuild_by_epoch.png", tracks, json.loads(diag[-1].read_text(encoding="utf-8")), plt)
        reg("09_historical_vs_rebuild_by_epoch.png", [diag[-1]] + [rep / f"timing_{k}.json" for k in tracks])

    # review panels: GT T3 / frozen T0-trained model at T3 / first recovered / final
    from teaset_frozen_eval import load_rois

    specs = [
        ("03", "8dna_envmap_primary", "8DNA envmap", "w21_reference_correction", "frozen/teaset_locked/T3/8dna_attached.exr", "gt_design/v2"),
        ("04", "8dna_common_light_secondary", "8DNA common light", "frozen/teaset_common_light_8dna",
         "frozen/teaset_common_light_8dna/T3/8dna_attached.exr", "gt_design/common_light"),
        ("05", "rna_common_light", "RNA common light", "frozen/teaset_common_light_8dna", "rna_teaset/frozen/T3_canonical.npy",
         "gt_design/common_light"),
    ]
    load = lambda p: np.load(p).astype(np.float32) if str(p).endswith(".npy") else L.load_exr(p)
    for num, key, title, gdir, frozen_p, rdir in specs:
        rec = tracks[key]
        gA, gB = L.RESULTS / gdir / "T3" / "gt_A.exr", L.RESULTS / gdir / "T3" / "gt_B.exr"
        gt = 0.5 * (L.load_exr(gA) + L.load_exr(gB))
        frozen = load(L.RESULTS / frozen_p)
        fr = rec["first_recovery"]
        final = (next(t for t in rec["quality_trace"] if t.get("is_validation_best")) if key == "rna_common_light"
                 else rec["quality_trace"][-1])
        first_img = None if fr is None else load(L.ROOT / fr["render"])
        z = np.load(L.RESULTS / rdir / "rois_T3.npz")
        mask = z["mask_interaction"] | z["mask_mover"] | np.load(L.RESULTS / rdir / "rois_T0.npz")["mask_mover"]
        hm = lambda s: f"{s / 60:.1f} min" if s < 3600 else f"{s / 3600:.2f} h"
        labels = [f"{title}: GT T3", "frozen T0-trained model at T3",
                  "no recovered checkpoint" if fr is None else f"first recovered: ep {fr['epoch']} @ {hm(fr['usable_elapsed_s'])}",
                  f"final ({'val-best' if key == 'rna_common_light' else 'last.ckpt'}): ep {final['epoch']}"
                  f"{'' if final['usable_elapsed_s'] is None else ' @ ' + hm(final['usable_elapsed_s'])}"]
        name = f"{num}_review_{key}.png"
        review_panel(out / name, out / f"{num}_review_{key}_crop_cycle.gif", L, gt, frozen, first_img, load(L.ROOT / final["render"]),
                     labels, mask, np)
        srcs = [gA, gB, L.RESULTS / frozen_p] + ([L.ROOT / fr["render"]] if fr else []) + [L.ROOT / final["render"]]
        reg(name, srcs)
        reg(f"{num}_review_{key}_crop_cycle.gif", srcs)

    # tables
    lines = ["| track | first recovery | historical selection usable | total fixed schedule | x 33.3 ms (total) | x 16.7 ms (total) | category (first / total) |",
             "|---|---|---|---|---|---|---|"]
    for k, r in tracks.items():
        b = r["budget"]
        f = b["time_to_first_recovery"]
        fmt = lambda s: "none" if s is None else (f"{s:.0f} s ({s / 60:.1f} min)" if s < 3600 else f"{s:.0f} s ({s / 3600:.2f} h)")
        lines.append(f"| {k} | {fmt(r['time_to_first_recovery_s'])}"
                     f"{'' if r['first_recovery'] is None else ' (ep ' + str(r['first_recovery']['epoch']) + ')'} | "
                     f"{fmt(r['time_to_historical_selection_s'])} (recovers: {r['recovery_verdict']['historical_selection_recovers']}) | "
                     f"{fmt(r['total_schedule_s'])} | {b['total_schedule']['x_30fps_33.3ms']:.3g} | {b['total_schedule']['x_60fps_16.7ms']:.3g} | "
                     f"{'-' if f is None else f['category']} / {b['total_schedule']['category']} |")
    (out / "06_latency_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    reg("06_latency_table.md", [rep / f"timing_{k}.json" for k in tracks])
    for k in tracks:
        shutil.copy2(rep / f"timing_{k}.json", out / f"07_timing_{k}.json")
        reg(f"07_timing_{k}.json", [rep / f"timing_{k}.json"])
    shutil.copy2(rep / "summary.json", out / "08_summary.json")
    reg("08_summary.json", [rep / "summary.json"])
    for name in files:
        files[name]["sha256"] = hashlib.sha256((out / name).read_bytes()).hexdigest()
    manifest = {"worklog": args.worklog, "run": L.rel(run), "protocol": "experiments/neural_recompute_cost/protocol/nrc_t3_recompute_v1.json",
                "project_commit_of_runs": git["commit"], "report_commit": L.git_head(L.ROOT), "report_dirty": L.git_dirty(L.ROOT),
                "display": "clamp(x^(1/2.2),0,1)", "error_maps": "per-pixel mean |display difference| vs GT, inferno, fixed 0..0.2",
                "note": "review panels: GT T3 (mean of seeds A/B) | frozen T0-trained model at T3 | first checkpoint satisfying the "
                        "historical worklog-22 recovery rule | the historical selection (8DNA last.ckpt, RNA validation-best)",
                "files": files}
    R.write_json(out / "manifest.json", manifest)
    print(json.dumps(summary["tracks"], indent=1, default=str)[:4000])
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
