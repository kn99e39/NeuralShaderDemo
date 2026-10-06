"""Tables, plots and evidence panels for one benchmark run.

python make_report.py <run-id> [--export <evaluation-number>]

Runs in an environment with numpy, matplotlib, Pillow and mitsuba (the
8DNA Windows venv). Reads results/full_recompute_benchmark/runs/<run-id>/,
writes results/full_recompute_benchmark/report/<run-id>/, and with
--export copies reviewer-facing files into results/evaluation/<n>/ with a
provenance manifest (sha256 of every exported file and of its sources).
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import frb_records as R  # noqa: E402

OUT_ROOT = os.path.join(ROOT, "results", "full_recompute_benchmark")
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e4e3df", "#fcfcfb"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_runs(run_dir):
    recs = {}
    for p in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
        n = os.path.basename(p)[:-5]
        if n.endswith(".condition") or n.startswith("run_manifest") or n.startswith("xl_build"):
            continue
        if n.endswith(".failed"):
            recs[n[:-7]] = {"failed": True, **R.load_json(p)}
            continue
        recs[n] = R.load_json(p)
    return recs


def row(cid, rec):
    if rec.get("failed"):
        c = rec["condition"]
        return {"condition": cid, "variant": c["variant"], "spp": c["spp"], "bounce": c["bounce_preset"],
                "persistent": c["persistent"], "status": f"FAILED rc={rec['host']['returncode']}"}
    c, s, sc = rec["condition"], rec["summary"], rec["scene"]
    st = s["steady"]
    a = sc["accounting"]
    h = rec.get("host", {})
    med = lambda k: st[k]["median"]
    return {
        "condition": cid, "variant": c["variant"], "spp": c["spp"], "bounce": c["bounce_preset"],
        "persistent": c["persistent"], "status": "ok",
        "load_s": sc["load_s"],
        "cold_frame_s": s["cold"]["frame_total_s"], "cold_setup_s": s["cold"]["setup_s"],
        "cold_sync_s": s["cold"]["sync_s"], "cold_device_update_s": s["cold"]["device_update_s"],
        "steady_n": st["frame_total_s"]["n"],
        "frame_median_s": med("frame_total_s"), "frame_p25_s": st["frame_total_s"]["p25"],
        "frame_p75_s": st["frame_total_s"]["p75"], "frame_min_s": st["frame_total_s"]["min"],
        "frame_max_s": st["frame_total_s"]["max"],
        "C_depsgraph_s": med("depsgraph_s"), "D_sync_s": med("sync_s"),
        "E_device_update_s": med("device_update_s"), "F_path_trace_s": med("path_trace_s"),
        "cycles_total_s": med("cycles_total_s"), "blender_overhead_s": med("blender_overhead_s"),
        "static_frame_median_s": s.get("static", {}).get("frame_total_s", {}).get("median"),
        "effective_fps": s["steady_fps"]["effective_fps"], "gap_30fps": s["steady_fps"]["gap_30fps"],
        "gap_60fps": s["steady_fps"]["gap_60fps"],
        "fps_cycles_only": s["steady_fps_cycles_only"]["effective_fps"],
        "fps_path_trace_only": s["steady_fps_path_trace_only"]["effective_fps"],
        "logical_triangles": a["logical_triangles"], "unique_triangles": a["unique_triangles"],
        "mesh_instances": a["mesh_instances"], "unique_evaluated_meshes": a["unique_evaluated_meshes"],
        "objects": a["scene_objects"], "materials_used": a["materials_used"], "images_used": a["images_used"],
        "emit_panels": a["emit_panel_objects"], "file_size_mb": a["file_size_bytes"] / 2**20,
        "cycles_mem_peak_mb": max((r["cycles_mem_peak_mb"] or 0) for r in rec["renders"]),
        "gpu_mem_peak_delta_mib": h.get("gpu_mem_peak_delta_mib"),
        "gpu_mem_baseline_mib": h.get("gpu_mem_baseline_mib"),
        "peak_working_set_mb": sc["memory_end"].get("peak_working_set_mb"),
        "gpu_temp_max_c": h.get("gpu_temp_max_c"), "gpu_sm_clock_median_mhz": h.get("gpu_sm_clock_median_mhz"),
        "bvh_tasks_steady": sorted({r["bvh_tasks_handled"] for r in rec["renders"] if r["phase"] == "steady"},
                                   key=lambda x: -1 if x is None else x),
    }


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURF)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_visible(False)


def _ref_lines(ax, x_right):
    for ms, lab in ((1000 / 30, "30 FPS (33.3 ms)"), (1000 / 60, "60 FPS (16.7 ms)")):
        ax.axhline(ms, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(x_right, ms * 1.08, lab, color=INK2, fontsize=8, ha="right", va="bottom")


def plot_spp(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    fig.patch.set_facecolor(SURF)
    series = [("persistent data on: frame total", True, "frame_median_s", S1),
              ("persistent data off: frame total", False, "frame_median_s", S2),
              ("persistent data on: path tracing only", True, "F_path_trace_s", S3)]
    for lab, pers, key, col in series:
        rs = sorted([r for r in rows if r["status"] == "ok" and r["variant"] == "original" and r["bounce"] == "original"
                     and r["persistent"] == pers and "composite" not in r["condition"] and "bounce" not in r["condition"]],
                    key=lambda r: r["spp"])
        if not rs:
            continue
        x = [r["spp"] for r in rs]
        y = [r[key] * 1000 for r in rs]
        ax.plot(x, y, color=col, linewidth=2, marker="o", markersize=6, label=lab, zorder=3)
        if key == "frame_median_s":
            lo = [r["frame_min_s"] * 1000 for r in rs]
            hi = [r["frame_max_s"] * 1000 for r in rs]
            ax.vlines(x, lo, hi, color=col, linewidth=1, alpha=0.6)
        ax.annotate(f"{y[-1]:.0f} ms", (x[-1], y[-1]), textcoords="offset points", xytext=(6, 0),
                    color=INK2, fontsize=8, va="center")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 4, 16, 64, 256])
    ax.set_xticklabels(["1", "4", "16", "64", "256"])
    _style(ax, "BMW27 current GI after the rear-car move: cost vs samples per pixel",
           "samples per pixel (1920x1080, OptiX, no denoising)", "steady median per frame (ms, log)")
    _ref_lines(ax, 256)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


def plot_breakdown(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), dpi=150, sharey=False)
    fig.patch.set_facecolor(SURF)
    parts = [("C depsgraph", "C_depsgraph_s", MUTED), ("D Blender->Cycles sync", "D_sync_s", S4),
             ("E device update (BVH etc.)", "E_device_update_s", S2), ("F path tracing", "F_path_trace_s", S1),
             ("Blender pipeline overhead", "blender_overhead_s", S3)]
    for ax, pers in zip(axes, (True, False)):
        rs = sorted([r for r in rows if r["status"] == "ok" and r["variant"] == "original" and r["bounce"] == "original"
                     and r["persistent"] == pers and "composite" not in r["condition"] and "bounce" not in r["condition"]],
                    key=lambda r: r["spp"])
        xs = np.arange(len(rs))
        bottom = np.zeros(len(rs))
        for lab, key, col in parts:
            v = np.array([max(r[key], 0.0) * 1000 for r in rs])
            ax.bar(xs, v, bottom=bottom, color=col, width=0.6, label=lab, edgecolor=SURF, linewidth=1)
            bottom += v
        ax.set_xticks(xs)
        ax.set_xticklabels([str(r["spp"]) for r in rs])
        _style(ax, f"persistent data {'on' if pers else 'off'}", "samples per pixel", "steady median (ms)")
        for i, b in enumerate(bottom):
            ax.text(i, b, f"{b:.0f}", ha="center", va="bottom", fontsize=7, color=INK2)
    axes[0].legend(frameon=False, fontsize=7, labelcolor=INK2, loc="upper left")
    fig.suptitle("Where the frame time goes (BMW27, after a mover change)", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


def plot_scale(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xl = [r for r in rows if r["status"] == "ok" and r["variant"].startswith("xl_")]
    if not xl:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), dpi=150)
    fig.patch.set_facecolor(SURF)
    combos = [("instanced", True, S1), ("instanced", False, S2), ("realized", True, S3), ("realized", False, S4)]
    for ax, spp in zip(axes, (64, 1)):
        for mode, pers, col in combos:
            rs = sorted([r for r in xl if r["spp"] == spp and r["persistent"] == pers and r["variant"].endswith(mode)],
                        key=lambda r: r["logical_triangles"])
            if not rs:
                continue
            ax.plot([r["logical_triangles"] / 1e6 for r in rs], [r["frame_median_s"] * 1000 for r in rs], color=col,
                    linewidth=2, marker="o", markersize=6,
                    label=f"{mode}, persistent {'on' if pers else 'off'}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        _style(ax, f"{spp} spp", "logical triangles (millions, log)", "steady median per frame (ms, log)")
        _ref_lines(ax, ax.get_xlim()[1])
    axes[0].legend(frameon=False, fontsize=7, labelcolor=INK2, loc="upper left")
    fig.suptitle("BMW Garage XL: frame cost vs scene scale (XLCamera, after a mover change)", x=0.01, ha="left",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)
    return True


def evidence(ev_dir, out_dir):
    """Stationary-mask transport change and reviewer panels."""
    import mitsuba as mi
    from PIL import Image
    mi.set_variant("scalar_rgb")

    def read(name):
        layers = dict(mi.Bitmap(os.path.join(ev_dir, name + ".exr")).split())
        rgb = np.array(layers["RenderLayer.Combined"])[..., :3].astype(np.float64)
        z = np.array(layers["RenderLayer.Depth"]).astype(np.float64)
        return rgb, z

    g0, z0 = read("G0_s0")
    g1, z1 = read("G1_s0")
    n0, zn = read("G0_s1")
    lum = lambda a: a @ np.array([0.2126, 0.7152, 0.0722])
    stationary = np.abs(z1 - z0) <= 1e-4 * np.maximum(np.abs(z0), 1e-6)
    d_change = np.abs(lum(g1) - lum(g0))
    d_noise = np.abs(lum(g0) - lum(n0))
    # 9x9 box means suppress per-pixel noise so the map shows structure, not grain.
    def box(a, k=9):
        c = np.cumsum(np.cumsum(np.pad(a, ((k // 2 + 1, k // 2), (k // 2 + 1, k // 2)), mode="edge"), 0), 1)
        return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)
    sig_change = np.abs(box(lum(g1) - lum(g0)))
    sig_noise = np.abs(box(lum(g0) - lum(n0)))
    m = stationary
    stats = {
        "pixels": int(m.size), "stationary_pixels": int(m.sum()), "stationary_fraction": float(m.mean()),
        "first_hit_changed_pixels": int((~m).sum()),
        "stationary_mean_abs_change": float(d_change[m].mean()),
        "stationary_mean_abs_noise_seed": float(d_noise[m].mean()),
        "stationary_change_to_noise": float(d_change[m].mean() / d_noise[m].mean()),
        "stationary_box9_mean_abs_change": float(sig_change[m].mean()),
        "stationary_box9_mean_abs_noise": float(sig_noise[m].mean()),
        "stationary_box9_change_to_noise": float(sig_change[m].mean() / sig_noise[m].mean()),
        "stationary_box9_frac_change_gt_3x_noise_p99": float(
            (sig_change[m] > 3 * np.percentile(sig_noise[m], 99)).mean()),
        "note": "luminance of linear Combined; stationary = depth equal in G0 and G1 (same seed, relative 1e-4)",
    }
    os.makedirs(out_dir, exist_ok=True)
    for n in ("G0_s0", "G1_s0"):
        shutil.copyfile(os.path.join(ev_dir, n + ".png"), os.path.join(out_dir, n + ".png"))
    a = Image.open(os.path.join(ev_dir, "G0_s0.png")).convert("RGB")
    b = Image.open(os.path.join(ev_dir, "G1_s0.png")).convert("RGB")
    w, h = a.size
    side = Image.new("RGB", (w, h // 2), SURF)
    side.paste(a.resize((w // 2, h // 2)), (0, 0))
    side.paste(b.resize((w // 2, h // 2)), (w // 2, 0))
    side.save(os.path.join(out_dir, "G0_G1_side_by_side.jpg"), quality=90)
    a.resize((w // 2, h // 2)).save(os.path.join(out_dir, "G0_G1_flicker.gif"), save_all=True,
                                     append_images=[b.resize((w // 2, h // 2))], duration=700, loop=0)
    # Change map: box-filtered |dL| on stationary pixels (sequential blue), first-hit-changed pixels in grey.
    scale = np.percentile(sig_change[m], 99.5)
    t = np.clip(sig_change / max(scale, 1e-9), 0, 1)
    lo, hi = np.array([0xfc, 0xfc, 0xfb]) / 255.0, np.array([0x10, 0x42, 0x81]) / 255.0
    img = lo[None, None, :] * (1 - t[..., None]) + hi[None, None, :] * t[..., None]
    img[~m] = np.array([0x89, 0x87, 0x81]) / 255.0
    Image.fromarray((img * 255).astype(np.uint8)).resize((w // 2, h // 2)).save(
        os.path.join(out_dir, "stationary_change_map.png"))
    tn = np.clip(sig_noise / max(scale, 1e-9), 0, 1)
    imgn = lo[None, None, :] * (1 - tn[..., None]) + hi[None, None, :] * tn[..., None]
    imgn[~m] = np.array([0x89, 0x87, 0x81]) / 255.0
    Image.fromarray((imgn * 255).astype(np.uint8)).resize((w // 2, h // 2)).save(
        os.path.join(out_dir, "stationary_noise_map_same_scale.png"))
    stats["change_map_scale_box9_luminance"] = float(scale)
    return stats


def md_table(rows, cols, fmt):
    out = ["| " + " | ".join(c for c, _ in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(fmt(r, k) for _, k in cols) + " |")
    return "\n".join(out)


def _fmt(r, k):
    v = r.get(k)
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "on" if v else "off"
    if isinstance(v, float):
        if k.endswith("_s"):
            return f"{v * 1000:.1f}"
        if k in ("effective_fps", "fps_cycles_only", "fps_path_trace_only"):
            return f"{v:.2f}"
        if k.startswith("gap"):
            return f"{v:.1f}x"
        return f"{v:.1f}"
    if isinstance(v, int) and v > 99999:
        return f"{v / 1e6:.2f}M"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--export")
    a = ap.parse_args()
    run_dir = os.path.join(OUT_ROOT, "runs", a.run_id)
    rep = os.path.join(OUT_ROOT, "report", a.run_id)
    os.makedirs(rep, exist_ok=True)
    recs = load_runs(run_dir)
    rows = [row(cid, rec) for cid, rec in sorted(recs.items())]
    with open(os.path.join(rep, "timing_table.csv"), "w", newline="", encoding="utf-8") as f:
        keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 999)
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()})
    plot_spp(rows, os.path.join(rep, "spp_vs_time.png"))
    plot_breakdown(rows, os.path.join(rep, "time_breakdown.png"))
    has_xl = plot_scale(rows, os.path.join(rep, "scale_vs_time.png"))
    ev = os.path.join(run_dir, "evidence_original")
    ev_stats = evidence(ev, os.path.join(rep, "evidence")) if os.path.exists(os.path.join(ev, "G1_s0.exr")) else None
    cols = [("condition", "condition"), ("spp", "spp"), ("persist.", "persistent"), ("cold ms", "cold_frame_s"),
            ("frame ms (median)", "frame_median_s"), ("min", "frame_min_s"), ("max", "frame_max_s"),
            ("C", "C_depsgraph_s"), ("D", "D_sync_s"), ("E", "E_device_update_s"), ("F", "F_path_trace_s"),
            ("overhead", "blender_overhead_s"), ("FPS", "effective_fps"), ("gap 30", "gap_30fps"),
            ("gap 60", "gap_60fps"), ("tris (logical)", "logical_triangles"), ("GPU +MiB", "gpu_mem_peak_delta_mib"),
            ("Cycles MB", "cycles_mem_peak_mb"), ("status", "status")]
    md = md_table(rows, cols, _fmt)
    with open(os.path.join(rep, "timing_table.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    xl_manifests = {os.path.basename(p): R.load_json(p)
                    for p in glob.glob(os.path.join(OUT_ROOT, "scenes", "*.manifest.json"))}
    summary = {"run_id": a.run_id, "rows": rows, "evidence": ev_stats,
               "xl_manifests": {k: {kk: v[kk] for kk in ("variant", "cars", "bays", "floor_scale_xy", "camera",
                                                           "visibility", "build_memory", "out_sha256")
                                    if kk in v} for k, v in xl_manifests.items()},
               "run_manifests": {os.path.basename(p): R.load_json(p)
                                 for p in glob.glob(os.path.join(run_dir, "run_manifest_*.json"))}}
    R.dump_json(summary, os.path.join(rep, "summary.json"))
    print(md)
    print(json.dumps(ev_stats, indent=1))
    if a.export:
        export(a, rep, run_dir, has_xl)


def export(a, rep, run_dir, has_xl):
    dst = os.path.join(ROOT, "results", "evaluation", a.export)
    if os.path.exists(dst) and os.listdir(dst):
        raise FileExistsError(f"{dst} is not empty; evaluation folders are never overwritten")
    os.makedirs(dst, exist_ok=True)
    files = [("01_spp_vs_time.png", "spp_vs_time.png"), ("02_time_breakdown.png", "time_breakdown.png"),
             ("04_timing_table.md", "timing_table.md"), ("04_timing_table.csv", "timing_table.csv"),
             ("05_summary.json", "summary.json"),
             ("06_G0.png", "evidence/G0_s0.png"), ("06_G1.png", "evidence/G1_s0.png"),
             ("07_G0_G1_side_by_side.jpg", "evidence/G0_G1_side_by_side.jpg"),
             ("07_G0_G1_flicker.gif", "evidence/G0_G1_flicker.gif"),
             ("08_stationary_change_map.png", "evidence/stationary_change_map.png"),
             ("08_stationary_noise_map_same_scale.png", "evidence/stationary_noise_map_same_scale.png")]
    if has_xl:
        files.insert(2, ("03_scale_vs_time.png", "scale_vs_time.png"))
    for p in sorted(glob.glob(os.path.join(rep, "xl_previews", "*.png"))):
        files.append(("09_" + os.path.basename(p), os.path.relpath(p, rep)))
    man = {"worklog": a.export, "run_id": a.run_id, "source_report_dir": rep, "source_run_dir": run_dir, "files": []}
    for out, src in files:
        sp = os.path.join(rep, src)
        if not os.path.exists(sp):
            continue
        shutil.copyfile(sp, os.path.join(dst, out))
        man["files"].append({"file": out, "source": sp, "sha256": sha(os.path.join(dst, out))})
    man["source_records"] = [{"file": os.path.basename(p), "sha256": sha(p)}
                             for p in sorted(glob.glob(os.path.join(run_dir, "*.json")))]
    R.dump_json(man, os.path.join(dst, "manifest.json"))
    print("exported", len(man["files"]), "files to", dst)


if __name__ == "__main__":
    main()
