"""Reviewer exports for worklog 30 (presentation only; reads existing outputs, computes no new verdict).

    windows/run.ps1 ../selective_refit_oracle/sro_exports.py --root <batch results dir> --dest results/evaluation/30

Panels use the display transform clamp(x^(1/2.2)) at one exposure; error and change maps use
fixed, locked scales.  Every exported file is listed in manifest.json with its sources and SHA-256.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import sro_common as S

# reference palette (dataviz skill, light mode), fixed slot order by arm
COLORS = {"C": "#2a78d6", "D": "#eb6834", "E": "#1baf7a", "S": "#eda100", "B": "#e87ba4", "B_hist": "#4a3aa7"}
LABELS = {"C": "C global warm", "D": "D global local-state", "E": "E oracle selective", "S": "S spatial equal-size",
          "B": "B scratch (matched)", "B_hist": "B scratch (WL27, 30 ep)"}
ERR_VMAX = 0.2
CHG_VMAX = 0.25


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--dest", required=True)
    args = ap.parse_args()
    import nrc_records as R
    import ednalib as L

    S.init()
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(args.root)
    dest = (S.ROOT / args.dest) if not Path(args.dest).is_absolute() else Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    met = json.loads((root / "metrics.json").read_text())
    reg = np.load(root / "regions" / "regions.npz")
    orc = np.load(root / "oracle" / "oracle_cells.npz")
    manifest = {"worklog": 30, "root": L.rel(root), "git": R.git_state(L.ROOT), "files": {}}

    def record(path, sources, note=""):
        manifest["files"][path.name] = {"sha256": L.sha256(path), "sources": [L.rel(Path(s)) if Path(s).is_absolute() else s for s in sources], "note": note}

    hist = S.ROOT / "results" / "8dna_replication"
    refd = hist / "w21_reference_correction"
    g0 = 0.5 * (L.load_exr(refd / "T0" / "gt_A.exr") + L.load_exr(refd / "T0" / "gt_B.exr"))
    g3 = 0.5 * (L.load_exr(refd / "T3" / "gt_A.exr") + L.load_exr(refd / "T3" / "gt_B.exr"))
    n0 = L.load_exr(hist / "frozen" / "teaset_locked" / "T0" / "8dna_upstream.exr")
    imgs = {"GT T0": (g0, [refd / "T0" / "gt_A.exr", refd / "T0" / "gt_B.exr"]),
            "GT T3": (g3, [refd / "T3" / "gt_A.exr", refd / "T3" / "gt_B.exr"]),
            "A frozen (attached)": (L.load_exr(hist / "frozen" / "teaset_locked" / "T3" / "8dna_attached.exr"),
                                    [hist / "frozen" / "teaset_locked" / "T3" / "8dna_attached.exr"])}
    bh = met["B_historical"]["trace"][-1]
    bh_path = S.ROOT / "results" / "neural_recompute_cost" / "v1_d2d6432" / "eval" / "8dna" / "w21_envmap" / f"epoch_{bh['epoch']:03d}.exr"
    imgs[f"B scratch WL27 ep{bh['epoch']}"] = (L.load_exr(bh_path), [bh_path])
    finals = {}
    for arm in ("B", "C", "D", "E", "S"):
        a = met["arms"].get(f"{arm}_s0")
        if not a or not a["trace"]:
            continue
        step = a["trace"][-1]["step"]
        p = root / "runs" / f"{arm}_s0" / "renders" / f"step_{step:06d}.exr"
        imgs[LABELS[arm]] = (L.load_exr(p), [p])
        finals[arm] = imgs[LABELS[arm]][0]

    disp = L.tonemap
    order = list(imgs)

    # 01 full-frame T3 comparison with error maps
    tiles_img, tiles_err = [], []
    for k in order:
        im = imgs[k][0]
        tiles_img.append(L.label(L.to_u8(disp(im)), k))
        e = np.abs(disp(im) - disp(g3)).mean(-1) if k not in ("GT T0", "GT T3") else np.zeros(im.shape[:2])
        tiles_err.append(L.label(L.error_map(e, ERR_VMAX), "|err| vs GT T3" if k not in ("GT T0", "GT T3") else "-"))
    half = (len(order) + 1) // 2
    pad = lambda t: t + [np.zeros_like(t[0])] * (half - len(t))
    rows = [np.concatenate(pad(tiles_img[:half]), 1), np.concatenate(pad(tiles_err[:half]), 1),
            np.concatenate(pad(tiles_img[half:]), 1), np.concatenate(pad(tiles_err[half:]), 1)]
    p = dest / "01_T3_full_frame_and_error.png"
    L.save_png(p, np.concatenate(rows, 0))
    record(p, [s for k in order for s in imgs[k][1]], f"display MAE error maps, inferno, 0..{ERR_VMAX}")

    # crop box around the interaction ROI and the mover
    m_int = reg["R_aff_stat"] | reg["R_mover"]
    ys, xs = np.nonzero(m_int)
    y0, y1 = max(ys.min() - 16, 0), min(ys.max() + 16, 511)
    x0, x1 = max(xs.min() - 16, 0), min(xs.max() + 16, 511)
    crop = lambda a: a[y0:y1, x0:x1]
    up = lambda a: np.kron(a, np.ones((2, 2, 1), a.dtype)) if a.ndim == 3 else np.kron(a, np.ones((2, 2), a.dtype))

    # 02 crops: images and errors
    ci = [L.label(up(crop(L.to_u8(disp(imgs[k][0])))), k) for k in order]
    ce = [L.label(up(crop(L.error_map(np.abs(disp(imgs[k][0]) - disp(g3)).mean(-1), ERR_VMAX))), "err") for k in order]
    p = dest / "02_affected_crop_images_and_errors.png"
    L.save_png(p, np.concatenate([np.concatenate(ci, 1), np.concatenate(ce, 1)], 0))
    record(p, [s for k in order for s in imgs[k][1]], "crop = bounding box of R_aff_stat and R_mover (+16 px), 2x")

    # 03 signed T0 -> T3 change on stable stationary pixels (GT and each model vs its own T0 model)
    cmap = matplotlib.colormaps["RdBu_r"]
    stable = reg["stable_stationary"]

    def signed(a, b):
        d = (disp(a) - disp(b)).mean(-1)
        rgb = cmap(np.clip(d / CHG_VMAX * 0.5 + 0.5, 0, 1))[..., :3]
        rgb[~stable] = 0.15
        return L.to_u8(rgb)

    ch = [L.label(up(crop(signed(g3, g0))), "GT dG")]
    for k in order[2:]:
        base = n0
        ch.append(L.label(up(crop(signed(imgs[k][0], base))), f"dN {k.split(' (')[0]}"))
    p = dest / "03_signed_change_stable_pixels.png"
    L.save_png(p, np.concatenate(ch, 1))
    record(p, [refd, hist / "frozen" / "teaset_locked" / "T0" / "8dna_upstream.exr"] + [s for k in order[2:] for s in imgs[k][1]],
           f"display-space change, RdBu_r, +-{CHG_VMAX}; dN = model at T3 - released model at T0 (each warm arm's own T0); "
           "B is a scratch model, so its dN also contains its static difference from the released model; non-stable pixels grey")

    # 04 oracle mask: triplane planes and image-space stencil coverage
    N = 64
    A = (orc["A3"] + orc["A0"]).astype(float)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8.4))
    for pl in range(3):
        sl = slice(pl * N * N, (pl + 1) * N * N)
        ax = axes[0, pl]
        ax.imshow(np.log10(1 + A[sl].reshape(N, N)), cmap="Greys", origin="lower")
        ax.contour(orc["M95"][sl].reshape(N, N), levels=[0.5], colors=[COLORS["E"]], linewidths=1.2)
        ax.contour(orc["S_spatial"][sl].reshape(N, N), levels=[0.5], colors=[COLORS["S"]], linewidths=1.0, linestyles="--")
        ax.set_title(["xy plane (col x, row y)", "yz plane (col y, row z)", "zx plane (col z, row x)"][pl], fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax = axes[1, pl]
        ax.imshow(np.log10(1 + orc["W3"][sl].reshape(N, N).astype(float)), cmap="Greys", origin="lower")
        ax.contour(orc["M95"][sl].reshape(N, N), levels=[0.5], colors=[COLORS["E"]], linewidths=1.2)
        ax.set_title("T3 support weight (log) with M95", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_ylabel("affected weight A (log)")
    from matplotlib.lines import Line2D
    fig.legend([Line2D([0], [0], color=COLORS["E"]), Line2D([0], [0], color=COLORS["S"], ls="--")],
               ["M95 oracle mask (E)", "equal-size spatial mask (S)"], loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Oracle affected weight per triplane cell (CRN T0/T3 path samples)", fontsize=11)
    p = dest / "04_oracle_mask_triplane.png"
    fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
    record(p, [root / "oracle" / "oracle_cells.npz"])

    # 04b image-space: does each pixel's 12-cell stencil (25 sub-pixel hits) touch M95?
    from sro_metrics import pixel_stencil_cells
    pc = pixel_stencil_cells(reg["x1_T3_sub"], N)
    ok = pc >= 0
    inm = np.where(ok, orc["M95"][np.clip(pc, 0, None)], False)
    frac = (inm.sum(1) / np.maximum(ok.sum(1), 1)).reshape(512, 512)
    vis = matplotlib.colormaps["viridis"](frac)[..., :3]
    vis[reg["background"]] = 0.1
    over = L.to_u8(0.5 * disp(g3) + 0.5 * vis)
    edge = np.zeros((512, 512), bool)
    for k, col in (("R_aff_stat", (255, 60, 60)), ("R_mover", (255, 200, 0))):
        m = reg[k]
        e = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
        over[e] = col
    p = dest / "05_stencil_share_in_M95_image.png"
    L.save_png(p, L.label(over, "share of pixel stencil in M95 (viridis 0..1); outlines: R_aff_stat red, R_mover yellow"))
    record(p, [root / "regions" / "regions.npz", root / "oracle" / "oracle_cells.npz"])

    # 06 quality vs adaptation latency
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for j, key in enumerate(("R_aff", "R_unaff")):
        ax = axes[j]
        for arm in ("C", "D", "E", "S", "B"):
            for seed, ls in ((0, "-"), (1, ":")):
                a = met["arms"].get(f"{arm}_s{seed}")
                if not a:
                    continue
                tr = [r for r in a["trace"] if r.get("adaptation_latency_s") is not None and r["step"] > 0]
                if not tr:
                    continue
                ax.plot([r["adaptation_latency_s"] for r in tr], [r[key] for r in tr], ls, color=COLORS[arm], lw=2,
                        marker="o", ms=4, label=LABELS[arm] + ("" if seed == 0 else " (seed 1)"))
        hb = met["B_historical"]["trace"]
        ax.plot([r["adaptation_latency_s"] for r in hb], [r[key] for r in hb], "-", color=COLORS["B_hist"], lw=2, marker="s", ms=3,
                label=LABELS["B_hist"])
        ax.axhline(met["A_frozen_attached"][key], color="#52514e", ls="--", lw=1.2, label="A frozen (attached)")
        ax.axhline(met["A_step0_upstream"][key], color="#8a8985", ls=":", lw=1.2, label="step 0 (released, T3 frame)")
        ax.set_xscale("log")
        ax.set_xlabel("adaptation latency [s] (excl. validation, oracle)")
        ax.set_ylabel(f"display MAE vs GT T3, {key}")
        ax.grid(alpha=0.25)
        ax.set_title({"R_aff": "affected region (changed stationary + mover)", "R_unaff": "unaffected stationary region"}[key], fontsize=10)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.12))
    p = dest / "06_quality_vs_latency.png"
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    record(p, [root / "metrics.json"])

    # 07 seam / collateral: |E_final - step0| on stable pixels, crop of the unaffected region bordering the footprint
    if "E" in finals:
        a_up = L.load_exr(hist / "frozen" / "teaset_locked" / "T3" / "8dna_upstream.exr")
        d = np.abs(disp(finals["E"]) - disp(a_up)).mean(-1)
        vis = L.error_map(d, 0.1)
        vis[reg["background"]] = 0
        vis2 = L.error_map(np.abs(disp(finals["E"]) - disp(g3)).mean(-1) - np.abs(disp(a_up) - disp(g3)).mean(-1) + 0.05, 0.1)
        un = reg["R_unaff"]
        e = un & ~(np.roll(un, 1, 0) & np.roll(un, -1, 0) & np.roll(un, 1, 1) & np.roll(un, -1, 1))
        vis[e] = (80, 255, 80)
        p = dest / "07_E_change_vs_step0_full.png"
        L.save_png(p, np.concatenate([L.label(vis, "|E final - step 0| (0..0.1), R_unaff outline green"),
                                      L.label(vis2, "err(E) - err(step 0), centred at 0.05 (0..0.1)")], 1))
        record(p, [root / "runs" / "E_s0", hist / "frozen" / "teaset_locked" / "T3" / "8dna_upstream.exr"])

    # 08 crop flicker GIF: GT T3 / A / D / E / C
    frames = [L.label(up(crop(L.to_u8(disp(imgs[k][0])))), k) for k in order if k.startswith(("GT T3", "A ", "C ", "D ", "E "))]
    p = dest / "08_crop_cycle.gif"
    L.save_gif(p, frames, 900)
    record(p, [s for k in order for s in imgs[k][1]])

    # 09 state / cost table (machine-readable copy of the key numbers)
    tab = {"oracle": json.loads((root / "oracle" / "oracle.json").read_text())["masks"],
           "screening": met["screening"], "gates": met["gates"], "static": met["static"], "noise": met["noise"]}
    p = dest / "09_key_numbers.json"
    S.write_json(p, tab)
    record(p, [root / "metrics.json", root / "oracle" / "oracle.json"])
    S.write_json(dest / "manifest.json", manifest)
    print("exported", len(manifest["files"]), "files to", dest, flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(rc)
