"""Worklog-32 reviewer exports (RNA venv, CPU).

    python rso_exports.py --root results/rna_surface_ownership --out results/evaluation/32

Reads diag_wl32/<CFG>/{maps.npz,diag.json}, diag_wl32/curves_<CFG>.json and the predeclared
gate numbers in <run>/<CFG>/eval/eval.json. Display transform everywhere: clamp(x^(1/2.2)).
Crop windows are chosen from geometry only (most intra-part seam pixels), never from errors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

RUNS = {"C1": "v1_858df4d/C1", "C2": "v1_858df4d_c2/C2"}
ARMS = ["A", "P1", "P4", "P16", "P16w", "REINIT"]
SERIES = dict(zip(ARMS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))  # reference palette slots 1-6
PART_HUE = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]  # parts plate=tray, teapot2=milk pot, teapot3=tin, teapot4=tea pot (slots 1, 2, 3, 7)
SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"])
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": "#c3c2b7", "font.size": 9})


def disp(x):
    return np.clip(np.clip(x, 0, None) ** (1 / 2.2), 0, 1)


def off(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def panel_views(cfg, mp, d, out):
    views = (0, 20)
    cols = [("GT", None), ("A", "A"), ("P16", "P16"), ("P16w", "P16w"), ("|err| A", "A"), ("|err| P16", "P16")]
    fig, axs = plt.subplots(len(views), len(cols), figsize=(2.3 * len(cols), 2.5 * len(views)))
    for r, v in enumerate(views):
        gt = disp(mp[f"gt_{v}"]); m = mp[f"owner16_{v}"] >= 0
        for c, (title, arm) in enumerate(cols):
            ax = axs[r, c]
            if title.startswith("|err|"):
                e = np.abs(disp(mp[f"pred_{arm}_{v}"]) - gt).mean(-1); e[~m] = 0
                im = ax.imshow(e, cmap=SEQ, vmin=0, vmax=0.3)
            else:
                ax.imshow(gt if arm is None else disp(mp[f"pred_{arm}_{v}"]))
            off(ax)
            if r == 0:
                ax.set_title(title, fontsize=9)
        axs[r, 0].set_ylabel(f"view {v}")
    cb = fig.colorbar(im, ax=axs[:, 4:], fraction=0.04, shrink=0.8); cb.set_label("display |error| (0-0.3)")
    fig.suptitle(f"{cfg}: val views, display transform clamp(x^(1/2.2))", fontsize=10)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


def panel_ownership(mp, d, out):
    unit_part = np.array(d["partition_unit_part"])
    views = (0, 10, 20, 30)
    fig, axs = plt.subplots(1, len(views), figsize=(3.2 * len(views), 3.4))
    for ax, v in zip(axs, views):
        o = mp[f"owner16_{v}"]; gt = disp(mp[f"gt_{v}"]).mean(-1)
        rgb = np.ones(o.shape + (3,)) * np.array(matplotlib.colors.to_rgb(SURF))
        for u in range(16):
            sel = o == u
            if sel.any():
                base = np.array(matplotlib.colors.to_rgb(PART_HUE[unit_part[u]]))
                rgb[sel] = 0.45 * base + 0.55 * gt[sel, None]
        rgb[mp[f"cross_{v}"]] = 1.0
        rgb[mp[f"intra_{v}"]] = matplotlib.colors.to_rgb(INK)
        ax.imshow(rgb); off(ax); ax.set_title(f"view {v}", fontsize=9)
        for u in range(16):
            ys, xs = np.nonzero(o == u)
            if len(ys) > 400:
                ax.text(np.median(xs), np.median(ys), str(u), fontsize=7, ha="center", va="center", color=INK,
                        bbox=dict(boxstyle="round,pad=0.1", fc=SURF, ec="none", alpha=0.8))
    handles = [matplotlib.patches.Patch(color=PART_HUE[i], label=n) for i, n in enumerate(["tray (9 units)", "milk pot (2)", "tin (1)", "tea pot (4)"])]
    handles += [matplotlib.lines.Line2D([], [], color=INK, lw=2, label="intra-part seam (ownership cut)"),
                matplotlib.lines.Line2D([], [], color="#bbbbbb", lw=2, label="cross-part boundary (white)")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=8)
    fig.suptitle("P16 surface ownership on the validation views (unit id at each unit's median pixel); identical partition in C1 and C2", fontsize=10)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


def seam_window(intra, w=96):
    s = np.cumsum(np.cumsum(intra.astype(np.int64), 0), 1)
    s = np.pad(s, ((1, 0), (1, 0)))
    tot = s[w:, w:] - s[:-w, w:] - s[w:, :-w] + s[:-w, :-w]
    y, x = np.unravel_index(np.argmax(tot), tot.shape)
    return y, x


def panel_seams(cfg, mp, out, w=96):
    views = (0, 20)
    arms = ["A", "P4", "P16", "P16w"]
    fig, axs = plt.subplots(len(views), 2 + len(arms), figsize=(2.2 * (2 + len(arms)), 2.4 * len(views)))
    for r, v in enumerate(views):
        y, x = seam_window(mp[f"intra_{v}"], w)
        sl = (slice(y, y + w), slice(x, x + w))
        gt = disp(mp[f"gt_{v}"])
        ov = gt[sl].copy(); ov[mp[f"intra_{v}"][sl]] = matplotlib.colors.to_rgb("#e34948")
        axs[r, 0].imshow(ov, interpolation="nearest"); axs[r, 0].set_title("GT + seam (red)" if r == 0 else "", fontsize=8)
        axs[r, 1].imshow(gt[sl], interpolation="nearest"); axs[r, 1].set_title("GT" if r == 0 else "", fontsize=8)
        for c, a in enumerate(arms):
            axs[r, 2 + c].imshow(disp(mp[f"pred_{a}_{v}"])[sl], interpolation="nearest")
            if r == 0:
                axs[r, 2 + c].set_title(a, fontsize=8)
        for ax in axs[r]:
            off(ax)
        axs[r, 0].set_ylabel(f"view {v} @({y},{x})", fontsize=8)
    fig.suptitle(f"{cfg}: {w}x{w} crops at the window with most intra-part seam pixels (chosen from geometry only)", fontsize=9)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


def panel_reinit(cfg, mp, d, out):
    u = d["output_preservation"]["reinitialised_units"]
    views = (0, 20)
    fig, axs = plt.subplots(len(views), 3, figsize=(7.5, 2.6 * len(views)))
    for r, v in enumerate(views):
        o = mp[f"owner16_{v}"]
        diff = np.abs(disp(mp[f"pred_REINIT_{v}"]) - disp(mp[f"pred_P16_{v}"])).mean(-1)
        changed = np.any(mp[f"pred_REINIT_{v}"] != mp[f"pred_P16_{v}"], -1)
        um = np.isin(o, u)
        axs[r, 0].imshow(np.where(um, 1.0, np.where(o >= 0, 0.35, 0.0)), cmap="Greys", vmin=0, vmax=1)
        axs[r, 0].set_title("u* (black) / frozen units (grey)" if r == 0 else "", fontsize=8)
        im = axs[r, 1].imshow(diff, cmap=SEQ, vmin=0, vmax=0.1)
        axs[r, 1].set_title("|REINIT - P16| display" if r == 0 else "", fontsize=8)
        axs[r, 2].imshow(np.where(changed, 1.0, 0.0), cmap="Greys", vmin=0, vmax=1)
        axs[r, 2].set_title("any bit changed (black)" if r == 0 else "", fontsize=8)
        for ax in axs[r]:
            off(ax)
        axs[r, 0].set_ylabel(f"view {v}")
    fig.colorbar(im, ax=axs[:, 1], fraction=0.05, shrink=0.8)
    op = d["output_preservation"]
    fig.suptitle(f"{cfg}: REINIT (unit {u} re-initialised and retrained, 15 units frozen) vs P16\n"
                 f"frozen-unit pixels bit-identical: {op['frozen_pixels_bit_identical']}/{op['frozen_pixels']} (all val views)", fontsize=9)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


def panel_curves(curves, evals, out):
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)
    for ax, cfg in zip(axs, ("C1", "C2")):
        for a in ARMS:
            vp = np.array(curves[cfg][a]["val_psnr"])
            ax.plot(np.arange(len(vp)), vp[:, 1], color=SERIES[a], lw=1.6, label=a)
            ax.plot([len(vp) - 1], [evals[cfg][a]], marker="o", ms=6, color=SERIES[a], mec=SURF, mew=1.5)
        ax.axhline(25, color=INK2, lw=1, ls="--")
        ax.set_title(f"{cfg}: Lightning val PSNR per epoch (lines); rso_eval final-weight PSNR (dots)", fontsize=8.5)
        ax.set_xlabel("epoch"); ax.set_ylabel("val PSNR (dB)")
        ax.grid(axis="y", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
        lo = min(min(np.array(curves[cfg][a]["val_psnr"])[20:, 1].min() for a in ARMS), 25) - 0.3
        hi = max(max(np.array(curves[cfg][a]["val_psnr"])[:, 1].max() for a in ARMS), 25) + 0.3
        ax.set_ylim(lo, hi)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[1].legend(loc="lower right", frameon=False, fontsize=8, ncol=2)
    fig.suptitle("dashed: predeclared static-credibility gate 25 dB, which applies to the dots (rso_eval, final weights), not to the Lightning curves", fontsize=9, color=INK2)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    curves, evals, files, key = {}, {}, [], {}
    for cfg, run in RUNS.items():
        mp = np.load(root / "diag_wl32" / cfg / "maps.npz")
        d = json.loads((root / "diag_wl32" / cfg / "diag.json").read_text())
        e = json.loads((root / run / "eval" / "eval.json").read_text())["arms"]
        curves[cfg] = json.loads((root / "diag_wl32" / f"curves_{cfg}.json").read_text())
        evals[cfg] = {a: e[a]["val_psnr_rna"] for a in ARMS}
        n = {"C1": 1, "C2": 2}[cfg]
        for name, fn in ((f"01_{cfg}_views.png", lambda p: panel_views(cfg, mp, d, p)),
                         (f"03_{cfg}_seam_crops.png", lambda p: panel_seams(cfg, mp, p)),
                         (f"04_{cfg}_reinit_output_preservation.png", lambda p: panel_reinit(cfg, mp, d, p))):
            fn(out / name); files.append(name)
        if n == 2:
            panel_ownership(mp, d, out / "02_ownership_map_P16.png"); files.append("02_ownership_map_P16.png")
        key[cfg] = {"gate_eval": {a: {k: e[a][k] for k in ("val_psnr_rna", "display_mae", "boundary_display_mae", "interior_display_mae", "parameters")} for a in ARMS},
                    "diag": {a: {k: d["arms"][a][k] for k in ("relmse", "logl1", "dmae", "hf_dmae", "hf_logl1", "intra_dmae", "cross_dmae", "interior_dmae", "seam_disc", "load_s", "inference_s_per_view")} for a in ARMS},
                    "output_preservation": d["output_preservation"]}
    panel_curves(curves, evals, out / "05_val_psnr_curves.png"); files.append("05_val_psnr_curves.png")
    (out / "06_key_numbers.json").write_text(json.dumps(key, indent=1)); files.append("06_key_numbers.json")
    manifest = {"worklog": "docs/worklogs/32_WORKLOG_RNA_SURFACE_OWNED_TRANSPORT_UNITS_BATCH1.md",
                "generator": "experiments/rna_surface_ownership/rso_exports.py",
                "sources": {cfg: {"run": f"results/rna_surface_ownership/{run}", "diag": f"results/rna_surface_ownership/diag_wl32/{cfg}"} for cfg, run in RUNS.items()},
                "files": {f: hashlib.sha256((out / f).read_bytes()).hexdigest() for f in files}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("\n".join(files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
