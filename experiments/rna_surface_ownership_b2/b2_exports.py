"""Worklog-34 reviewer exports (RNA venv, GPU).

    python b2_exports.py --root results/rna_surface_ownership_b2 --out results/evaluation/34

Display transform clamp(x^(1/2.2)) everywhere; error maps |ldr(pred) - ldr(ref)| on one 0-0.15 scale;
crop windows are chosen from geometry only (most clean seam pixels), never from errors.
  01_C2_view<v>.png        8x reference | A | P16 | P16-PU | P16w and their error maps (dev views 0, 20)
  02_C2_seam_crops.png     seam windows: reference, A, P16, P16-PU, P16w, P16w-PU + P16 / P16-PU error
  03_C2_ownership_band.png P16 owners (part hue), seams and the PU band (w > 0)
  04_distance_profile.png  error ratio vs distance to the nearest same-part seam (C2, C1)
  05_C2_output_influence.png REINIT vs P16 under P16-PU: changed pixels vs predicted S_12 U B_12
  06_key_numbers.json, manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "rna_surface_ownership"))
import b2_compose as C  # noqa: E402
import rso_model as M  # noqa: E402

SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
SER = {"A": "#2a78d6", "P4": "#1baf7a", "P16": "#eda100", "P16w": "#e87ba4"}  # reference palette slots 1, 3, 4, 5 (worklog-32 arm colours)
PART_HUE = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"])
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": "#c3c2b7", "font.size": 9})
W = "/mnt/c/Projects/NeuralShaderDemo"
S = f"{W}/results/rna_surface_ownership"


def disp(x):
    return np.clip(np.clip(x, 0, None) ** (1 / 2.2), 0, 1)


def ldr(x):
    x = np.clip(x, 0, None)
    return ((x / (1 + x)) ** (1 / 2.2)).mean(-1)


def off(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def window(score, w=96):
    s = np.pad(np.cumsum(np.cumsum(score, 0), 1), ((1, 0), (1, 0)))
    tot = s[w:, w:] - s[:-w, w:] - s[w:, :-w] + s[:-w, :-w]
    return np.unravel_index(np.argmax(tot), tot.shape)


def seam_mask(o16, part_of):
    sm = np.zeros_like(o16, bool)
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        sh = np.roll(o16, (dy, dx), (0, 1))
        sm |= (sh != o16) & (sh >= 0) & (o16 >= 0) & (part_of(sh) == part_of(o16))
    return sm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    mp = np.load(root / "diag" / "C2" / "maps.npz")
    up = np.load(f"{S}/v1_858df4d_c2/partitions/partition_P16.npz")["unit_part"]
    part_of = lambda o: np.where(o >= 0, up[np.maximum(o, 0)], -1)

    # 01 panels
    for v in (0, 20):
        ref = mp[f"ref_{v}"]; mask = mp[f"owner16_{v}"] >= 0
        cols = [("8x reference", ref), ("A", mp[f"pred_A_{v}"]), ("P16 (hard)", mp[f"pred_P16_{v}"]), ("P16-PU", mp[f"blend_P16_{v}"]), ("P16w", mp[f"pred_P16w_{v}"])]
        fig, axs = plt.subplots(2, len(cols), figsize=(2.4 * len(cols), 5.0))
        for c, (t, img) in enumerate(cols):
            axs[0, c].imshow(disp(img)); axs[0, c].set_title(t, fontsize=9); off(axs[0, c])
            if c:
                im = axs[1, c].imshow(np.where(mask, np.abs(ldr(img) - ldr(ref)), 0), cmap=SEQ, vmin=0, vmax=0.15)
            else:
                axs[1, c].imshow(np.where(mask, ldr(ref), 0), cmap="Greys_r")
            off(axs[1, c])
        axs[1, 0].set_title("reference (RNA tonemap)", fontsize=8)
        fig.colorbar(im, ax=axs[1, :], fraction=0.02, shrink=0.8).set_label("|ldr(pred) - ldr(ref)| (0-0.15)", fontsize=7)
        fig.suptitle(f"C2 dev view {v}: P16-PU = unchanged P16 parameters, partition-of-unity composition within 7.2 mm of same-part seams", fontsize=9)
        name = f"01_C2_view{v}.png"; fig.savefig(out / name, dpi=140, bbox_inches="tight"); plt.close(fig); files.append(name)

    # 02 seam crops (geometry-only window: most seam pixels)
    fig, axs = plt.subplots(2, 8, figsize=(17, 4.6))
    for r, v in enumerate((0, 20)):
        o = mp[f"owner16_{v}"]; sm = seam_mask(o, part_of)
        y, x = window(sm.astype(float)); sl = (slice(y, y + 96), slice(x, x + 96))
        ref = mp[f"ref_{v}"]
        ov = disp(ref)[sl].copy(); ov[sm[sl]] = matplotlib.colors.to_rgb("#e34948")
        cols = [("ref + seam (red)", ov, None), ("8x reference", disp(ref)[sl], None), ("A", disp(mp[f"pred_A_{v}"])[sl], None),
                ("P16 (hard)", disp(mp[f"pred_P16_{v}"])[sl], None), ("P16-PU", disp(mp[f"blend_P16_{v}"])[sl], None),
                ("P16w (hard)", disp(mp[f"pred_P16w_{v}"])[sl], None),
                ("|P16 - ref|", np.abs(ldr(mp[f"pred_P16_{v}"]) - ldr(ref))[sl], "e"), ("|P16-PU - ref|", np.abs(ldr(mp[f"blend_P16_{v}"]) - ldr(ref))[sl], "e")]
        for c, (t, img, k) in enumerate(cols):
            ax = axs[r, c]
            if k == "e":
                ax.imshow(img, cmap=SEQ, vmin=0, vmax=0.15, interpolation="nearest")
            else:
                ax.imshow(img, interpolation="nearest")
            off(ax)
            if r == 0:
                ax.set_title(t, fontsize=8)
        axs[r, 0].set_ylabel(f"view {v} @({y},{x})", fontsize=8)
    fig.suptitle("C2 seam crops, 96x96 at the window with most same-part seam pixels (chosen from geometry only); error scale 0-0.15", fontsize=9)
    fig.savefig(out / "02_C2_seam_crops.png", dpi=140, bbox_inches="tight"); plt.close(fig); files.append("02_C2_seam_crops.png")

    # 03 ownership + band
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    for ax, v in zip(axs, (0, 20)):
        o = mp[f"owner16_{v}"]; base = ldr(mp[f"ref_{v}"])
        rgb = np.ones(o.shape + (3,)) * np.array(matplotlib.colors.to_rgb(SURF))
        for u in range(16):
            sel = o == u
            rgb[sel] = 0.5 * np.array(matplotlib.colors.to_rgb(PART_HUE[up[u]])) + 0.5 * base[sel, None]
        d = mp[f"dseam_{v}"]
        band = (d >= 0) & (d < 2.0)
        rgb[band] = 0.5 * rgb[band] + 0.5 * np.array(matplotlib.colors.to_rgb("#eda100"))
        rgb[seam_mask(o, part_of)] = matplotlib.colors.to_rgb(INK)
        ax.imshow(rgb); off(ax); ax.set_title(f"view {v}", fontsize=9)
    handles = [matplotlib.patches.Patch(color=PART_HUE[i], label=n) for i, n in enumerate(["tray (9)", "milk pot (2)", "tin (1)", "tea pot (4)"])]
    handles += [matplotlib.patches.Patch(color="#eda100", label="pixels within 2 footprints of a seam (~PU band)"),
                matplotlib.lines.Line2D([], [], color=INK, lw=2, label="same-part seam")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7)
    fig.suptitle("P16 ownership, seams and the partition-of-unity band", fontsize=9)
    fig.savefig(out / "03_C2_ownership_band.png", dpi=140, bbox_inches="tight"); plt.close(fig); files.append("03_C2_ownership_band.png")

    # 04 distance profile
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    labels = ["0-1", "1-2", "2-4", "4-8", "8-16", "16-32", "32+"]
    for ax, c in zip(axs, ("C2", "C1")):
        d = json.loads((root / "diag" / c / "diag.json").read_text())
        A = [p["abs_err"] for p in d["profile"]["A"]]
        for a in ("P4", "P16", "P16w"):
            r = [p["abs_err"] / q for p, q in zip(d["profile"][a], A)]
            ax.plot(range(len(r)), r, color=SER[a], lw=1.8, marker="o", ms=4, label=f"{a} / A")
        ax.axhline(1, color=INK2, lw=0.8, ls="--")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("distance to nearest same-part seam (pixel footprints, ~3.6 mm)"); ax.set_title(f"{c} (dev views vs 8x reference)", fontsize=9)
        ax.grid(axis="y", color="#e6e5e0", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].set_ylabel("mean |error| ratio to A"); axs[1].legend(frameon=False, fontsize=8)
    fig.savefig(out / "04_distance_profile.png", dpi=140, bbox_inches="tight"); plt.close(fig); files.append("04_distance_profile.png")

    # 05 output influence (fresh forward on dev view 0)
    import h5py
    import torch

    f = h5py.File(f"{S}/v1_858df4d/datasets_c2/teaset_T0_val.h5", "r")
    res = int(f.attrs["resolution"][0]); aabb = (np.array(f.attrs["aabb_min"], np.float64), np.array(f.attrs["aabb_max"], np.float64))
    comps = {}
    for a in ("P16", "REINIT"):
        dd = Path(f"{S}/v1_858df4d_c2/C2/{a}")
        specs = [{"params": s["params"], "lo": np.array(s["lo"]), "hi": np.array(s["hi"])} for s in json.loads((dd / "specs.json").read_text())]
        m = M.OwnedTriplaneSet(specs, aabb).cuda(); m.load_state_dict(torch.load(dd / "state.pt", map_location="cuda")); m.eval(); m.set_iteration_index(10 ** 9)
        comps[a] = C.PUComposer.from_partition(m, f"{S}/v1_858df4d_c2/partitions/partition_P16.npz", 0.0072)
    v = 0
    g = lambda k: np.nan_to_num(f[k][v].astype(np.float32))
    msk = g("alpha")[:, 0] > 0
    o16 = np.full(res * res, -1); o16[msk] = mp[f"owner16_{v}"].reshape(-1)[msk]
    args_ = (g("position")[msk], g("camera_dir")[msk], g("light_dir")[msk], g("normal")[msk], g("diffuse_direct")[msk][:, 0] > 0, o16[msk])
    pu = {a: comps[a](*args_) for a in comps}
    changed = np.zeros(res * res, bool); changed[msk] = (pu["REINIT"] != pu["P16"]).any(1)
    infl = np.zeros(res * res, bool); infl[msk] = comps["P16"].influenced_by(12, args_[0], args_[5])
    own12 = o16 == 12
    rgb = np.ones((res * res, 3)) * np.array(matplotlib.colors.to_rgb(SURF)); rgb[msk] = 0.85
    rgb[own12] = matplotlib.colors.to_rgb("#2a78d6"); rgb[infl & ~own12] = matplotlib.colors.to_rgb("#eda100")
    rgb[changed & ~infl] = matplotlib.colors.to_rgb("#e34948")
    fig, axs = plt.subplots(1, 2, figsize=(8, 4.2))
    axs[0].imshow(rgb.reshape(res, res, 3)); off(axs[0]); axs[0].set_title("predicted influence of unit 12:\nowned S_12 (blue), neighbour band B_12 (amber)", fontsize=8)
    axs[1].imshow(changed.reshape(res, res), cmap="Greys", vmin=0, vmax=1); off(axs[1])
    axs[1].set_title(f"pixels changed REINIT-PU vs P16-PU (black)\noutside predicted set: {int((changed & ~infl).sum())}", fontsize=8)
    fig.suptitle(f"C2 view {v}: unit 12 re-initialised and retrained (worklog-32 REINIT), composed with P16-PU; {int(infl.sum())} influenced px, {int((infl & ~own12).sum())} of them in neighbours", fontsize=8)
    fig.savefig(out / "05_C2_output_influence.png", dpi=140, bbox_inches="tight"); plt.close(fig); files.append("05_C2_output_influence.png")

    key = {"diag": {c: json.loads((root / "diag" / c / "diag.json").read_text()) for c in ("C2", "C1")},
           "support": json.loads((root / "support.json").read_text()) if (root / "support.json").exists() else None,
           "eval": {c: json.loads((root / "eval" / c / "eval.json").read_text()) for c in ("C2", "C1") if (root / "eval" / c / "eval.json").exists()},
           "tests": json.loads((root / "tests" / "tests.json").read_text())}
    for c in key["diag"].values():
        c.pop("footprints", None)
    (out / "06_key_numbers.json").write_text(json.dumps(key, indent=1)); files.append("06_key_numbers.json")
    (out / "manifest.json").write_text(json.dumps({"worklog": "docs/worklogs/34_WORKLOG_SURFACE_OWNED_BOUNDARY_ATTRIBUTION_AND_PU_COMPOSITION.md",
                                                   "generator": "experiments/rna_surface_ownership_b2/b2_exports.py", "source": "results/rna_surface_ownership_b2",
                                                   "files": {x: hashlib.sha256((out / x).read_bytes()).hexdigest() for x in files}}, indent=1))
    print("\n".join(files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
