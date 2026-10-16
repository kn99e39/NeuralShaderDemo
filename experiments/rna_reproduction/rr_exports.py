"""Worklog-35 reviewer exports (RNA venv; CPU).

    python rr_exports.py --root results/rna_reproduction --out results/evaluation/35

  01_lego_frame<f>.png     native Cycles reference (2048 spp, OIDN) | RNA HQ (released ckpt) | RNA small | |error| maps
  02_lego_crops.png        brightest-highlight and darkest-shadowed-asset windows (chosen from the reference only)
  03_val_psnr_per_view.png per-view validation PSNR of released / reproduced checkpoints (if present)
  04_training_curves.png   reproduced training curves vs released logged best (if present)
  05_key_numbers.json, manifest.json
Display: clamp(x^(1/2.2)); error |ldr(pred) - ldr(ref)| with the code's tonemap, one 0-0.15 scale.
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

SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
SER = {"released HQ": "#2a78d6", "released small": "#eb6834", "reproduced HQ": "#1baf7a", "reproduced small": "#eda100"}
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"])
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": "#c3c2b7", "font.size": 9})


def read(p):
    import pyexr

    return pyexr.open(str(p)).get().astype(np.float32)


def disp(x):
    return np.clip(np.clip(x, 0, None) ** (1 / 2.2), 0, 1)


def ldr(x):
    x = np.clip(x, 0, None)
    return ((x / (1 + x)) ** (1 / 2.2)).mean(-1)


def off(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def window(score, w=160):
    s = np.pad(np.cumsum(np.cumsum(score, 0), 1), ((1, 0), (1, 0)))
    tot = s[w:, w:] - s[:-w, w:] - s[w:, :-w] + s[:-w, :-w]
    return np.unravel_index(np.argmax(tot), tot.shape)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    R = root / "work" / "renders" / "rna_tog"
    files = []
    frames = (13, 19, 26, 33, 68)
    data = {}
    for f in frames:
        ref = read(R / "lego" / f"ref_{f}.exr")
        data[f] = {"ref": ref[..., :3], "alpha": ref[..., 3] if ref.shape[-1] == 4 else None,
                   "HQ": read(R / "lego" / f"rna_{f}.exr")[..., :3], "small": read(R / "lego_rt" / f"rna_{f}.exr")[..., :3]}
    for f in (13, 33):
        d = data[f]; m = d["alpha"] > 0
        cols = [("Cycles reference\n(2048 spp, OIDN)", disp(d["ref"]), None), ("RNA HQ (released)", disp(d["HQ"]), None),
                ("RNA small (released)", disp(d["small"]), None),
                ("|HQ - ref|", np.where(m, np.abs(ldr(d["HQ"]) - ldr(d["ref"])), 0), "e"), ("|small - ref|", np.where(m, np.abs(ldr(d["small"]) - ldr(d["ref"])), 0), "e")]
        fig, axs = plt.subplots(1, len(cols), figsize=(3.2 * len(cols), 3.6))
        for ax, (t, img, k) in zip(axs, cols):
            if k:
                im = ax.imshow(img, cmap=SEQ, vmin=0, vmax=0.15)
            else:
                ax.imshow(img)
            ax.set_title(t, fontsize=8); off(ax)
        fig.colorbar(im, ax=axs, fraction=0.012, shrink=0.8).set_label("|ldr error| (0-0.15)", fontsize=7)
        fig.suptitle(f"Official Lego, lego_cmp frame {f} (point light from lego.json, 1024^2): upstream scripts/render.py, released checkpoints", fontsize=9)
        name = f"01_lego_frame{f}.png"; fig.savefig(out / name, dpi=130, bbox_inches="tight"); plt.close(fig); files.append(name)
    fig, axs = plt.subplots(4, 5, figsize=(14, 11.5))
    r = 0
    for f in (13, 33):
        d = data[f]; m = d["alpha"] > 0; lum = ldr(d["ref"])
        for kind, score in (("brightest highlight", np.where(m, lum, 0) ** 4), ("darkest asset region", np.where(m, 1.0 - lum, 0) * m)):
            y, x = window(score); sl = (slice(y, y + 160), slice(x, x + 160))
            for c, (t, img, k) in enumerate((("reference", disp(d["ref"])[sl], None), ("RNA HQ", disp(d["HQ"])[sl], None), ("RNA small", disp(d["small"])[sl], None),
                                             ("|HQ - ref|", np.where(m, np.abs(ldr(d["HQ"]) - lum), 0)[sl], "e"), ("|small - ref|", np.where(m, np.abs(ldr(d["small"]) - lum), 0)[sl], "e"))):
                ax = axs[r, c]
                ax.imshow(img, cmap=SEQ if k else None, vmin=0 if k else None, vmax=0.15 if k else None, interpolation="nearest"); off(ax)
                if r == 0:
                    ax.set_title(t, fontsize=8)
            axs[r, 0].set_ylabel(f"frame {f}\n{kind}\n@({y},{x})", fontsize=7)
            r += 1
    fig.suptitle("160x160 crops; windows chosen from the reference only (luminance^4 / darkness within the asset); error scale 0-0.15", fontsize=9)
    fig.savefig(out / "02_lego_crops.png", dpi=130, bbox_inches="tight"); plt.close(fig); files.append("02_lego_crops.png")
    evals = {p.stem: json.loads(p.read_text()) for p in sorted((root / "val_eval").glob("*.json"))} if (root / "val_eval").exists() else {}
    if evals:
        fig, ax = plt.subplots(figsize=(10, 3.4))
        for k, e in evals.items():
            lab = k.replace("_", " ")
            col = SER.get(lab, "#6250d6")
            ax.plot(range(len(e["per_view_psnr"])), e["per_view_psnr"], marker="o", ms=3, lw=1.4, color=col, label=f"{lab}: mean {e['view_mean_psnr']:.2f} dB")
        ax.axhline(34.09, color=SER["released HQ"], ls="--", lw=0.9); ax.axhline(28.77, color=SER["released small"], ls="--", lw=0.9)
        ax.text(39.5, 34.09, " Table 1 HQ 34.09", fontsize=7, color=INK2, va="center"); ax.text(39.5, 28.77, " Table 1 small 28.77", fontsize=7, color=INK2, va="center")
        ax.set_xlabel("validation view"); ax.set_ylabel("PSNR (dB, code tonemap, alpha>0)")
        ax.grid(axis="y", color="#e6e5e0", lw=0.6); ax.legend(frameon=False, fontsize=7, loc="lower left")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.savefig(out / "03_val_psnr_per_view.png", dpi=130, bbox_inches="tight"); plt.close(fig); files.append("03_val_psnr_per_view.png")
    cv = root / "train_curves.json"
    if cv.exists():
        c = json.loads(cv.read_text())
        fig, ax = plt.subplots(figsize=(8, 3.4))
        for k, cur in c.items():
            vp = np.array(cur["val_psnr"])
            ax.plot(vp[:, 0], vp[:, 1], lw=1.6, color=SER.get(k, "#6250d6"), label=k)
        ax.axhline(33.94, color=SER["released HQ"], ls=":", lw=0.9); ax.axhline(28.95, color=SER["released small"], ls=":", lw=0.9)
        ax.set_xlabel("training step"); ax.set_ylabel("Lightning val_psnr (dB)"); ax.legend(frameon=False, fontsize=8)
        ax.set_title("reproduced training (dotted: best val_psnr logged inside the released checkpoints)", fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.savefig(out / "04_training_curves.png", dpi=130, bbox_inches="tight"); plt.close(fig); files.append("04_training_curves.png")
    key = {"fig8_like": json.loads((root / "compare_fig8.json").read_text()) if (root / "compare_fig8.json").exists() else None,
           "val_eval": {k: {kk: vv for kk, vv in v.items() if kk != "per_view_psnr"} for k, v in evals.items()},
           "qualify": json.loads((root / "qualify.json").read_text()) if (root / "qualify.json").exists() else None,
           "n1_noise": {p.stem: {kk: vv for kk, vv in json.loads(p.read_text()).items() if kk != "note"} for p in sorted((root / "val_eval_n1").glob("*.json"))}
           if (root / "val_eval_n1").exists() else None,
           "n1_check": json.loads((root / "n1_check.json").read_text()) if (root / "n1_check.json").exists() else None,
           "train_timing": json.loads((root / "train_timing.json").read_text()) if (root / "train_timing.json").exists() else None}
    (out / "05_key_numbers.json").write_text(json.dumps(key, indent=1, default=str)); files.append("05_key_numbers.json")
    (out / "manifest.json").write_text(json.dumps({"worklog": "docs/worklogs/35_WORKLOG_RNA_REFERENCE_REPRODUCTION_LEGO.md", "generator": "experiments/rna_reproduction/rr_exports.py",
                                                   "source": "results/rna_reproduction", "files": {x: hashlib.sha256((out / x).read_bytes()).hexdigest() for x in files}}, indent=1))
    print("\n".join(files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
