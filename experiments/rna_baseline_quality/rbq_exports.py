"""Worklog-33 reviewer exports (RNA venv, GPU for the forward passes).

    python rbq_exports.py --root results/rna_baseline_quality --out results/evaluation/33

Panels (display transform clamp(x^(1/2.2)) everywhere; crops chosen from reference/visibility
only, never from model error):
  01_<asset>_view<v>.png   GT | converged ref (teaset) | RNA | |RNA-GT| | |RNA-ref| | |GT-ref| | GT high-frequency mask
  02_teaset_crops.png      brightest-highlight window and largest shadowed-receiver window, C1 and C2
  03_train_val_curves.png  train vs validation loss of Rain, C1 A, C2 A, C2 P16w
  04_factorial.png         D1 cells, pooled PSNR (lit / shadowed)
  05_key_numbers.json, manifest.json
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(1, str(ROOT / "external" / "relightable-neural-assets"))
SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # reference palette slots 1-4
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"])
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": "#c3c2b7", "font.size": 9})
W = "/mnt/c/Projects/NeuralShaderDemo"
S = f"{W}/results/rna_surface_ownership"
ASSETS = {
    "rain": {"cfg": "configs/rain.yml", "weights": f"{W}/results/batch1_hq_dynamic_failure/cloth/static/training/batch1a-rain-static-hq/version_0/checkpoints/last.ckpt",
             "val": f"{W}/results/batch1_hq_dynamic_failure/cloth/static/dataset/rain_static_val.h5"},
    "C1": {"cfg": "configs/C1.yml", "weights": f"{S}/v1_858df4d/C1/A/state.pt", "val": f"{W}/results/8dna_replication/rna_teaset/datasets/teaset_T0_val.h5"},
    "C2": {"cfg": "configs/C2_valonly.yml", "weights": f"{S}/v1_858df4d_c2/C2/A/state.pt", "val": f"{S}/v1_858df4d/datasets_c2/teaset_T0_val.h5"},
}


def disp(x):
    return np.clip(np.clip(x, 0, None) ** (1 / 2.2), 0, 1)


def off(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def predict_view(asset, v, root):
    import h5py
    import torch
    from rna import config, models

    a = ASSETS[asset]
    conf = config.get_neumat_config(config_filename=str((root / a["cfg"]).resolve()))
    f = h5py.File(a["val"], "r")
    res = int(f.attrs["resolution"][0])
    aabb = (np.array(f.attrs["aabb_min"], np.float64), np.array(f.attrs["aabb_max"], np.float64))
    model = models.NeuralSurfaceTriPlane(**dict(conf.model_params)).cuda()
    sd = torch.load(a["weights"], map_location="cuda", weights_only=False)
    if "state_dict" in sd:
        sd = {k[6:]: t for k, t in sd["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(sd); model.eval(); model.set_iteration_index(10 ** 9)
    g = lambda k: np.nan_to_num(f[k][v].astype(np.float32))
    m = g("alpha")[:, 0] > 0
    pos = (g("position")[m] - aabb[0]) / (aabb[1] - aabb[0])
    T = lambda x: torch.tensor(x, dtype=torch.float32, device="cuda").unsqueeze(0)
    with torch.no_grad():
        pr = model(T(pos), T(g("camera_dir")[m]), T(g("light_dir")[m]), T(g("normal")[m])).squeeze(0)
    vis = g("diffuse_direct")[m][:, 0] > 0
    rgb = torch.where(torch.tensor(vis, device="cuda")[:, None], pr[:, :3], pr[:, 3:]).cpu().numpy()
    full = lambda x: (lambda o: (o.__setitem__(m, x), o)[1])(np.zeros((res * res, 3), np.float32)).reshape(res, res, 3)
    visf = np.zeros(res * res, bool); visf[m] = vis
    return {"gt": full(g("color")[m]), "pred": full(rgb), "mask": m.reshape(res, res), "vis": visf.reshape(res, res), "res": res}


def hf_mask(gt, mask):
    lum = disp(gt).mean(-1)
    grad = np.zeros_like(lum)
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        ok = mask & np.roll(mask, (dy, dx), (0, 1))
        grad = np.maximum(grad, np.where(ok, np.abs(lum - np.roll(lum, (dy, dx), (0, 1))), 0))
    return mask & (grad >= np.quantile(grad[mask], 0.9))


def window(score, w=96):
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
    files = []
    views = {}
    for asset in ("rain", "C1", "C2"):
        for v in (0, 20):
            d = predict_view(asset, v, root)
            refp = root / "reference" / f"{asset}_view{v}_ref.npz"
            d["ref"] = np.load(refp)["color"].reshape(d["res"], d["res"], 3) if refp.exists() else None
            views[(asset, v)] = d
            cols = [("GT", disp(d["gt"]), None)]
            if d["ref"] is not None:
                cols.append(("converged ref (8x spp)", disp(d["ref"]), None))
            cols.append(("RNA (final weights)", disp(d["pred"]), None))
            err = lambda a, b: np.where(d["mask"], np.abs(disp(a) - disp(b)).mean(-1), 0)
            cols.append(("|RNA - GT|", err(d["pred"], d["gt"]), "e"))
            if d["ref"] is not None:
                cols.append(("|RNA - ref|", err(d["pred"], d["ref"]), "e"))
                cols.append(("|GT - ref| (target noise)", err(d["gt"], d["ref"]), "e"))
            cols.append(("GT high-frequency 10%", hf_mask(d["gt"], d["mask"]).astype(float), "m"))
            fig, axs = plt.subplots(1, len(cols), figsize=(2.3 * len(cols), 2.7))
            for ax, (t, img, kind) in zip(axs, cols):
                if kind == "e":
                    im = ax.imshow(img, cmap=SEQ, vmin=0, vmax=0.3)
                elif kind == "m":
                    ax.imshow(img, cmap="Greys", vmin=0, vmax=1)
                else:
                    ax.imshow(img)
                ax.set_title(t, fontsize=8); off(ax)
            fig.colorbar(im, ax=axs, fraction=0.015, shrink=0.8).set_label("display |error| (0-0.3)", fontsize=7)
            label = {"rain": "Rain (official Cycles generator; known-good control)", "C1": "Teaset C1 (bridge, Ni alpha 0.05/0.1, 512 spp)",
                     "C2": "Teaset C2 (bridge, Ni alpha 0.3, 128 spp)"}[asset]
            fig.suptitle(f"{label} - val view {v}", fontsize=9)
            name = f"01_{asset}_view{v}.png"
            fig.savefig(out / name, dpi=140, bbox_inches="tight"); plt.close(fig); files.append(name)

    # 02 teaset crops: brightest highlight (ref luminance) and largest shadowed receiver (visibility)
    fig, axs = plt.subplots(4, 4, figsize=(9, 9.4))
    r = 0
    for asset in ("C1", "C2"):
        d = views[(asset, 0)]
        base = d["ref"] if d["ref"] is not None else d["gt"]
        for kind, score in (("brightest highlight", np.where(d["mask"], base.mean(-1), 0) ** 2),
                            ("shadowed receiver", (d["mask"] & ~d["vis"]).astype(float))):
            y, x = window(score)
            sl = (slice(y, y + 96), slice(x, x + 96))
            for c, (t, img) in enumerate((("GT", d["gt"]), ("converged ref", base), ("RNA", d["pred"]))):
                axs[r, c].imshow(disp(img)[sl], interpolation="nearest"); off(axs[r, c])
                if r == 0:
                    axs[r, c].set_title(t, fontsize=8)
            axs[r, 3].imshow(np.where(d["mask"], np.abs(disp(d["pred"]) - disp(base)).mean(-1), 0)[sl], cmap=SEQ, vmin=0, vmax=0.3,
                             interpolation="nearest"); off(axs[r, 3])
            if r == 0:
                axs[r, 3].set_title("|RNA - ref|", fontsize=8)
            axs[r, 0].set_ylabel(f"{asset} view 0\n{kind}\n@({y},{x})", fontsize=7)
            r += 1
    fig.suptitle("Teaset 96x96 crops (windows chosen from reference luminance / visibility only)", fontsize=9)
    fig.savefig(out / "02_teaset_crops.png", dpi=140, bbox_inches="tight"); plt.close(fig); files.append("02_teaset_crops.png")

    # 03 curves
    c32 = {c: json.loads((ROOT / "results/rna_surface_ownership/diag_wl32" / f"curves_{c}.json").read_text()) for c in ("C1", "C2")}
    cre = json.loads((root / "curves_rain_einar.json").read_text())
    runs = [("Rain A", cre["rain"]), ("Teaset C1 A", c32["C1"]["A"]), ("Teaset C2 A", c32["C2"]["A"]), ("Teaset C2 P16w (16x params)", c32["C2"]["P16w"])]
    fig, axs = plt.subplots(1, len(runs), figsize=(13, 3.3), sharey=True)
    for ax, (t, cv) in zip(axs, runs):
        tr = np.array(cv["loss"]); va = np.array(cv["val_loss"])
        k = 25
        sm = np.convolve(tr[:, 1], np.ones(k) / k, mode="valid")
        ax.plot(np.linspace(0, len(va), len(sm)), sm, color=SER[0], lw=1.6, label="train loss (25-log moving mean)")
        ax.plot(np.arange(len(va)), va[:, 1], color=SER[1], lw=1.6, label="validation loss")
        ax.set_yscale("log"); ax.set_title(t, fontsize=9); ax.set_xlabel("epoch")
        ax.grid(axis="y", color="#e6e5e0", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].set_ylabel("upstream visibility-weighted log-L2")
    axs[-1].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle("Train vs validation loss (same RNA config and contract); Teaset gap 7-9x, Rain 2.3x; P16w lowers only the train loss", fontsize=9)
    fig.savefig(out / "03_train_val_curves.png", dpi=140, bbox_inches="tight"); plt.close(fig); files.append("03_train_val_curves.png")

    # 04 factorial
    an = {c: json.loads((root / f"analysis_{c}.json").read_text()) for c in ("C1", "C2") if (root / f"analysis_{c}.json").exists()}
    cells = ["TS_train_cam_stored", "TP_train_cam_new_perpixel_lights", "TI_train_cam_perimage_light", "VP_val_cam_perpixel_lights", "VI_val_cam_stored"]
    short = ["TS stored train\n(seen samples)", "TP train cam\nnew px lights", "TI train cam\nper-image light", "VP val cam\npx lights", "VI val cam\nstored val"]
    fig, axs = plt.subplots(1, len(an), figsize=(6 * len(an), 3.6))
    axs = np.atleast_1d(axs)
    for ax, (c, a) in zip(axs, an.items()):
        D = a["control_D1"]
        xs = np.arange(len(cells)); wbar = 0.26
        for j, (key, lab, col) in enumerate((("pooled_psnr", "all pixels", SER[0]), ("lit_psnr", "lit", SER[2]), ("shadow_psnr", "shadowed", SER[1]))):
            vals = [D[cl][key] if cl in D else np.nan for cl in cells]
            ax.bar(xs + (j - 1) * wbar, vals, wbar - 0.03, color=col, label=lab)
        ax.set_xticks(xs); ax.set_xticklabels(short, fontsize=7)
        lo = np.nanmin([D[cl][k] for cl in D for k in ("pooled_psnr", "lit_psnr", "shadow_psnr")]) - 1
        ax.set_ylim(lo, None); ax.set_ylabel("PSNR (dB, pooled)"); ax.set_title(f"{c}: RNA A on the D1 cells", fontsize=9)
        ax.grid(axis="y", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[-1].legend(frameon=False, fontsize=8)
    fig.savefig(out / "04_factorial.png", dpi=140, bbox_inches="tight"); plt.close(fig); files.append("04_factorial.png")

    key = {c: {"control_C_pooled": a["control_C"].get("pooled"), "control_C_tails": a["control_C"].get("tails"),
               "control_D1": {k: {kk: vv for kk, vv in v.items() if kk != "per_view_psnr"} for k, v in a["control_D1"].items()}} for c, a in an.items()}
    key["control_A"] = {p.stem: {k: v for k, v in json.loads(p.read_text()).items() if k not in ("U", "O")} | {
        "U_pooled": json.loads(p.read_text())["U"]["pooled_psnr"], "U_view_mean": json.loads(p.read_text())["U"]["mean_per_view_psnr"]}
        for p in sorted((root / "A_eval_paths").glob("*.json"))}
    if (root / "batch1_arms_vs_reference.json").exists():
        key["batch1_arms_vs_converged_reference_views_0_10_20_30"] = json.loads((root / "batch1_arms_vs_reference.json").read_text())
    if (root / "d2" / "A100" / "train.json").exists():
        t = json.loads((root / "d2" / "A100" / "train.json").read_text())
        key["D2_train_100_views"] = {"best_val_psnr_lightning": t.get("best_val_psnr"), "seconds": t["seconds"]["total"],
                                     "final_eval": key["control_A"].get("C1_A100_final")}
    (out / "05_key_numbers.json").write_text(json.dumps(key, indent=1)); files.append("05_key_numbers.json")
    (out / "manifest.json").write_text(json.dumps({"worklog": "docs/worklogs/33_WORKLOG_RNA_STATIC_BASELINE_ROOT_CAUSE_ATTRIBUTION.md",
                                                   "generator": "experiments/rna_baseline_quality/rbq_exports.py", "source": "results/rna_baseline_quality",
                                                   "files": {f: hashlib.sha256((out / f).read_bytes()).hexdigest() for f in files}}, indent=1))
    print("\n".join(files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
