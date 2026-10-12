"""Reviewer exports for worklog 31 (presentation only; computes no verdict).

    windows/run.ps1 ../surface_owned_radiance/sor_exports.py --root <batch dir> --dest results/evaluation/31
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import sor_common as S

COLORS = {"B": "#e87ba4", "C": "#2a78d6", "DG": "#eb6834", "E": "#1baf7a", "E5": "#eda100", "U": "#4a3aa7"}
LABELS = {"B": "B all-object scratch", "C": "C all-object warm (= all-local)", "DG": "DG geometry-dirty only",
          "E": "E oracle dirty objects (0.02)", "E5": "E5 oracle dirty objects (0.05, sensitivity)",
          "U": "U unit-level oracle M95 (diagnostic)"}
HIST = S.ROOT / "results" / "8dna_replication"


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
    dest = S.ROOT / args.dest
    dest.mkdir(parents=True, exist_ok=True)
    met = json.loads((root / "metrics.json").read_text())
    manifest = {"worklog": 31, "root": L.rel(root), "git": R.git_state(L.ROOT), "files": {}}

    def record(p, sources, note=""):
        manifest["files"][p.name] = {"sha256": L.sha256(p), "sources": sources, "note": note}

    disp = L.tonemap
    g = {s: 0.5 * (L.load_exr(HIST / "w21_reference_correction" / s / "gt_A.exr") + L.load_exr(HIST / "w21_reference_correction" / s / "gt_B.exr")) for s in ("T0", "T3")}
    cls = {s: np.load(HIST / "transport_decomposition" / "w21_envmap" / f"image_{s}.npz")["classes"] for s in ("T0", "T3")}
    direct = {s: cls[s][:, :, 0] + cls[s][:, :, 1] for s in cls}
    reg = np.load(S.ROOT / "results/selective_refit_oracle/v1_bc085ea/regions/regions.npz")
    arms = {}
    for name in ("C_s0", "B_s0", "DG_s0", "E_s0", "E5_s0", "U_s0"):
        d = root / "arms" / name
        if (d / "snapshots.npz").exists():
            z = np.load(d / "snapshots.npz")
            arms[name] = {"final": z[sorted(z.files)[-1]], "step0": z[sorted(z.files)[0]]}
    t0 = np.load(root / "arms" / "T0_s3" / "snapshots.npz")
    t0_ind = t0[sorted(t0.files)[-1]]
    imgs = {"GT T0": g["T0"], "GT T3": g["T3"], "T0 model at T0": direct["T0"] + t0_ind,
            "A frozen at T3": direct["T3"] + arms["C_s0"]["step0"]}
    for name, lab in (("B_s0", "B scratch"), ("C_s0", "C all warm"), ("DG_s0", "DG mover only"), ("E_s0", "E oracle objs"),
                      ("E5_s0", "E5 (0.05)"), ("U_s0", "U units M95")):
        if name in arms:
            imgs[lab] = direct["T3"] + arms[name]["final"]
    order = list(imgs)

    # 01 full frames + error maps at T3 (GT T0 / T0-model rows compare with GT T0)
    tiles, errs = [], []
    for k in order:
        ref = g["T0"] if k in ("GT T0", "T0 model at T0") else g["T3"]
        tiles.append(L.label(L.to_u8(disp(imgs[k])), k))
        errs.append(L.label(L.error_map(np.abs(disp(imgs[k]) - disp(ref)).mean(-1), 0.2), "|err| (0..0.2)"))
    half = (len(order) + 1) // 2
    pad = lambda t: t + [np.zeros_like(t[0])] * (half - len(t))
    p = dest / "01_full_frame_and_error.png"
    L.save_png(p, np.concatenate([np.concatenate(pad(tiles[:half]), 1), np.concatenate(pad(errs[:half]), 1),
                                  np.concatenate(pad(tiles[half:]), 1), np.concatenate(pad(errs[half:]), 1)], 0))
    record(p, ["references", "arms/*/snapshots.npz"], "total = WL24 physical direct + learned >= 2-vertex image")

    m = reg["R_aff_stat"] | reg["R_mover"]
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = max(ys.min() - 16, 0), min(ys.max() + 16, 511), max(xs.min() - 16, 0), min(xs.max() + 16, 511)
    crop = lambda a: np.kron(a[y0:y1, x0:x1], np.ones((2, 2, 1), a.dtype)) if a.ndim == 3 else np.kron(a[y0:y1, x0:x1], np.ones((2, 2), a.dtype))
    p = dest / "02_affected_crop.png"
    L.save_png(p, np.concatenate([np.concatenate([L.label(crop(L.to_u8(disp(imgs[k]))), k) for k in order], 1),
                                  np.concatenate([L.label(crop(L.error_map(np.abs(disp(imgs[k]) - disp(g["T0"] if k in ("GT T0", "T0 model at T0") else g["T3"])).mean(-1), 0.2)), "err") for k in order], 1)], 0))
    record(p, ["references", "arms"], "crop = bounding box of R_aff_stat and R_mover (+16 px), 2x")

    cmap = matplotlib.colormaps["RdBu_r"]
    stable = reg["stable_stationary"]

    def signed(a, b):
        dd = (disp(a) - disp(b)).mean(-1)
        rgb = cmap(np.clip(dd / 0.25 * 0.5 + 0.5, 0, 1))[..., :3]
        rgb[~stable] = 0.15
        return L.to_u8(rgb)

    n0 = imgs["T0 model at T0"]
    tiles = [L.label(crop(signed(g["T3"], g["T0"])), "GT dG")] + [L.label(crop(signed(imgs[k], n0)), f"dN {k}") for k in order[3:]]
    p = dest / "03_signed_change_stable.png"
    L.save_png(p, np.concatenate(tiles, 1))
    record(p, ["references", "arms"], "display-space T0->T3 change on stable stationary pixels, RdBu_r +-0.25; dN = arm at T3 - this representation's T0 model")

    # 04 ownership + oracle: per-pixel object id and the oracle change energy of the units each pixel reads
    orc = np.load(root / "oracle" / "oracle_units.npz")
    oj = json.loads((root / "oracle" / "oracle.json").read_text())
    pal = {"plate": (90, 140, 230), "teapot2": (240, 120, 50), "teapot3": (120, 200, 120), "teapot4": (220, 80, 160)}
    own = np.zeros((512, 512, 3), np.uint8)
    for k, c in pal.items():
        own[reg["part_T3"] == k] = c
    txt = " ".join(f"{k}:{oj['object_change_share'][k]:.3f}" for k in S.PARTS)
    p = dest / "04_object_ownership_and_change_share.png"
    L.save_png(p, L.label(own, f"owners (tray blue, milk pot orange, tin green, tea pot pink); change share {txt}; dirty@0.02={oj['dirty_objects']}"))
    record(p, [str(root / "oracle" / "oracle.json")])

    # 05 quality vs latency
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for j, key in enumerate(("R_aff", "R_aff_stat", "R_unaff")):
        ax = axes[j]
        for name, a in met["arms"].items():
            arm = a["arm"]
            tr = [r for r in a["trace"] if r["step"] > 0]
            if not tr:
                continue
            ax.plot([r["adaptation_latency_s"] for r in tr], [r[key] for r in tr], "-" if a["seed"] == 0 else ":",
                    color=COLORS.get(arm, "#52514e"), lw=2, marker="o", ms=4, label=LABELS.get(arm, arm) + ("" if a["seed"] == 0 else " (seed 1)"))
        if met.get("A_frozen"):
            ax.axhline(met["A_frozen"][key], color="#52514e", ls="--", lw=1.2, label="A frozen")
        ax.set_xscale("log"); ax.grid(alpha=0.25)
        ax.set_xlabel("adaptation latency [s] (excl. evaluation, oracle)")
        ax.set_ylabel(f"display MAE vs GT T3, {key}")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.14))
    p = dest / "05_quality_vs_latency.png"
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    record(p, ["metrics.json"])

    # 06 GIF crop cycle
    frames = [L.label(crop(L.to_u8(disp(imgs[k]))), k) for k in ("GT T3", "A frozen at T3", "C all warm", "E oracle objs", "DG mover only") if k in imgs]
    p = dest / "06_crop_cycle.gif"
    L.save_gif(p, frames, 900)
    record(p, ["arms"])

    p = dest / "07_key_numbers.json"
    S.write_json(p, {"static": met["static"], "gates": met["gates"], "screening": met["screening"], "oracle": met["oracle"]})
    record(p, ["metrics.json"])
    S.write_json(dest / "manifest.json", manifest)
    print("exported", len(manifest["files"]), flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(rc)
