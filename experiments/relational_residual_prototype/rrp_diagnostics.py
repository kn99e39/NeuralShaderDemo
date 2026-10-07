"""Post-hoc failure attribution for the relational-residual prototype (analysis of existing outputs only).

    windows/run.ps1 ../relational_residual_prototype/rrp_diagnostics.py --run <run dir>

Declared after the evidence run showed that no branch satisfies the recovery rule
(worklog 28, batch section 15).  No model is trained or changed here.  Questions:

  A. static vs transport-delta error.  On stable T3 pixels (identical queries to T0),
     e(T3) = N(T3) - G(T3) = [N(T0) - G(T0)] + [dN - dG],  dN = N(T3) - N(T0), dG = G(T3) - G(T0).
     The first term is the branch's static (T0) error, the second its error in the
     configuration change.  The historical ratio mixes them; this separates them.
  B. in-sample vs held-out pixels: the same split at T0 (training) and T3 (held out)
     on the spatial validation blocks.
  C. distribution shift: per query, probes hitting the moved part at T3 vs the
     maximum over the training states for the same query.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "8dna_deformation_replication"))
sys.path.insert(0, str(HERE.parents[0] / "neural_recompute_cost"))

import numpy as np  # noqa: E402

import rrp_common as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    run = Path(args.run).resolve()
    proto = C.protocol()
    import ednalib as L
    import nrc_records as R

    git = R.git_state(L.ROOT)
    R.require_clean(git, args.allow_dirty)
    tm = L.tonemap
    data = {s: dict(np.load(run / "dataset" / f"{s}.npz")) for s in ("T0", "T1b", "T2", "T3")}
    training = json.loads((run / "models" / "training.json").read_text())
    names = proto["parts_order"]
    rec = {"git": git, "declared": "post hoc, after the evidence run; analysis of existing outputs only", "branches": {}}

    def paired(s):
        """indices of the pixels stable in both T0 and s (same ROI pixels), in each state's stable order"""
        p0 = data["T0"]["roi_pixels"][data["T0"]["stable_pixels"]]
        ps = data[s]["roi_pixels"][data[s]["stable_pixels"]]
        common, i0, i1 = np.intersect1d(p0, ps, return_indices=True)
        return common, i0, i1

    for tag in training["runs"]:
        pred = dict(np.load(run / "models" / f"pred_{tag}.npz"))
        out = {}
        for s in ("T1b", "T2", "T3"):
            pix, i0, i1 = paired(s)
            g0, gs = data["T0"]["gt_pixels"][i0], data[s]["gt_pixels"][i1]
            f0, fs = data["T0"]["frozen_pixels"][i0], data[s]["frozen_pixels"][i1]
            n0, ns = f0 + pred["T0"][i0], fs + pred[s][i1]
            dG, dN, dF = gs - g0, ns - n0, fs - f0
            val = C.validation_pixels(pix, 512, proto["split"]["val_block"], proto["split"]["val_modulus"])

            def stats(sel):
                e_static = np.abs(tm(n0[sel]) - tm(g0[sel])).mean()
                e_state = np.abs(tm(ns[sel]) - tm(gs[sel])).mean()
                # display-space change error vs the reference change, and the frozen model's
                d_err = np.abs((tm(ns[sel]) - tm(n0[sel])) - (tm(gs[sel]) - tm(g0[sel]))).mean()
                d_frozen = np.abs((tm(fs[sel]) - tm(f0[sel])) - (tm(gs[sel]) - tm(g0[sel]))).mean()
                d_ref = np.abs(tm(gs[sel]) - tm(g0[sel])).mean()
                lin = dict(dN=dN[sel].reshape(-1), dG=dG[sel].reshape(-1))
                gain = float(lin["dN"] @ lin["dG"] / (lin["dG"] @ lin["dG"]))
                return {"pixels": int(sel.sum()), "error_T0_display": float(e_static), f"error_{s}_display": float(e_state),
                        "change_error_display": float(d_err), "frozen_change_error_display": float(d_frozen),
                        "reference_change_display": float(d_ref), "change_error_over_reference_change": float(d_err / d_ref),
                        "frozen_change_error_over_reference_change": float(d_frozen / d_ref), "gain_linear": gain}

            out[s] = {"all": stats(np.ones(len(pix), bool)), "train_pixels": stats(~val), "val_pixels": stats(val)}
        rec["branches"][tag] = out
        a = out["T3"]["all"]
        print(tag, f"T3: static(T0) err {a['error_T0_display']:.4f}  T3 err {a['error_T3_display']:.4f}  change err "
                   f"{a['change_error_display']:.4f} (frozen {a['frozen_change_error_display']:.4f}, |dG| {a['reference_change_display']:.4f})  "
                   f"gain {a['gain_linear']:.3f}", flush=True)

    # C. distribution shift of the probe state at T3 relative to the training states (same queries)
    mover = names.index(proto["moved_part_for_reporting_only"])
    counts = {}
    for s in ("T0", "T1b", "T2", "T3"):
        z = np.load(run / "probes" / f"{s}.npz")
        k = int(z["samples_per_pixel"])
        pix = z["pixels"][np.flatnonzero(z["stable"])]
        c = (z["remote_part"] == mover).sum(1).reshape(-1, k)  # per query sample
        counts[s] = (pix, c)
    p3, c3 = counts["T3"]
    common = p3
    mx = np.full(c3.shape, -1)
    for s in ("T0", "T1b", "T2"):
        ps, cs = counts[s]
        cm, i3, js = np.intersect1d(p3, ps, return_indices=True)
        tmp = np.full(c3.shape, -1)
        tmp[i3] = cs[js]
        mx = np.maximum(mx, tmp)
    have = mx >= 0
    rec["distribution_shift_T3"] = {
        "moved_part_hits_per_query_mean": {s: float(counts[s][1].mean()) for s in counts},
        "fraction_of_T3_queries_above_training_max": float(np.mean(c3[have] > mx[have])),
        "mean_excess_over_training_max_when_above": float(np.mean((c3 - mx)[have & (c3 > mx)])) if np.any(have & (c3 > mx)) else 0.0,
        "note": "counts of probes (out of K) hitting the moved part, per query sample; the moved part's name is used for reporting only"}
    print(json.dumps(rec["distribution_shift_T3"]), flush=True)
    R.write_json(run / "eval" / "diagnostics.json", rec)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
