"""The batch's recovery evaluation reproduces worklog 22's refit verdicts exactly.

    windows/run.ps1 ../neural_recompute_cost/tests/test_historical_rule.py

Recomputes the three worklog-22 refit blocks from the historical renders with
the same function the evaluation scripts import (cross_backbone_eval.refit_block)
and the same inputs the batch protocol names, and requires equality with
results/8dna_replication/cross_backbone/cross_backbone.json.  It also checks
that the reference/ROI/T0-model files the protocol names exist and that the
historical artifacts this batch reads are unchanged (sha256 of the T3 refit
renders and checkpoints recorded by render_8dna_refits.py).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "8dna_deformation_replication"))


def main() -> int:
    import numpy as np

    import ednalib as L

    L.init_upstream()
    from cross_backbone_eval import refit_block
    from teaset_frozen_eval import load_rois

    import nrc_eval_8dna as E8

    hist = json.loads((L.RESULTS / "cross_backbone" / "cross_backbone.json").read_text(encoding="utf-8"))
    nrc = json.loads((HERE / "protocol" / "nrc_t3_recompute_v1.json").read_text(encoding="utf-8"))
    rr = L.RESULTS / "refit" / "renders"
    failures = []

    def check(name, got, want, keys=("refit_T3_interaction_error", "T0_model_error_same_surface", "error_ratio", "gain", "recovers")):
        for k in keys:
            if got[k] != want[k]:
                failures.append(f"{name}.{k}: {got[k]!r} != {want[k]!r}")
        print(name, {k: got[k] for k in keys}, flush=True)

    for regime, hkey, tkey in (("w21_envmap", "refit_8dna_w21_envmap_corrected", "8dna_envmap_primary"),
                               ("common_light", "refit_8dna_common_light", "8dna_common_light_secondary")):
        spec = E8.REGIMES[regime]
        gt = {s: (L.load_exr(L.RESULTS / spec["gt"] / s / "gt_A.exr"), L.load_exr(L.RESULTS / spec["gt"] / s / "gt_B.exr"))
              for s in ("T0", "T3")}
        rois = load_rois(L.RESULTS / spec["rois"], ("T0", "T3"))
        blk = refit_block(gt, rois, L.load_exr(L.RESULTS / spec["t0_model"]), L.load_exr(rr / regime / "T3_refit_at_T3.exr"))
        check(regime, blk, hist[hkey])
        hv = nrc["tracks"][tkey]["evaluation"]["historical_values"]
        if (hv["error_ratio"], hv["gain"], hv["recovers"]) != (blk["error_ratio"], blk["gain"], blk["recovers"]):
            failures.append(f"protocol historical_values for {tkey} differ from the recomputation")

    cb = json.loads(open(L.EXPERIMENT / "protocol/teaset_cross_backbone_locked.json", encoding="utf-8").read())
    fdir = L.RESULTS / cb["frozen_output"]
    gt = {s: (L.load_exr(fdir / s / "gt_A.exr"), L.load_exr(fdir / s / "gt_B.exr")) for s in ("T0", "T3")}
    rois = load_rois(L.RESULTS / cb["gt_output"], ("T0", "T3"))
    rdir = L.RESULTS / cb["rna_output"]
    blk = refit_block(gt, rois, np.load(rdir / "T0_canonical.npy").astype(np.float32),
                      np.load(rdir / "refit_T3_current.npy").astype(np.float32))
    check("rna_common_light", blk, hist["refit_rna_common_light"])
    hv = nrc["tracks"]["rna_common_light"]["evaluation"]["historical_values"]
    if (hv["error_ratio"], hv["gain"], hv["recovers"]) != (blk["error_ratio"], blk["gain"], blk["recovers"]):
        failures.append("protocol historical_values for rna_common_light differ from the recomputation")

    # the rule's threshold semantics on synthetic numbers: <= 1.25 and >= 0.5 are inclusive
    g0 = np.full((4, 4, 3), 0.5, np.float32)
    ys = np.zeros(16, bool)
    ys[:4] = True
    syn_rois = {"T0": {"interaction": (ys, np.arange(4))}, "T3": {"interaction": (ys, np.arange(4))}}
    syn_gt = {"T0": (g0, g0), "T3": (g0 * 1.2, g0 * 1.2)}
    n0 = g0 * 1.02  # a model with a small static error, so the ratio's denominator is non-zero
    b = refit_block(syn_gt, syn_rois, n0, g0 * 1.2 * 1.02)
    if not (b["recovers"] and abs(b["gain"] - 1.02) < 1e-5):
        failures.append(f"synthetic tracking model not recovered: {b}")
    b = refit_block(syn_gt, syn_rois, n0, n0)
    if b["recovers"] or abs(b["gain"]) > 1e-6:
        failures.append(f"synthetic frozen model counted as recovered: {b}")

    # historical inputs this batch reads are the worklog-22 files
    rj = json.loads((rr / "renders.json").read_text(encoding="utf-8"))
    for name, c in rj["checkpoints"].items():
        if L.sha256(L.ROOT / c["path"]) != c["sha256"]:
            failures.append(f"historical checkpoint {name} changed")
    for name in ("T0_canonical.npy", "refit_T3_current.npy"):
        if not (rdir / name).exists():
            failures.append(f"missing {name}")
    for f in ("w21_reference_correction/T0/gt_A.exr", "w21_reference_correction/T3/gt_B.exr", "rna_teaset/features/T3.npz",
              "refit/renders/w21_envmap/T0_retrain_at_T0.exr", "refit/renders/common_light/T0_retrain_at_T0.exr"):
        if not (L.RESULTS / f).exists():
            failures.append(f"missing {f}")

    print("FAIL" if failures else "PASS", *failures, sep="\n  ")
    return 1 if failures else 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
