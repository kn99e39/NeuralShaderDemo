"""Post-hoc analysis of existing worklog-30 outputs (no training; not a protocol quantity).

    windows/run.ps1 ../selective_refit_oracle/sro_posthoc.py --root <batch results dir>

1. Ownership swap from existing checkpoints: render at T3 (protocol render settings)
     H_shared = C_s0 final shared tensors + released triplane
     H_trip   = released shared tensors + C_s0 final triplane
   and score them with sro_metrics' region / tracking functions, to see which part carries the
   global warm-start's recovery.  Labelled post hoc; it changes no verdict.
2. Per-step optimisation cost of every arm (optimisation seconds / steps), incl. the E_mask cost run.
3. Relative parameter change of C_s0 per tensor group.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import sro_common as S


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    import nrc_records as R
    import ednalib as L

    git = R.git_state(L.ROOT)
    R.require_clean(git)
    mi, dr = S.init()
    import numpy as np
    import torch
    import teaset_parts as T
    from models.integrator import load_asset
    from cross_backbone_eval import refit_block
    from teaset_frozen_eval import load_rois
    from sro_metrics import change_tracking, region_errors

    root = Path(args.root)
    out_dir = root / "posthoc"
    out_dir.mkdir(exist_ok=True)
    rel = torch.load(S.released_checkpoint(), map_location="cpu", weights_only=False)
    c = torch.load(root / "runs" / "C_s0" / "snapshots" / "step_032768.ckpt", map_location="cpu", weights_only=False)
    hyb = {}
    for name, take_c in (("H_shared_from_C", lambda k: k != S.TRIPLANE_KEY), ("H_triplane_from_C", lambda k: k == S.TRIPLANE_KEY)):
        sd = {k: (c["state_dict"][k] if take_c(k) else rel["state_dict"][k]) for k in rel["state_dict"]}
        p = out_dir / f"{name}.ckpt"
        torch.save({"state_dict": sd, "hyper_parameters": rel["hyper_parameters"]}, p)
        hyb[name] = p

    proto = S.protocol()
    rp = proto["scene"]["render"]
    scene = mi.load_dict(T.scene_dict(rp["res"], S.states()["T3"]))
    H = S.ROOT / "results" / "8dna_replication"
    gt = {s: (L.load_exr(H / "w21_reference_correction" / s / "gt_A.exr"), L.load_exr(H / "w21_reference_correction" / s / "gt_B.exr")) for s in ("T0", "T3")}
    g0, g3 = 0.5 * (gt["T0"][0] + gt["T0"][1]), 0.5 * (gt["T3"][0] + gt["T3"][1])
    rois = load_rois(H / "gt_design" / "v2", ("T0", "T3"))
    n0 = L.load_exr(H / "frozen" / "teaset_locked" / "T0" / "8dna_upstream.exr")
    regions = dict(np.load(root / "regions" / "regions.npz"))
    rec = {"schema": "sro_posthoc/v1", "git": git, "label": "post hoc analysis; not a protocol quantity", "swap": {}}
    for name, p in hyb.items():
        integ = L.load_neural_integrator("teaset")
        integ.asset_models = [load_asset("8dna", str(p))]
        integ.is_prepared = False
        img, t = L.render_chunked(scene, integ, rp["spp"], rp["chunk"], rp["seed"])
        L.save_exr(out_dir / f"{name}.exr", img)
        row = region_errors(img, g3, regions, rois)
        blk = refit_block(gt, rois, n0, img)
        row["rule"] = {k: blk[k] for k in ("error_ratio", "gain", "recovers")}
        row["tracking"] = {k: change_tracking(img, n0, g3, g0, regions[k]) for k in ("R_aff_stat", "R_unaff")}
        row["render_s"] = t
        rec["swap"][name] = row
        print(name, {k: round(row[k], 4) for k in ("R_aff", "R_aff_stat", "R_mover", "R_unaff")}, row["rule"], flush=True)

    # per-step optimisation cost
    cost = {}
    for ev in sorted(list((root / "runs").glob("*/events.jsonl")) + list((root / "cost").glob("*/events.jsonl"))):
        iv = R.intervals(R.load_events(ev))
        opt = [r for r in iv if r["phase"] == "optimization" and r["seconds"] is not None]
        steps = sum(int(r["end_event"]["end_step"]) - int(r["key"]) for r in opt)
        secs = sum(r["seconds"] for r in opt)
        cost[ev.parent.name] = {"steps": steps, "optimization_s": secs, "ms_per_step": 1000 * secs / max(steps, 1)}
    rec["per_step_cost"] = cost

    # relative change per tensor group, C_s0 final vs released
    grp = {}
    for k, v in rel["state_dict"].items():
        g = ".".join(k.split(".")[1:3]) if "flow" in k else ".".join(k.split(".")[1:2])
        d = (c["state_dict"][k].float() - v.float())
        a = grp.setdefault(g, [0.0, 0.0])
        a[0] += float((d ** 2).sum()); a[1] += float((v.float() ** 2).sum())
    rec["C_s0_relative_change_l2"] = {g: (a[0] ** 0.5) / max(a[1] ** 0.5, 1e-30) for g, a in grp.items()}
    S.write_json(out_dir / "posthoc.json", rec)
    print(json.dumps(rec["per_step_cost"], indent=1), json.dumps(rec["C_s0_relative_change_l2"], indent=1), flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(rc)
