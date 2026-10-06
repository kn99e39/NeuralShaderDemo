"""Quality-vs-time trace of the timed 8DNA rebuild (untimed evaluation).

    windows/run.ps1 ../neural_recompute_cost/nrc_eval_8dna.py --run <run dir>

Every per-epoch snapshot is rendered at T3 exactly as render_8dna_refits.py
renders a refit (upstream neuralpath, mode upstream, the regime's neural spp,
chunk and seed 0, loaded through the unchanged upstream load_asset), in the
worklog-21 envmap regime (primary) and the common-light regime (secondary),
and scored with the unchanged historical rule cross_backbone_eval.refit_block
against the historical T0_retrain renders and references.

Also checks the checkpoint-to-snapshot mapping: the last snapshot must hold
exactly the weights of the schedule's last.ckpt (the historical selection).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "8dna_deformation_replication"))

import nrc_records as R  # noqa: E402

REGIMES = {
    "w21_envmap": {"protocol": "protocol/teaset_frozen_locked.json", "gt": "w21_reference_correction", "rois": "gt_design/v2",
                   "t0_model": "refit/renders/w21_envmap/T0_retrain_at_T0.exr"},
    "common_light": {"protocol": "protocol/teaset_cross_backbone_locked.json", "gt": "frozen/teaset_common_light_8dna",
                     "rois": "gt_design/common_light", "t0_model": "refit/renders/common_light/T0_retrain_at_T0.exr"},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--regimes", nargs="*", default=list(REGIMES))
    args = ap.parse_args()
    run = Path(args.run)
    nrc = json.loads((HERE / "protocol" / "nrc_t3_recompute_v1.json").read_text(encoding="utf-8"))
    import ednalib as L

    git = R.git_state(L.ROOT)
    R.require_clean(git, args.allow_dirty)
    mi, dr = L.init_upstream()
    import torch
    from models.integrator import load_asset

    from cross_backbone_eval import refit_block
    from teaset_frozen_eval import load_rois

    t8 = nrc["tracks"]["8dna_envmap_primary"]["training"]
    snaps = sorted((run / "8dna" / "snapshots").glob("epoch_*.ckpt"))
    if not snaps:
        raise SystemExit("no snapshots")
    out_dir = run / "eval" / "8dna"
    rec = {"environment": L.environment_record(), "git": git, "smoke": args.smoke, "regimes": {}, "snapshots": {}}

    # checkpoint-to-snapshot mapping: last snapshot == last.ckpt weights, bit for bit
    last = run / "8dna" / "train" / t8["experiment_name"] / "last.ckpt"
    ck = torch.load(last, map_location="cpu", weights_only=False)
    sn = torch.load(snaps[-1], map_location="cpu", weights_only=False)
    same = (set(ck["state_dict"]) == set(sn["state_dict"])
            and all(torch.equal(ck["state_dict"][k], sn["state_dict"][k]) for k in ck["state_dict"]))
    exp = run / "8dna" / f"{t8['experiment_name']}_released_layout.ckpt"
    ex = torch.load(exp, map_location="cpu", weights_only=False)
    same_export = all(torch.equal(ex["state_dict"][k], sn["state_dict"][k]) for k in ex["state_dict"])
    rec["mapping"] = {"last_ckpt": L.rel(last), "last_ckpt_sha256": L.sha256(last), "last_ckpt_epoch": int(ck.get("epoch", -1)),
                      "last_snapshot": snaps[-1].name, "last_snapshot_equals_last_ckpt": bool(same),
                      "export": L.rel(exp), "export_sha256": L.sha256(exp), "export_equals_last_snapshot": bool(same_export)}
    if not (same and same_export):
        raise SystemExit(f"snapshot/last.ckpt/export mismatch: {rec['mapping']}")
    for p in snaps:
        rec["snapshots"][p.name] = L.sha256(p)

    for regime in args.regimes:
        spec = REGIMES[regime]
        proto = json.loads(open(L.EXPERIMENT / spec["protocol"], encoding="utf-8").read())
        nr = dict(proto["neural"])
        if args.smoke:
            nr["spp"] = nrc["smoke"]["8dna_eval_spp"]
        gt = {s: (L.load_exr(L.RESULTS / spec["gt"] / s / "gt_A.exr"), L.load_exr(L.RESULTS / spec["gt"] / s / "gt_B.exr"))
              for s in ("T0", "T3")}
        rois = load_rois(L.RESULTS / spec["rois"], ("T0", "T3"))
        n_t0 = L.load_exr(L.RESULTS / spec["t0_model"])
        scene = mi.load_dict(T_scene(L, proto))
        rows = []
        for p in snaps:
            ep = int(p.stem.split("_")[1])
            integ = L.load_neural_integrator(proto["asset"])
            integ.asset_models = [load_asset("8dna", str(p))]
            integ.is_prepared = False
            img, t = L.render_chunked(scene, integ, nr["spp"], nr["chunk"], nr["seed"])
            path = out_dir / regime / f"epoch_{ep:03d}.exr"
            L.save_exr(path, img)
            blk = refit_block(gt, rois, n_t0, img)
            rows.append({"epoch": ep, "snapshot": p.name, "render": L.rel(path), "render_s": t, "spp": nr["spp"], **blk})
            print(regime, ep, f"ratio {blk['error_ratio']:.3f} gain {blk['gain']:.3f} recovers {blk['recovers']} ({t:.0f}s)", flush=True)
            R.write_json(run / "eval" / "8dna_eval.partial.json", dict(rec, regimes=dict(rec["regimes"], **{regime: rows})))
        rec["regimes"][regime] = {"protocol": spec["protocol"], "references": spec["gt"], "rois": spec["rois"],
                                  "t0_model_render": spec["t0_model"], "neural": nr,
                                  "historical": nrc["tracks"]["8dna_envmap_primary" if regime == "w21_envmap" else
                                                              "8dna_common_light_secondary"]["evaluation"]["historical_values"],
                                  "rows": rows}
    R.write_json(run / "eval" / "8dna_eval.json", rec)
    (run / "eval" / "8dna_eval.partial.json").unlink(missing_ok=True)
    return 0


def T_scene(L, proto):
    import teaset_parts as T

    return T.scene_dict(proto["res"], proto["states"]["T3"], proto.get("lighting"))


if __name__ == "__main__":
    rc = main()
    # skip the interpreter teardown (access violation once DrJit and torch are loaded)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
