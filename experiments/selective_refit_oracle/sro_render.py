"""Render snapshots at T3 exactly as the historical refit renders (render_8dna_refits.py).

    windows/run.ps1 ../selective_refit_oracle/sro_render.py --run <train dir> [--steps 0 128 ...] [--spp N]

Upstream neuralpath, mode upstream (the refit's own frame is T3), worklog-21 envmap scene,
512^2, the locked neural spp / chunk / seed 0, every snapshot loaded through the unchanged
upstream load_asset.  Writes <run>/renders/step_XXXXXX.exr and renders.json.  Existing
renders are kept (rendering is deterministic for a fixed seed; worklog 27).
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
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, nargs="*", default=None)
    ap.add_argument("--spp", type=int, default=None, help="smoke only")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    import nrc_records as R
    import ednalib as L

    git = R.git_state(L.ROOT)
    R.require_clean(git, args.allow_dirty or args.spp is not None)
    mi, dr = S.init()
    import teaset_parts as T
    from models.integrator import load_asset

    proto = S.protocol()
    rp = proto["scene"]["render"]
    spp = args.spp or rp["spp"]
    run = Path(args.run)
    snaps = sorted((run / "snapshots").glob("step_*.ckpt"))
    if args.steps is not None:
        snaps = [p for p in snaps if int(p.stem.split("_")[1]) in args.steps]
    scene = mi.load_dict(T.scene_dict(rp["res"], S.states()["T3"]))
    out = run / "renders"
    rec_path = out / "renders.json"
    rec = json.loads(rec_path.read_text()) if rec_path.exists() else {"renders": {}}
    rec["environment"] = L.environment_record()
    rec["git"] = git
    for p in snaps:
        path = out / f"{p.stem}.exr"
        if path.exists() and p.stem in rec["renders"]:
            continue
        integ = L.load_neural_integrator("teaset")
        integ.asset_models = [load_asset("8dna", str(p))]
        integ.is_prepared = False
        img, t = L.render_chunked(scene, integ, spp, rp["chunk"], rp["seed"])
        L.save_exr(path, img)
        rec["renders"][p.stem] = {"snapshot": L.rel(p), "snapshot_sha256": L.sha256(p), "render": L.rel(path),
                                  "seconds": t, "spp": spp, "chunk": rp["chunk"], "seed": rp["seed"]}
        S.write_json(rec_path, rec)
        print(run.name, p.stem, f"{t:.0f}s", flush=True)
    S.write_json(rec_path, rec)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(rc)
