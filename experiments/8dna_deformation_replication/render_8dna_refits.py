"""Render the 8DNA same-state refit checkpoints (protocol/teaset_8dna_refit_locked.json).

For each regime (worklog-21 envmap, cross-backbone common light) renders
  T0_retrain at T0  and  T3_refit at T3
with the upstream neuralpath integrator in 'upstream' mode (a refit's own
frame is its training state, so no pullback), the regime's neural spp and
seed 0, loading last.ckpt through the unchanged upstream load_asset.
"""

from __future__ import annotations

import json

import ednalib as L
import teaset_parts as T

REGIMES = {
    "w21_envmap": "protocol/teaset_frozen_locked.json",
    "common_light": "protocol/teaset_cross_backbone_locked.json",
}


def main() -> int:
    refit = json.loads(open(L.EXPERIMENT / "protocol/teaset_8dna_refit_locked.json", encoding="utf-8").read())
    mi, dr = L.init_upstream()
    from models.integrator import load_asset

    root = L.RESULTS / "refit"
    out = root / "renders"
    record = {"environment": L.environment_record(), "checkpoints": {}, "renders": {}}
    for name, spec in refit["trainings"].items():
        ckpt = root / name / "last.ckpt"
        record["checkpoints"][name] = {"path": L.rel(ckpt), "sha256": L.sha256(ckpt), "state": spec["state"]}
    for regime, pfile in REGIMES.items():
        proto = json.loads(open(L.EXPERIMENT / pfile, encoding="utf-8").read())
        nr = proto["neural"]
        for name, spec in refit["trainings"].items():
            s = spec["state"]
            scene = mi.load_dict(T.scene_dict(proto["res"], proto["states"][s], proto.get("lighting")))
            integ = L.load_neural_integrator(proto["asset"])
            integ.asset_models = [load_asset("8dna", str(root / name / "last.ckpt"))]
            integ.is_prepared = False
            img, t = L.render_chunked(scene, integ, nr["spp"], nr["chunk"], nr["seed"])
            path = out / regime / f"{name}_at_{s}.exr"
            L.save_exr(path, img)
            record["renders"][f"{regime}/{name}"] = {"path": L.rel(path), "seconds": t, "spp": nr["spp"]}
            print(regime, name, f"{t:.0f}s", flush=True)
    L.write_json(out / "renders.json", record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
