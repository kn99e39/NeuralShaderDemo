"""Render the 8DNA same-state refit checkpoints (protocol/teaset_8dna_refit_locked.json).

For each regime (worklog-21 envmap, cross-backbone common light) renders
  T0_retrain at T0  and  T3_refit at T3
with the upstream neuralpath integrator in 'upstream' mode (a refit's own
frame is its training state, so no pullback), the regime's neural spp and
seed 0, loading last.ckpt through the unchanged upstream load_asset.
"""

from __future__ import annotations

import json

import torch

import ednalib as L
import teaset_parts as T


def as_released_layout(ckpt_path, out_path):
    """Make a train.py checkpoint loadable by upstream `load_asset` on torch 2.8.

    The state_dict already has the layout load_asset expects ('model.<param>',
    one prefix it strips), so the weights are copied through untouched.  Two
    container details differ from the released files and are normalised here:
    train.py stores hyper_parameters as an OmegaConf DictConfig, and its
    Lightning checkpoint also carries callback and optimiser objects.  torch 2.8
    loads with weights_only=True by default and refuses both, so this writes
    only the two fields load_asset reads, hyper_parameters as a plain dict.
    """
    from omegaconf import OmegaConf

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ck["hyper_parameters"]
    hp = OmegaConf.to_container(hp, resolve=True) if OmegaConf.is_config(hp) else dict(hp)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": dict(ck["state_dict"]), "hyper_parameters": {"model": hp["model"]}}, out_path)
    return out_path

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
    loadable = {}
    for name, spec in refit["trainings"].items():
        ckpt = root / name / "last.ckpt"
        loadable[name] = as_released_layout(ckpt, out / f"{name}_released_layout.ckpt")
        record["checkpoints"][name] = {"path": L.rel(ckpt), "sha256": L.sha256(ckpt), "state": spec["state"],
                                       "released_layout_copy": L.rel(loadable[name]),
                                       "released_layout_sha256": L.sha256(loadable[name])}
    for regime, pfile in REGIMES.items():
        proto = json.loads(open(L.EXPERIMENT / pfile, encoding="utf-8").read())
        nr = proto["neural"]
        for name, spec in refit["trainings"].items():
            s = spec["state"]
            scene = mi.load_dict(T.scene_dict(proto["res"], proto["states"][s], proto.get("lighting")))
            integ = L.load_neural_integrator(proto["asset"])
            integ.asset_models = [load_asset("8dna", str(loadable[name]))]
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
