"""Worklog-35 native RNA runtime qualification (RNA venv with the official custom bpy; read-only).

    cd results/rna_reproduction/work && PYTHONPATH=<upstream> python rr_qualify.py --out <json>

Checks, without changing anything: the official Lego scene opens; its AOV passes, view layers,
render settings and principled materials; the custom Cycles query-file hook exists; both released
checkpoints load through the upstream Lightning module; their hyper-parameters and tensor shapes
match config/train/lego_vis.yml and lego_vis_rt.yml; a forward pass on random inputs is finite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(p, n=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(n)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import bpy
    import torch
    from rna import config, interfaces

    rec = {"bpy": bpy.app.version_string, "torch": torch.__version__, "cuda_device": torch.cuda.get_device_name(0)}
    blend = "data/scenes/Lego_rough_aovs_normalized.blend"
    bpy.ops.wm.open_mainfile(filepath=blend)
    sc = bpy.context.scene
    vl = sc.view_layers[0]
    rec["scene"] = {
        "file": blend, "sha256": sha(blend), "engine": sc.render.engine,
        "objects": sorted({o.type for o in sc.objects}), "n_mesh": sum(o.type == "MESH" for o in sc.objects),
        "view_layers": [v.name for v in sc.view_layers], "aovs": [(a.name, a.type) for a in vl.aovs],
        "passes": {k: getattr(vl, k) for k in ("use_pass_combined", "use_pass_diffuse_direct", "use_pass_normal", "use_pass_position") if hasattr(vl, k)},
        "cycles": {k: (getattr(sc.cycles, k) if not hasattr(getattr(sc.cycles, k, None), "__len__") or isinstance(getattr(sc.cycles, k), str) else str(getattr(sc.cycles, k)))
                   for k in ("samples", "max_bounces", "glossy_bounces", "diffuse_bounces", "blur_glossy", "sample_clamp_direct", "sample_clamp_indirect", "pixel_filter_type", "filter_width")},
        "world_has_query_hook": hasattr(sc.world.cycles, "query_filename"),
        "materials": {},
    }
    for m in bpy.data.materials:
        if m.use_nodes:
            for nd in m.node_tree.nodes:
                if nd.type == "BSDF_PRINCIPLED":
                    g = lambda k: None if k not in nd.inputs else ("linked" if nd.inputs[k].is_linked else round(float(nd.inputs[k].default_value), 4))
                    rec["scene"]["materials"][f"{m.name}/{nd.name}"] = {"roughness": g("Roughness"), "metallic": g("Metallic"), "specular": g("Specular")}
    rec["checkpoints"] = {}
    try:  # unchanged environment: does the released checkpoint unpickle with the released code?
        torch.load("data/ckpts/lego.ckpt", map_location="cpu", weights_only=False)
        rec["unaliased_load"] = "ok"
    except Exception as e:  # noqa: BLE001
        rec["unaliased_load"] = f"{type(e).__name__}: {e}"
    import sys
    import rna
    import rna.config

    sys.modules.setdefault("neumat", rna); sys.modules.setdefault("neumat.config", rna.config)  # rr_run.py adapter
    for name, cfg in (("lego.ckpt", "config/train/lego_vis.yml"), ("lego_rt.ckpt", "config/train/lego_vis_rt.yml")):
        p = Path("data/ckpts") / name
        conf = config.get_neumat_config(config_filename=cfg)
        ck = torch.load(p, map_location="cpu", weights_only=False)
        sd = {k: v for k, v in ck["state_dict"].items() if k.startswith("model.")}
        mod = interfaces.NeuralSurfaceTriplaneModule.load_from_checkpoint(str(p), map_location="cuda", strict=True)
        mod.eval()
        n = 4096
        pos = torch.rand(1, n, 3, device="cuda"); d = torch.nn.functional.normalize(torch.randn(3, 1, n, 3, device="cuda"), dim=-1)
        with torch.no_grad():
            out = mod.model(pos, d[0], d[1], d[2])
        rec["checkpoints"][name] = {
            "sha256": sha(p), "epoch": ck.get("epoch"), "global_step": ck.get("global_step"),
            "hyper_parameters_keys": sorted(ck.get("hyper_parameters", {}).keys())[:20],
            "callbacks": list(ck.get("callbacks", {}).keys())[:5],
            "params": int(sum(v.numel() for v in sd.values())),
            "shapes": {k: list(v.shape) for k, v in sd.items() if "feature_grid" in k or k.endswith("layers.0.weight") or k.endswith(".weight") and "layers" in k},
            "config_mlp_width": int(conf.model_params.mlp_hidden_layer_neurons), "config_res": list(conf.model_params.resolution),
            "forward_finite": bool(torch.isfinite(out).all()), "out_shape": list(out.shape),
        }
    Path(args.out).write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps(rec, indent=1, default=str)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
