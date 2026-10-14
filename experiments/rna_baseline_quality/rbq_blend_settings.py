"""Worklog-33 control B1: Cycles settings and materials stored in the upstream-generator .blend
files (custom bpy 3.5.1 in the RNA venv; read-only).

    python rbq_blend_settings.py --out <json> <file.blend> ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("blends", nargs="+")
    args = ap.parse_args()
    import bpy

    keys = ("samples", "max_bounces", "diffuse_bounces", "glossy_bounces", "transmission_bounces", "transparent_max_bounces",
            "blur_glossy", "sample_clamp_direct", "sample_clamp_indirect", "caustics_reflective", "caustics_refractive",
            "pixel_filter_type", "filter_width", "use_denoising", "use_adaptive_sampling", "light_sampling_threshold")
    res = {}
    for b in args.blends:
        bpy.ops.wm.open_mainfile(filepath=b)
        sc = bpy.context.scene
        cyc = {k: (getattr(sc.cycles, k) if hasattr(sc.cycles, k) else None) for k in keys}
        cyc = {k: (v if isinstance(v, (int, float, bool, str)) or v is None else str(v)) for k, v in cyc.items()}
        mats = {}
        for m in bpy.data.materials:
            if not m.use_nodes:
                continue
            for n in m.node_tree.nodes:
                if n.type == "BSDF_PRINCIPLED":
                    g = lambda name: (None if name not in n.inputs else
                                      ("linked" if n.inputs[name].is_linked else float(n.inputs[name].default_value)))
                    mats[f"{m.name}/{n.name}"] = {"roughness": g("Roughness"), "metallic": g("Metallic"), "specular": g("Specular")}
                elif n.type.startswith("BSDF_"):
                    mats.setdefault(f"{m.name}/{n.name}", {"type": n.type})
        res[Path(b).name] = {"cycles": cyc, "render_engine": sc.render.engine, "materials": mats}
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
