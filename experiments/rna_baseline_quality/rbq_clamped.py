"""Worklog-33 control B2a: validation targets under an upstream-like (biased) estimator.

    windows/run.ps1 ../rna_baseline_quality/rbq_clamped.py --config C1|C2 --dest <dir>

The official RNA generator renders with Cycles under (blender_render_dataset.py:230-240 + the
generator .blend files, see rbq_blend_settings.py): sample_clamp_direct 20, sample_clamp_indirect
10, glossy_bounces 4, blur_glossy 1.0, Blackman-Harris 1.5 px. The project bridge is unbiased and
unclamped. This renders val views 0/10/20/30 with rna_bridge.render_view unchanged except that its
tracer is replaced (in this process only) by a copy of rna_bridge.trace_directional that
  * accumulates each sample's direct contribution (camera-hit vertex) and indirect contribution
    (later vertices) separately and clamps each Cycles-style (scale so the max channel <= clamp),
  * stops after 4 bounces past the camera hit (glossy_bounces 4; every teaset BSDF is glossy).
Not emulated: blur_glossy, the pixel filter. Same cameras, lights, spp and render seeds as the
stored H5 (replayed from generate_h5's RNG).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPL = HERE.parents[0] / "8dna_deformation_replication"
sys.path.insert(0, str(REPL))

VIEWS = (0, 10, 20, 30)
CLAMP_DIRECT, CLAMP_INDIRECT, MAX_VERTICES = 20.0, 10.0, 5  # vertices 0..4 = camera hit + 4 bounces


def make_tracer(L):
    def trace_clamped(scene, sampler, ray, to_light, irradiance, rr_depth=5, max_depth=1024, squares=None):
        assert squares is None, "delta directional light only (the dataset regime)"
        mi, dr = L.init_upstream()
        ctx = mi.BSDFContext()
        ray = mi.Ray3f(ray)
        Ld = mi.Spectrum(0.0)
        Li = mi.Spectrum(0.0)
        beta = mi.Spectrum(1.0)
        eta = mi.Float(1.0)
        depth = mi.UInt32(0)
        active = mi.Bool(True)
        loop = mi.Loop(name="clamped directional path tracer", state=lambda: (sampler, ray, Ld, Li, beta, eta, depth, active))
        loop.set_max_iterations(MAX_VERTICES)
        while loop(active):
            si = scene.ray_intersect(ray, active)
            active &= si.is_valid()
            bsdf = si.bsdf(ray)
            emit = active & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
            f = bsdf.eval(ctx, si, si.to_local(to_light), emit)
            unoccluded = ~scene.ray_test(si.spawn_ray(to_light), emit)
            hit = emit & unoccluded
            Ld[hit & (depth == 0)] += beta * f * irradiance
            Li[hit & (depth > 0)] += beta * f * irradiance
            bs, w = bsdf.sample(ctx, si, sampler.next_1d(active), sampler.next_2d(active), active)
            beta[active] *= w
            eta[active] *= bs.eta
            ray = si.spawn_ray(si.to_world(bs.wo))
            depth[active] += 1
            q = dr.minimum(dr.max(beta) * dr.sqr(eta), 0.99)
            rr = depth >= rr_depth
            active &= (sampler.next_1d(active) < q) | ~rr
            beta[rr] = beta * dr.rcp(q)
            active &= dr.any(dr.neq(beta, 0.0)) & (depth < MAX_VERTICES)
        sd = dr.minimum(1.0, CLAMP_DIRECT / dr.maximum(dr.max(Ld), 1e-20))
        si_ = dr.minimum(1.0, CLAMP_INDIRECT / dr.maximum(dr.max(Li), 1e-20))
        return Ld * sd + Li * si_

    return trace_clamped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=("C1", "C2"))
    ap.add_argument("--dest", required=True)  # not --out: PowerShell would parse it as -OutVariable
    args = ap.parse_args()
    import ednalib as L
    import rna_bridge as B
    import teaset_parts as T

    mi, dr = L.init_upstream()
    proto = json.loads((REPL / "protocol" / "teaset_cross_backbone_locked.json").read_text(encoding="utf-8"))
    gen = dict(proto["rna_dataset"])
    if args.config == "C2":
        gen["spp"] = 128
        orig = T.scene_dict

        def scene_dict_rough(res, translations=None, lighting=None, integrator=None):
            d = orig(res, translations, lighting, integrator)
            for part in T.PARTS:
                d["group0"][part]["bsdf"]["bsdf"]["alpha"] = 0.3
            return d

        T.scene_dict = scene_dict_rough
    B.trace_directional = make_tracer(L)  # this process only; rna_bridge.py is unchanged
    state, split = "T0", "val"
    scene = mi.load_dict(T.scene_dict(64, proto["states"][state], None))
    res = gen["resolution"]
    out = Path(args.dest); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(gen["seeds"][split] + 7919 * list(proto["states"]).index(state))
    plan = []
    for v in range(gen["views"][split]):
        h = rng.random() * 0.9999
        r = np.sqrt(max(0.0, 1 - h * h))
        phi = 2 * np.pi * rng.random()
        origin = np.array(gen["camera_lookat"]) + gen["camera_radius"] * np.array([r * np.cos(phi), h, r * np.sin(phi)])
        to_light = B.hemisphere_dirs(rng, 1, 0.999)
        plan.append((origin, to_light, int(rng.integers(1 << 20))))
    for v in VIEWS:
        origin, to_light, seed = plan[v]
        sensor = mi.load_dict(B.sensor_dict(origin, gen["camera_lookat"], gen["fov"], res))
        view = B.render_view(scene, sensor, res, np.repeat(to_light, res * res, 0), gen["spp"], gen["chunk"], seed, gen["irradiance"])
        np.savez_compressed(out / f"{args.config}_view{v}_clamped.npz", color=view["color"].astype(np.float32), alpha=view["alpha"].astype(np.float32))
        print(args.config, v, "clamped", flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush(); sys.stderr.flush()
    import os

    os._exit(rc)
