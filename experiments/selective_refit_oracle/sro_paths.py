"""Upstream 8DNA training path samples, one chunk at a time, for any scene.

`PathGen.trace` is the body of upstream `PathSamplingDataset.reload`
(external/8dna26/utils/dataset/path_sampling_dataset.py) for one chunk,
unchanged except that the random inputs are passed in, so the same draws can
be traced in two scenes (common random numbers).  `draw_chunk` draws them in
upstream's order from the global torch RNG, so `generate` reproduces upstream
buffers for the same torch seed (test_sro_paths.py checks this bit for bit).

The returned fields are upstream's processed training fields (xi, wi, xo, wo,
throughput, valid) plus the raw depth and occluded flag.
"""

from __future__ import annotations

import sro_common as S


class PathGen:
    def __init__(self, scene_dict: dict, H: int, bounce: int = 2):
        mi, dr = S.init()
        self.mi = mi
        self.H = H
        self.bounce = bounce
        self.scene = mi.load_dict(scene_dict)
        self.integrator = mi.load_dict({"type": "prbvolpath", "max_depth": -1, "rr_depth": 5, "hide_emitters": False})
        self.bbox = self.scene.shapes()[0].bbox()
        from utils.light_transport import surface_scatter

        self.scatter_fn = surface_scatter

    def trace(self, sample_xi, sample_wi, seed: int, SPP: int) -> dict:
        import torch
        import torch.nn.functional as NF
        from utils import sample_cos, sample_sphere

        mi = self.mi
        sensor = self.scene.sensors()[0]
        xi = sample_sphere(sample_xi)
        s, t = mi.coordinate_system(xi)
        M = torch.stack([s.torch(), t.torch(), xi], -1)
        wi = sample_cos(sample_wi)
        wi = (M @ wi.unsqueeze(-1)).squeeze(-1)
        d = -wi
        o = xi - mi.math.RayEpsilon * d
        ray = mi.Ray3f(mi.Vector3f(o), mi.Vector3f(d))
        sampler, _ = self.integrator.prepare(sensor, seed, SPP, [])
        throughput, depth, xi, xo, wo, occluded = self.scatter_fn(
            self.integrator, scene=self.scene, sampler=sampler.clone(), ray=ray,
            depth=mi.UInt32(0), active=mi.Bool(True))
        depth_t = depth.torch().long()
        valid = depth >= 1
        occluded |= depth < self.bounce
        last_ray = mi.Ray3f(xo, wo)
        intersected, _, tmax = self.bbox.ray_intersect(last_ray)
        valid &= intersected
        xo = last_ray(tmax)
        throughput = throughput.torch()
        valid = valid.torch().bool()
        xi = xi.torch()
        xo = (xo - self.bbox.center()).torch()
        wo = wo.torch()
        occ = occluded.torch().bool().unsqueeze(-1)
        wo = NF.normalize(wo, dim=-1)
        xo = NF.normalize(xo, dim=-1)
        wo = torch.where(occ, xo, wo)
        throughput = torch.where(occ, 0, throughput)
        throughput[throughput.isnan()] = 0
        return {"xi": xi, "wi": wi, "xo": xo, "wo": wo, "throughput": throughput, "valid": valid,
                "depth": depth_t, "occluded": occ.squeeze(-1)}


def strata(N: int, H: int, device="cuda"):
    import torch

    i = torch.arange(H, device=device)
    k = torch.arange(N, device=device)
    s_xi = torch.stack(torch.meshgrid(i, i, indexing="ij"), -1).float().reshape(-1, 1, 2)
    s_wi = torch.stack(torch.meshgrid(k, k, indexing="ij"), -1).float().reshape(1, -1, 2)
    return s_xi, s_wi


def draw_chunk(idx: int, N: int, H: int, SPP: int, s_xi, s_wi, device="cuda"):
    """Random inputs of chunk `idx` in upstream reload's draw order (global torch RNG)."""
    import torch

    sample_xi = torch.rand(H * H, SPP, 2, device=device).add(s_xi).div(H).reshape(-1, 2)
    sample_wi = torch.rand(H * H, SPP, 2, device=device).add(s_wi[:, idx * SPP:(idx + 1) * SPP]).div(N).reshape(-1, 2)
    seed = torch.randint(0, torch.iinfo(torch.int32).max, (1,)).item()
    return sample_xi, sample_wi, seed


def n_chunks(N: int, SPP: int) -> int:
    return (N * N) // SPP


def generate(gen: PathGen, N: int, H: int, SPP: int, device="cuda"):
    """Yield (idx, fields) for every chunk of one upstream reload (global RNG must be seeded by the caller)."""
    s_xi, s_wi = strata(N, H, device)
    for idx in range(n_chunks(N, SPP)):
        a, b, seed = draw_chunk(idx, N, H, SPP, s_xi, s_wi, device)
        yield idx, gen.trace(a, b, seed, SPP)
