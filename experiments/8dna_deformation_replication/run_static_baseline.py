"""Released static 8DNA baseline versus a matched path-traced reference.

The neural render follows demo/demo.ipynb: `neuralpath`, max_depth 10,
rr_depth 5, the released checkpoint for shape id `instance0`, `spp` samples
accumulated as spp/4 renders with seeds seed..seed+spp/4-1.  The reference is
the same scene rendered with its own integrator (prb for surfaces, prbvolpath
for volumes), which is also what upstream's validation compares against.

Two independent reference renders give the reference noise floor; a second
neural seed gives the neural Monte Carlo floor.
"""

from __future__ import annotations

import argparse
import importlib

import numpy as np

import ednalib as L


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True)
    ap.add_argument("--scene-fn", default="get_scene")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--spp", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--ref-spp", type=int, default=4096)
    ap.add_argument("--ref-chunk", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--integrator", default="neuralpath")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"{args.asset}_{args.scene_fn}_{args.res}"
    out = L.RESULTS / "static_baseline" / tag
    mi, dr = L.init_upstream()
    import torch

    record = {
        "question": "Does the released checkpoint reproduce its asset on this host?",
        "protocol": vars(args),
        "official_repository": L.OFFICIAL_REPOSITORY,
        "official_commit": L.OFFICIAL_COMMIT,
        "checkpoint": L.rel(L.checkpoint_path(args.asset)),
        "checkpoint_sha256": L.sha256(L.checkpoint_path(args.asset)),
        "environment": L.environment_record(),
    }
    scene = mi.load_dict(getattr(importlib.import_module(f"scenes.{args.asset}"), args.scene_fn)(args.res))
    integrator = L.load_neural_integrator(args.asset, args.integrator)

    torch.cuda.reset_peak_memory_stats()
    neural, t_neural = L.render_chunked(scene, integrator, args.spp, args.chunk, args.seed)
    peak_neural = torch.cuda.max_memory_allocated()
    neural_b, _ = L.render_chunked(scene, integrator, args.spp, args.chunk, args.seed + 100_000)
    ref_a, t_ref = L.render_reference(scene, args.ref_spp, args.ref_chunk, args.seed + 200_000)
    ref_b, _ = L.render_reference(scene, args.ref_spp, args.ref_chunk, args.seed + 300_000)
    ref = 0.5 * (ref_a + ref_b)

    for name, img in [("neural", neural), ("neural_seedB", neural_b), ("reference_A", ref_a), ("reference_B", ref_b), ("reference", ref)]:
        L.save_exr(out / f"{name}.exr", img)
    err = np.abs(L.tonemap(neural) - L.tonemap(ref)).mean(-1)
    side = np.concatenate([L.label(L.to_u8(L.tonemap(ref)), "path-traced reference"),
                           L.label(L.to_u8(L.tonemap(neural)), "released 8DNA"),
                           L.label(L.error_map(err, 0.2), "|display diff| (0..0.2)")], 1)
    L.save_png(out / "side_by_side.png", side)

    record.update({
        "finite": bool(np.isfinite(neural).all()),
        "metrics_vs_reference": L.metrics(neural, ref),
        "neural_seed_repeat": L.metrics(neural_b, neural),
        "reference_repeat_A_vs_B": L.metrics(ref_a, ref_b),
        "runtime_seconds": {"neural": t_neural, "reference_one_render": t_ref},
        "peak_torch_cuda_bytes_neural": int(peak_neural),
        "outputs": {k: L.rel(out / f"{k}.exr") for k in ["neural", "neural_seedB", "reference_A", "reference_B", "reference"]}
                   | {"side_by_side": L.rel(out / "side_by_side.png")},
    })
    L.write_json(out / "baseline.json", record)
    print(record["metrics_vs_reference"], record["neural_seed_repeat"], record["reference_repeat_A_vs_B"], record["runtime_seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
