"""Stage-by-stage runtime smoke test (implementation check only; closes no gate).

Stages: Mitsuba/OptiX reference render -> upstream CUDA extension JIT build
-> released checkpoint load -> 1-spp official neural render.  Each stage is
written to the output JSON before the next starts, so an abort names the
failing stage.
"""

from __future__ import annotations

import argparse
import time
import traceback

import numpy as np

import ednalib as L


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="teaset")
    ap.add_argument("--res", type=int, default=64)
    args = ap.parse_args()
    out = L.RESULTS / "smoke" / f"smoke_{args.asset}.json"
    record: dict = {"asset": args.asset, "res": args.res, "stages": {}}

    def stage(name, fn):
        record["stages"][name] = {"status": "running"}
        L.write_json(out, record)
        t = time.perf_counter()
        try:
            value = fn()
            record["stages"][name] = {"status": "ok", "seconds": time.perf_counter() - t}
            return value
        except Exception as exc:  # recorded, then re-raised
            record["stages"][name] = {"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()}
            L.write_json(out, record)
            raise

    mi, dr = stage("import_upstream", L.init_upstream)
    record["environment"] = L.environment_record()
    import importlib

    scene = stage("load_scene", lambda: mi.load_dict(importlib.import_module(f"scenes.{args.asset}").get_scene(args.res)))
    ref, _ = stage("reference_render", lambda: L.render_reference(scene, 4, 4, 0))
    record["reference_mean"] = float(ref.mean())
    integrator = stage("load_checkpoint", lambda: L.load_neural_integrator(args.asset))
    img, _ = stage("neural_render", lambda: L.render_chunked(scene, integrator, 1, 1, 0))
    record["neural_mean"] = float(img.mean())
    record["neural_finite"] = bool(np.isfinite(img).all())
    L.write_json(out, record)
    print(record["stages"], record["reference_mean"], record["neural_mean"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
