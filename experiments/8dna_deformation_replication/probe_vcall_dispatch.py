"""Runtime diagnostic: DrJit 0.4.6 virtual-call dispatch on this GPU.

    windows/run.ps1 probe_vcall_dispatch.py <res> <spp> <vcall_record 0|1>

Renders one released-teaset neural frame and records time and output.  Run
each case in its own process with an external timeout: with VCallRecord off,
renders above 8192 lanes never return on the RTX 5080 (see ednalib).
"""

from __future__ import annotations

import importlib
import sys
import time

import numpy as np

import ednalib as L


def main() -> int:
    res, spp, record = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == "1"
    mi, dr = L.init_upstream()
    dr.set_flag(dr.JitFlag.VCallRecord, record)
    scene = mi.load_dict(importlib.import_module("scenes.teaset").get_scene(res))
    integrator = L.load_neural_integrator("teaset")
    t = time.perf_counter()
    img = np.array(mi.render(scene, integrator=integrator, spp=spp, seed=0))
    out = L.RESULTS / "vcall_probe" / f"probe_{res}_{spp}_{int(record)}.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, img)
    print(f"PROBE res={res} spp={spp} lanes={res * res * spp} VCallRecord={record} "
          f"seconds={time.perf_counter() - t:.2f} mean={img.mean():.5f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
