"""Timed RNA training-target generation for one locked teaset state (instrumentation only).

    windows/run.ps1 ../neural_recompute_cost/nrc_rna_h5_timed.py --events <events.jsonl> \
        --protocol <derived protocol, path relative to experiments/8dna_deformation_replication> \
        --state T3 --split train|val

Runs the historical producer ``rna_bridge.generate_h5`` (worklog 22) unchanged.
The derived protocol is the locked ``teaset_cross_backbone_locked.json`` with
only ``rna_dataset_dir`` redirected to this batch's run directory, so the
historical datasets are never overwritten; every generation setting
(views, spp, chunk, seeds, cameras, lights) is the locked one.

Phases: process_setup (interpreter start -> imports/runtime init),
generate_h5, and inside it one view_render per view.  Scene preparation is
derived as generate_h5 begin -> first view_render begin (the state's scene
load, H5 file creation and the first sensor); the remainder of generate_h5
(sensor loads, H5 writes, camera sampling, attributes, sha256 of the file) is
reported as other.  (Mitsuba caches attribute lookups per variant, so
mi.load_dict itself is not wrapped.)
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "8dna_deformation_replication"))

import nrc_records as R  # noqa: E402


def main() -> int:
    created = R.process_create_time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--split", choices=("train", "val"), required=True)
    ap.add_argument("--stage", default=None)
    ap.add_argument("--allow-dirty", action="store_true", help="smoke runs only; never evidence")
    args = ap.parse_args()
    log = R.EventLog(args.events, args.stage or f"rna_h5_{args.split}")
    log.emit("process_created", t_created=created)
    import ednalib as L

    git = R.git_state(L.ROOT)
    R.require_clean(git, args.allow_dirty)
    poller = R.GpuMemoryPoller()
    poller.start()
    mi, dr = L.init_upstream()
    import torch

    import rna_bridge as B

    log.emit("imports_done")
    log.emit("environment", environment=L.environment_record(), git=git, argv=sys.argv[1:])

    n = {"view": 0}
    orig_render = B.render_view

    @functools.wraps(orig_render)
    def render_view(*a, **k):
        v = n["view"]
        log.begin("view_render", key=v)
        try:
            return orig_render(*a, **k)
        finally:
            torch.cuda.synchronize()
            log.end("view_render", key=v)
            n["view"] += 1

    B.render_view = render_view
    log.begin("generate_h5")
    B.generate_h5(argparse.Namespace(protocol=args.protocol, state=args.state, split=args.split))
    log.end("generate_h5")
    proto = __import__("json").loads(open(L.EXPERIMENT / args.protocol, encoding="utf-8").read())
    out = L.RESULTS / proto["rna_dataset_dir"] / f"teaset_{args.state}_{args.split}.h5"
    mem = poller.halt()
    mem.update({"torch_max_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
                "host_peak_working_set_mb": R.host_peak_memory_mb()})
    # generate_h5 already hashed the file into <name>.cameras.json; no second pass here
    cams = __import__("json").loads(out.with_suffix(".cameras.json").read_text(encoding="utf-8"))
    log.emit("output", path=L.rel(out), sha256=cams["sha256"], bytes=out.stat().st_size, views=n["view"])
    log.emit("memory", **mem)
    log.emit("process_done")
    log.close()
    return 0


if __name__ == "__main__":
    rc = main()
    # skip the interpreter teardown, which crashes on this Windows setup once DrJit and torch are loaded
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
