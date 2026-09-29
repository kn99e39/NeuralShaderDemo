"""Train (refit) 8DNA from scratch on one locked teaset configuration.

    windows/run.ps1 train_8dna_state.py --state T3 --protocol protocol/teaset_frozen_locked.json [upstream train.py args...]

Registers a scene module `scenes.teaset_<state>` whose get_scene(res) is the
upstream teaset scene with that state's part translations (teaset_parts.
scene_dict), then runs the unmodified upstream train.py with
`scene=teaset_<state>`.  Architecture, capacity, loss, dataset sampler and
schedule are upstream's configs/default.yaml unless explicitly overridden on
the command line (feasibility runs only).  The released checkpoint stores
model hyper-parameters only; they equal default.yaml's model block.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
import types

import ednalib as L
import teaset_parts as T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--protocol", required=True)
    args, rest = ap.parse_known_args()
    proto = json.loads(open(L.EXPERIMENT / args.protocol, encoding="utf-8").read())
    translations = proto["states"][args.state]
    L.init_upstream()
    name = f"teaset_{args.state}"
    module = types.ModuleType(f"scenes.{name}")
    module.get_scene = lambda res, *_, **__: T.scene_dict(res, translations)
    sys.modules[f"scenes.{name}"] = module
    import scenes

    setattr(scenes, name, module)
    sys.argv = [str(L.UPSTREAM / "train.py"), f"scene={name}", *rest]
    print("upstream train.py", sys.argv[1:], "translations", translations, flush=True)
    runpy.run_path(str(L.UPSTREAM / "train.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
