"""Run an unmodified upstream RNA script with the one compatibility adapter the released checkpoints need.

    PYTHONPATH=<upstream> python rr_run.py <upstream script.py> [script args...]

Adapter (worklog 35, non-behavioural): the released data/ckpts/lego*.ckpt pickle their hyper-parameters
with the class `neumat.config.BaseNeuMatConfig` - the package's pre-release name. The released code
names the same dataclass `rna.config.BaseNeuMatConfig`. Registering `neumat` / `neumat.config` as
aliases of `rna` / `rna.config` lets torch unpickle the checkpoints; no code, weight or numeric path is
changed. Everything else is the upstream script executed as __main__ (runpy).
"""

from __future__ import annotations

import runpy
import sys


def main() -> None:
    import rna
    import rna.config

    sys.modules.setdefault("neumat", rna)
    sys.modules.setdefault("neumat.config", rna.config)
    script = sys.argv[1]
    sys.argv = [script] + sys.argv[2:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
