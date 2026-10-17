"""Architecture Batch 3 (worklog 36): train owned-unit RNA on the native Lego data with the UNCHANGED worklog-32
trainer (rso_train.py), swapping only the owner lookup (Lego triangle-identified surface point -> k-means cell).

    B3_MESH=<lego_mesh.npz> python b3_train.py --config <yml> --partition <partition_L16.npz> --out <dir> [rso_train args]
    optional: B3_RESUME_CKPT=<lightning ckpt> resumes (Trainer.fit(ckpt_path=...)) after a user-requested pause.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "rna_surface_ownership"))
import b3_model as B  # noqa: E402
import rso_model as M  # noqa: E402
import rso_partition  # noqa: E402
import rso_train  # noqa: E402


def main() -> int:
    mesh = os.environ["B3_MESH"]
    rso_partition.Owner = lambda partition: B.LegoOwner(partition, mesh)  # rso_train imports Owner from here inside main()
    M.OwnedSurfaceDataModule = B.LegoOwnedDataModule
    if os.environ.get("B3_RESUME_CKPT"):
        import pytorch_lightning as pl

        _fit = pl.Trainer.fit
        ck = os.environ["B3_RESUME_CKPT"]

        def fit_resumed(self, *a, **k):
            k.setdefault("ckpt_path", ck)
            return _fit(self, *a, **k)

        pl.Trainer.fit = fit_resumed
    return rso_train.main()


if __name__ == "__main__":
    sys.exit(main())
