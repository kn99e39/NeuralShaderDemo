"""Worklog 36: fill the per-H5 owner caches (b3_model.cached_owners) once, so training and evaluation use identical
owners and the trainer's data setup does not repeat the triangle lookup. Usage: python b3_precompute_owners.py <partition> <mesh> <h5> ..."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b3_model as B  # noqa: E402

own = B.LegoOwner(sys.argv[1], sys.argv[2])
for h5 in sys.argv[3:]:
    t0 = time.time()
    ds = B.LegoOwnedSurfaceDataset(datafile=h5, use_for_visibility="diffuse_direct", use_positions=True, device="cpu", ops_device="cuda", owner=own)
    print(h5, "slices", ds.total_slices, "owner counts", ds.owner_counts.tolist(), "max dist %.4f" % ds.owner_max_dist, "%.0f s" % (time.time() - t0), flush=True)
    del ds
