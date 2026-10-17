"""Architecture Batch 3 (worklog 36): inspect and export the official Lego surface for ownership (custom bpy; read-only).

    cd results/rna_reproduction/work && python b3_mesh_inspect.py --out <dir>

Exports the evaluated (modifiers applied) world-space triangle mesh of every renderable mesh object:
vertices, triangles, per-triangle material slot, object index and connected-component id (shared
vertices), plus a JSON summary (counts, areas per material/component). Nothing in the .blend changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import bpy

    bpy.ops.wm.open_mainfile(filepath="data/scenes/Lego_rough_aovs_normalized.blend")
    dg = bpy.context.evaluated_depsgraph_get()
    V, F, M, O = [], [], [], []
    mats = []
    off = 0
    objs = [o for o in bpy.context.scene.objects if o.type == "MESH" and not o.hide_render]
    for oi, o in enumerate(objs):
        e = o.evaluated_get(dg)
        me = e.to_mesh()
        me.calc_loop_triangles()
        mw = np.array(o.matrix_world)
        v = np.array([mw @ np.array([*x.co, 1.0]) for x in me.vertices])[:, :3]
        tri = np.array([t.vertices[:] for t in me.loop_triangles], np.int64)
        mi = np.array([t.material_index for t in me.loop_triangles], np.int64)
        names = [s.material.name if s.material else "" for s in o.material_slots]
        base = len(mats); mats += [f"{o.name}/{n}" for n in names]
        V.append(v); F.append(tri + off); M.append(mi + base); O.append(np.full(len(tri), oi)); off += len(v)
        e.to_mesh_clear()
    V = np.concatenate(V); F = np.concatenate(F); M = np.concatenate(M); O = np.concatenate(O)
    # connected components over shared vertices (union-find)
    parent = np.arange(len(V))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    for a, b, c in F:
        ra, rb, rc = find(a), find(b), find(c)
        parent[rb] = ra; parent[rc] = ra
    comp_v = np.array([find(i) for i in range(len(V))])
    _, comp = np.unique(comp_v[F[:, 0]], return_inverse=True)
    area = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "lego_mesh.npz", V=V.astype(np.float64), F=F, material=M, obj=O, comp=comp, area=area)
    summ = {"objects": [o.name for o in objs], "vertices": int(len(V)), "triangles": int(len(F)), "materials": mats,
            "area_total": float(area.sum()), "bbox": [V.min(0).tolist(), V.max(0).tolist()],
            "area_per_material": {mats[k]: float(area[M == k].sum()) for k in np.unique(M)},
            "tris_per_material": {mats[k]: int((M == k).sum()) for k in np.unique(M)},
            "components": int(comp.max() + 1),
            "component_area_quantiles": np.quantile(np.bincount(comp, weights=area), [0, 0.5, 0.9, 0.99, 1]).tolist()}
    (out / "lego_mesh.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
