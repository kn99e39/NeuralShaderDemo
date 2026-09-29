"""Geometry-only measurements for teaset part configurations (no rendering).

For every state in the given state file:
  * per-part translation and the instance bounding box;
  * exact minimum mesh distance between every pair of parts (FCL);
  * the minimum over a 64-step linear sweep from the canonical state, so a
    state is only accepted if no part passes through another on the way;
  * rigidity/topology: each part's vertices minus its translation equal the
    canonical vertices bit-for-bit, with identical face arrays;
  * materials: every BSDF parameter Mitsuba exposes equals the canonical one.
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np

import ednalib as L
import teaset_parts as T


def min_distance(a: tuple, b: tuple) -> tuple[float, list, list]:
    import trimesh

    ma = trimesh.Trimesh(a[0], a[1], process=False)
    mb = trimesh.Trimesh(b[0], b[1], process=False)
    cm = trimesh.collision.CollisionManager()
    cm.add_object("a", ma)
    if cm.in_collision_single(mb):
        return 0.0, [], []
    d, data = cm.min_distance_single(mb, return_data=True)
    return float(d), data.point("a").tolist(), data.point("__external").tolist()


def pair_distances(translations: dict) -> dict:
    """Exact pairwise distances; the tray is also split into floor and rim.

    Pots rest on the flat tray floor (y ~ 0.004), so their whole-tray distance
    is zero in every state.  Floor = tray faces with centroid y <= 0.01, rim =
    the rest; a moved part must keep floor contact and clear the rim.
    """
    meshes = {p: T.part_mesh(p, translations) for p in T.PARTS}
    v, f = meshes["plate"]
    rim = v[f].mean(1)[:, 1] > 0.01
    subs = {"plate_floor": (v, f[~rim]), "plate_rim": (v, f[rim])}
    out = {f"{a}-{b}": min_distance(meshes[a], meshes[b]) for a, b in itertools.combinations(T.PARTS, 2)}
    for p in T.PARTS:
        if p != "plate":
            for k, m in subs.items():
                out[f"{p}-{k}"] = min_distance(meshes[p], m)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True, help="JSON: {state: {part: [tx, ty, tz]}}")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    states = json.loads(open(args.states, encoding="utf-8").read())["states"]
    mi, dr = L.init_upstream()

    canon = {p: T.part_mesh(p) for p in T.PARTS}
    canon_scene = mi.load_dict(T.scene_dict(64))
    canon_bsdf = {k: np.array(v).tolist() for k, v in mi.traverse(canon_scene).items() if "bsdf" in k}
    record = {
        "parts": {p: {"file": T.PARTS[p][0], "object": T.PARTS[p][1], "role": T.PARTS[p][2],
                      "vertices": len(canon[p][0]), "faces": len(canon[p][1]),
                      "bbox_min": canon[p][0].min(0).tolist(), "bbox_max": canon[p][0].max(0).tolist()}
                  for p in T.PARTS},
        "canonical_instance_bbox": [list(canon_scene.shapes()[0].bbox().min), list(canon_scene.shapes()[0].bbox().max)],
        "canonical_bsdf_params": canon_bsdf,
        "states": {},
    }
    for name, translations in states.items():
        pairs = pair_distances(translations)
        sweep = {k: float("inf") for k in pairs}
        if any(np.any(np.asarray(t) != 0) for t in translations.values()):
            for s in np.linspace(0, 1, 65)[1:]:
                step = {p: (np.asarray(t) * s).tolist() for p, t in translations.items()}
                for k, (d, _, _) in pair_distances(step).items():
                    sweep[k] = min(sweep[k], d)
        rigid = {}
        for p in T.PARTS:
            v, f = T.part_mesh(p, translations)
            back = v - np.asarray(translations.get(p, (0, 0, 0)), float)
            rigid[p] = {"max_abs_vertex_residual": float(np.abs(back - canon[p][0]).max()),
                        "faces_identical": bool(np.array_equal(f, canon[p][1]))}
        scene = mi.load_dict(T.scene_dict(64, translations))
        bsdf = {k: np.array(v).tolist() for k, v in mi.traverse(scene).items() if "bsdf" in k}
        record["states"][name] = {
            "translations": translations,
            "instance_bbox": [list(scene.shapes()[0].bbox().min), list(scene.shapes()[0].bbox().max)],
            "pair_min_distance": {k: {"distance": d, "closest_a": a, "closest_b": b} for k, (d, a, b) in pairs.items()},
            "sweep_min_distance": sweep,
            "rigidity_topology": rigid,
            "bsdf_params_identical": bsdf == canon_bsdf,
        }
        print(name, {k: round(d, 4) for k, (d, _, _) in pairs.items()})
    L.write_json(args.out, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
