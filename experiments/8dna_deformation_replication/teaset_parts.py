"""Teaset part-rigid configurations and the frozen-8DNA correspondence adapter.

The released `teaset` asset is one Mitsuba instance (`instance0`) of a
shapegroup holding four rigid rough-nickel meshes.  A geometry configuration
here translates whole parts inside that shapegroup; topology, materials,
camera and lighting stay the upstream `get_scene` values.

Correspondence contract (translation-only part motion)
------------------------------------------------------
Upstream queries the frozen model for an instance with model-to-world
M = [R | t] as

    xi_can = R^T (xi - t),   wi_can = R^T wi,
    xo_dir = R xo_can,       wo = R wo_can,

and projects the exit direction from the envelope box's centre onto that box
(`EightDNA.eval_asset` / `sample_asset`).  `PartRigidAsset` applies exactly
that code per hit part, with M_p = [I | t_p] for a part translated by t_p, so
a current hit on part p is queried at its analytic canonical point and the
direction frame is unchanged.  The envelope box is

    'attached' (primary):  canonical asset box + t_p, i.e. the whole canonical
                           asset frame rides with the hit part, which is what
                           upstream would do for an instance translated by t_p;
    'fixed' (declared sensitivity): the canonical asset box, unmoved.

At the canonical state every t_p = 0 and both reduce to the upstream call.
Parts are identified per lane by the mesh pointer Mitsuba reports in
`si.shape` for an instanced hit.
"""

from __future__ import annotations

import numpy as np

import ednalib as L

PARTS = {
    # scene-dict key inside group0: (obj file, object name in the file, role)
    "plate": ("teaplate.obj", "SM_Tray", "tray"),
    "teapot2": ("teapot2.obj", "SM_Milk_Pot", "milk pot"),
    "teapot3": ("teapot3.obj", "SM_Biscuit_Tin", "biscuit tin"),
    "teapot4": ("teapot4.obj", "SM_Tea_Pot", "tea pot"),
}


def _t(translations: dict | None, part: str) -> np.ndarray:
    return np.asarray((translations or {}).get(part, (0.0, 0.0, 0.0)), dtype=np.float64)


def scene_dict(res: int, translations: dict | None = None, lighting: dict | None = None,
               integrator: dict | None = None) -> dict:
    """Upstream teaset get_scene(res) with per-part translations inside group0.

    Parts with zero translation get no `to_world` key, so the canonical state is
    the unmodified upstream dictionary.  `lighting=None` keeps the upstream
    envmap (worklog 21).  `lighting={"type": "directional", "to_light": [x, y, z],
    "irradiance": E}` replaces it with a black world and one directional light
    (the cross-backbone common-light regime); geometry, camera and materials
    are unchanged.  An optional "reference_integrator" replaces the scene's
    reference integrator (prb) for that regime; `integrator` overrides it for
    any regime.
    """
    mi, _ = L.init_upstream()
    from scenes.teaset import get_scene

    d = get_scene(res)
    for part in PARTS:
        t = _t(translations, part)
        if np.any(t != 0):
            d["group0"][part]["to_world"] = mi.ScalarTransform4f.translate([float(v) for v in t])
    if lighting is not None:
        if lighting["type"] != "directional":
            raise ValueError(lighting)
        del d["background"]
        to_light = np.asarray(lighting["to_light"], float)
        to_light /= np.linalg.norm(to_light)
        d["sun"] = {"type": "directional", "direction": [float(v) for v in -to_light],
                    "irradiance": {"type": "rgb", "value": float(lighting["irradiance"])}}
        if "reference_integrator" in lighting:
            d["integrator"] = dict(lighting["reference_integrator"])
    if integrator is not None:
        d["integrator"] = dict(integrator)
    return d


def part_mesh(part: str, translations: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(vertices, faces) of one part as Mitsuba loads it, in the given configuration."""
    mi, _ = L.init_upstream()
    mesh = mi.load_dict({"type": "obj", "filename": str(L.UPSTREAM / "scenes" / "teaset" / PARTS[part][0])})
    params = mi.traverse(mesh)
    v = np.array(params["vertex_positions"], dtype=np.float64).reshape(-1, 3) + _t(translations, part)
    f = np.array(params["faces"], dtype=np.int64).reshape(-1, 3)
    return v, f


def _fibonacci_dirs(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.cos(phi), np.sin(theta) * np.sin(phi)], -1)


def shape_ids(si) -> "object":
    """Per-lane registry id of the hit mesh (0 for misses)."""
    mi, dr = L.init_upstream()
    return dr.reinterpret_array_v(mi.UInt32, si.shape)


def part_shape_ids(scene, translations: dict | None = None) -> dict[str, int]:
    """Registry id of each part's mesh inside `scene`, found by probing its vertices.

    Rays aimed at a part's own vertices from 48 directions; a hit that lands on
    the aimed vertex belongs to that part.  The id must be unanimous among
    such hits and distinct across parts.
    """
    mi, dr = L.init_upstream()
    dirs = _fibonacci_dirs(48)
    ids: dict[str, int] = {}
    for part in PARTS:
        v, _ = part_mesh(part, translations)
        v = v[:: max(1, len(v) // 300)]
        target = np.repeat(v, len(dirs), 0)
        d = np.tile(dirs, (len(v), 1))
        origin = target + 4.0 * d
        ray = mi.Ray3f(mi.Point3f(*origin.T.astype(np.float32)), mi.Vector3f(*(-d).T.astype(np.float32)))
        si = scene.ray_intersect(ray)
        hit = np.stack([np.array(si.p.x), np.array(si.p.y), np.array(si.p.z)], -1)
        ok = np.array(si.is_valid()) & (np.linalg.norm(hit - target, axis=1) < 1e-4)
        sid = np.array(shape_ids(si))[ok]
        values, counts = np.unique(sid, return_counts=True)
        if len(values) == 0:
            raise RuntimeError(f"no probe ray reached {part}")
        best = values[np.argmax(counts)]
        if counts.max() < 0.98 * counts.sum():
            raise RuntimeError(f"ambiguous shape id for {part}: {dict(zip(values.tolist(), counts.tolist()))}")
        ids[part] = int(best)
    if len(set(ids.values())) != len(ids):
        raise RuntimeError(f"part shape ids not distinct: {ids}")
    return ids


class PartRigidAsset:
    """Wraps the released EightDNA model; dispatches eval/sample per hit part."""

    def __init__(self, model, part_ids: dict[str, int], translations: dict | None, canonical_bbox, envelope: str):
        mi, dr = L.init_upstream()
        import torch

        if envelope not in ("attached", "fixed"):
            raise ValueError(envelope)
        self.model = model
        self.parts = []
        for part, sid in part_ids.items():
            t = _t(translations, part)
            m2w = torch.zeros(3, 4, device="cuda")
            m2w[:, :3] = torch.eye(3, device="cuda")
            m2w[:, 3] = torch.tensor(t, dtype=torch.float32, device="cuda")
            if envelope == "attached":
                lo, hi = np.array(canonical_bbox.min) + t, np.array(canonical_bbox.max) + t
                bbox = mi.ScalarBoundingBox3f(mi.ScalarPoint3f(*lo), mi.ScalarPoint3f(*hi))
            else:
                bbox = canonical_bbox
            self.parts.append((part, mi.UInt32(sid), m2w, mi.BoundingBox3f(bbox.min, bbox.max)))

    def _dispatch(self, si, active):
        mi, dr = L.init_upstream()
        sid = shape_ids(si)
        covered = mi.Bool(False)
        for part, pid, m2w, bbox in self.parts:
            act = active & dr.eq(sid, pid)
            covered |= act
            yield act, m2w, bbox
        if dr.any(active & ~covered):
            raise RuntimeError("asset lane hit a mesh that is not a declared part")

    def eval_asset(self, scene, si, m2w, bbox, active_asset, channel, sample2_1, sample2_2):
        _, dr = L.init_upstream()
        result = None
        for act, m2w_p, bbox_p in self._dispatch(si, active_asset):
            if not dr.any(act):
                continue
            r = self.model.eval_asset(scene, si, m2w_p, bbox_p, act, channel, sample2_1, sample2_2)
            if result is None:
                result = list(r)
            else:
                for k in range(4):
                    result[k][act] = r[k]
        return tuple(result)

    def sample_asset(self, scene, si, m2w, bbox, active_asset, channel, sample1, sample2_1, sample2_2,
                     bsdf_si, bsdf_sample, bsdf_weight):
        _, dr = L.init_upstream()
        for act, m2w_p, bbox_p in self._dispatch(si, active_asset):
            if dr.any(act):
                self.model.sample_asset(scene, si, m2w_p, bbox_p, act, channel, sample1, sample2_1, sample2_2,
                                        bsdf_si, bsdf_sample, bsdf_weight)


def configure(integrator, base_model, scene, mode: str, translations: dict | None = None, canonical_bbox=None):
    """Point an upstream neuralpath integrator at `scene` in one query mode.

    mode 'upstream'  : released code path unchanged (current-world queries,
                       current instance box) -- canonical baseline, and the
                       declared coordinate-mismatch diagnostic for moved states;
    mode 'attached'  : PartRigidAsset, attached envelope (primary contract);
    mode 'fixed'     : PartRigidAsset, fixed envelope (declared sensitivity).
    """
    integrator.is_prepared = False
    integrator.asset_models = [base_model]
    integrator.prepare_scene(scene)
    if mode != "upstream":
        ids = part_shape_ids(scene, translations)
        integrator.asset_models = [PartRigidAsset(base_model, ids, translations, canonical_bbox, mode)]
    return integrator
