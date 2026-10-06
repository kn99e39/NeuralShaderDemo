"""Focused Blender-side tests (run inside Blender 5.1).

blender --factory-startup -b --log cycles --log-level debug --log-file <log>
        --python tests/test_blender_side.py -- <protocol.json> <log>

1. accounting  : a synthetic scene with known counts - one cube (12 tris), a
                 collection holding a 2-subdivision icosphere (80 tris)
                 instanced three times, and the instanced collection's own
                 copy - gives logical/unique triangles, instance and unique
                 mesh counts exactly.
2. switching   : on BMW27, G0 -> G1 -> G0 restores every depsgraph instance
                 matrix bit-for-bit; in G1 exactly the mover's instances move,
                 each by the protocol translation, and nothing else changes.
3. contract    : apply_render_contract sets the protocol values and leaves the
                 as-loaded bounce settings alone for preset 'original'.
4. live parse  : a 1-spp render's own log slice parses, with
                 samples_rendered == 1 and non-negative phases.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "blender"))
import bench_common as B  # noqa: E402
import bpy  # noqa: E402

R = B.R


def test_accounting():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    bpy.ops.mesh.primitive_cube_add()
    coll = bpy.data.collections.new("Src")
    scene.collection.children.link(coll)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2)
    ico = bpy.context.active_object
    for c in ico.users_collection:
        c.objects.unlink(ico)
    coll.objects.link(ico)
    for k in range(3):
        e = bpy.data.objects.new(f"inst{k}", None)
        e.instance_type = "COLLECTION"
        e.instance_collection = coll
        e.location = (3 * (k + 1), 0, 0)
        scene.collection.objects.link(e)
    bpy.context.view_layer.update()
    a = B.scene_accounting(scene, None)
    assert a["logical_triangles"] == 12 + 4 * 80, a
    assert a["unique_triangles"] == 12 + 80, a
    assert a["mesh_instances"] == 5, a
    assert a["unique_evaluated_meshes"] == 2, a
    assert a["scene_objects"] == 5, a  # cube, icosphere (via child collection), 3 empties
    print("PASS accounting", a["logical_triangles"], a["unique_triangles"])


def _matrices():
    dg = bpy.context.evaluated_depsgraph_get()
    out = []
    for oi in dg.object_instances:
        par = oi.parent.original.name if oi.is_instance and oi.parent else None
        out.append((oi.object.original.name, par, tuple(tuple(r) for r in oi.matrix_world)))
    return out


def test_switching(protocol):
    bpy.ops.wm.open_mainfile(filepath=protocol["scene_source"]["path"], load_ui=False)
    mover = B.Mover(protocol)
    name = protocol["mover"]["object"]
    d = protocol["mover"]["G1_translation_m"]
    m0 = _matrices()
    mover.set("G1"); bpy.context.view_layer.update()
    m1 = _matrices()
    mover.set("G0"); bpy.context.view_layer.update()
    m0b = _matrices()
    assert m0 == m0b, "G0 -> G1 -> G0 did not restore every instance matrix bit-for-bit"
    assert len(m0) == len(m1)
    moved = 0
    for (n, p, a), (n1, p1, b) in zip(m0, m1):
        assert (n, p) == (n1, p1)
        if p == name or n == name:
            moved += 1
            for k in range(3):
                assert abs((b[k][3] - a[k][3]) - d[k]) < 1e-5, (n, a, b)
                assert all(abs(b[k][c] - a[k][c]) < 1e-6 for c in range(3))
        else:
            assert a == b, f"non-mover instance {n} (parent {p}) changed"
    assert moved > 10, moved
    print("PASS switching", moved, "mover instances moved;", len(m0) - moved, "unchanged")


def test_contract(protocol):
    scene = bpy.context.scene
    before = {k: getattr(scene.cycles, k) for k in protocol["bounce_as_loaded"]}
    assert before == protocol["bounce_as_loaded"], before
    s = B.apply_render_contract(scene, protocol, 64, "original", True)
    assert s["samples"] == 64 and s["resolution"] == [1920, 1080, 100]
    assert s["use_adaptive_sampling"] is False and s["use_denoising"] is False
    assert s["use_persistent_data"] is True and s["use_compositing"] is False
    assert {k: s[k] for k in before} == before
    s = B.apply_render_contract(scene, protocol, 64, "low", False)
    assert s["max_bounces"] == 2 and s["glossy_bounces"] == 2 and s["transparent_max_bounces"] == 128
    print("PASS contract")


def test_live_parse(protocol, logpath):
    bpy.ops.wm.open_mainfile(filepath=protocol["scene_source"]["path"], load_ui=False)
    B.enable_optix()
    B.apply_render_contract(bpy.context.scene, protocol, 1, "original", False)
    pos = os.path.getsize(logpath)
    bpy.ops.render.render(write_still=False)
    import time
    for _ in range(100):
        try:
            with open(logpath, "rb") as f:
                f.seek(pos)
                p = R.parse_cycles_render_log(f.read().decode("utf-8", "replace"))
            break
        except ValueError:
            time.sleep(0.02)
    assert p["samples_rendered"] == 1, p
    assert all(p[k] >= 0 for k in ("sync_s", "path_trace_s", "cycles_total_s")), p
    print("PASS live_parse", {k: round(v, 4) for k, v in p.items() if isinstance(v, float)})


if __name__ == "__main__":
    try:
        protocol_path, logpath = B.script_args()[:2]
        protocol = B.load_protocol(protocol_path)
        test_accounting()
        test_switching(protocol)
        test_contract(protocol)
        test_live_parse(protocol, logpath)
        print("ALL_PASS", flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)
