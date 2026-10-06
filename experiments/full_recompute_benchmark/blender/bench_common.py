"""Blender-side helpers shared by the benchmark scripts (runs inside Blender)."""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import hashlib
import json
import os
import platform
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
if EXP not in sys.path:
    sys.path.insert(0, EXP)

import frb_records as R  # noqa: E402


def script_args() -> list:
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def load_protocol(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def enable_optix() -> list:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.refresh_devices()
    used = []
    for d in prefs.devices:
        d.use = d.type == "OPTIX"
        if d.use:
            used.append(d.name)
    if not used:
        raise RuntimeError("no OptiX device available")
    return used


def apply_render_contract(scene, protocol: dict, spp: int, bounce_preset: str, persistent: bool) -> dict:
    """Apply the protocol's controlled settings; everything else stays as loaded."""
    rc = protocol["render_contract"]
    c = scene.cycles
    scene.render.engine = "CYCLES"
    c.device = "GPU"
    scene.render.resolution_x, scene.render.resolution_y = rc["resolution"]
    scene.render.resolution_percentage = 100
    c.samples = int(spp)
    c.use_adaptive_sampling = rc["adaptive_sampling"]
    c.use_denoising = rc["denoising"]
    c.seed = rc["seed"]
    c.use_animated_seed = False
    scene.render.use_persistent_data = bool(persistent)
    scene.render.use_compositing = rc["compositing"]
    scene.render.use_sequencer = rc["sequencer"]
    preset = protocol["bounce_presets"][bounce_preset]
    if preset != "as_loaded":
        for k, v in preset.items():
            setattr(c, k, v)
    return effective_settings(scene)


def effective_settings(scene) -> dict:
    c = scene.cycles
    keys = ["samples", "use_adaptive_sampling", "use_denoising", "seed", "max_bounces",
            "diffuse_bounces", "glossy_bounces", "transmission_bounces", "volume_bounces",
            "transparent_max_bounces", "caustics_reflective", "caustics_refractive",
            "blur_glossy", "sample_clamp_direct", "sample_clamp_indirect", "use_light_tree",
            "pixel_filter_type", "filter_width", "use_auto_tile", "tile_size", "device"]
    out = {k: getattr(c, k) for k in keys if hasattr(c, k)}
    out.update({
        "resolution": [scene.render.resolution_x, scene.render.resolution_y,
                       scene.render.resolution_percentage],
        "use_persistent_data": scene.render.use_persistent_data,
        "use_compositing": scene.render.use_compositing,
        "use_sequencer": scene.render.use_sequencer,
        "camera": scene.camera.name,
        "camera_lens": scene.camera.data.lens,
        "camera_dof": [scene.camera.data.dof.use_dof, scene.camera.data.dof.focus_distance,
                       scene.camera.data.dof.aperture_fstop],
    })
    return out


class Mover:
    """Rigid G0/G1 switching of the protocol mover. G0 is read from the file."""

    def __init__(self, protocol: dict):
        mv = protocol["mover"]
        self.obj = bpy.data.objects[mv["object"]]
        self.g0_loc = tuple(self.obj.location)
        self.g0_rot = tuple(self.obj.rotation_euler)
        d = mv["G1_translation_m"]
        self.g1_loc = tuple(a + b for a, b in zip(self.g0_loc, d))
        self.state = "G0"

    def set(self, state: str) -> bool:
        loc = {"G0": self.g0_loc, "G1": self.g1_loc}[state]
        changed = state != self.state
        self.obj.location = loc
        self.obj.rotation_euler = self.g0_rot
        self.state = state
        return changed

    def describe(self) -> dict:
        return {"object": self.obj.name, "G0_location": list(self.g0_loc),
                "G1_location": list(self.g1_loc), "rotation_euler": list(self.g0_rot)}


def scene_accounting(scene, scene_path: str | None) -> dict:
    """Counts from the evaluated depsgraph; logical = summed over instances."""
    dg = bpy.context.evaluated_depsgraph_get()
    n_inst = 0
    n_inst_mesh = 0
    logical_tris = 0
    unique: dict = {}
    mats = set()
    obj_types: dict = {}
    for oi in dg.object_instances:
        n_inst += 1
        o = oi.object
        obj_types[o.type] = obj_types.get(o.type, 0) + 1
        if o.type != "MESH":
            continue
        n_inst_mesh += 1
        key = o.data.as_pointer()
        if key not in unique:
            m = o.data
            m.calc_loop_triangles()
            unique[key] = len(m.loop_triangles)
            for s in m.materials:
                if s:
                    mats.add(s.name)
        logical_tris += unique[key]
    images = [i for i in bpy.data.images if i.source in ("FILE", "SEQUENCE", "TILED") and i.users]
    lights = [o for o in scene.objects if o.type == "LIGHT"]
    emit_objs = [o for o in scene.objects if o.type == "MESH" and any(
        s.material and s.material.name == "Emit" for s in o.material_slots)]
    return {
        "scene_objects": len(scene.objects),
        "scene_objects_by_type": _count_types(scene.objects),
        "depsgraph_instances": n_inst,
        "depsgraph_instances_by_type": obj_types,
        "mesh_instances": n_inst_mesh,
        "unique_evaluated_meshes": len(unique),
        "logical_triangles": logical_tris,
        "unique_triangles": sum(unique.values()),
        "materials_used": len(mats),
        "materials_in_file": len(bpy.data.materials),
        "images_used": len(images),
        "image_pixels": sum(i.size[0] * i.size[1] for i in images),
        "light_objects": len(lights),
        "emit_panel_objects": len(emit_objs),
        "file_size_bytes": os.path.getsize(scene_path) if scene_path else None,
    }


def _count_types(objs) -> dict:
    out: dict = {}
    for o in objs:
        out[o.type] = out.get(o.type, 0) + 1
    return out


class _PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.wintypes.DWORD), ("PageFaultCount", ctypes.wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def process_memory_mb() -> dict:
    """Working set / peak working set / private bytes of this Blender process (Windows)."""
    if platform.system() != "Windows":
        return {}
    pmc = _PMC()
    pmc.cb = ctypes.sizeof(pmc)
    k32 = ctypes.WinDLL("kernel32")
    psapi = ctypes.WinDLL("psapi")
    k32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.wintypes.HANDLE, ctypes.POINTER(_PMC), ctypes.wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        return {}
    mb = 1024.0 * 1024.0
    return {"working_set_mb": pmc.WorkingSetSize / mb, "peak_working_set_mb": pmc.PeakWorkingSetSize / mb,
            "private_mb": pmc.PagefileUsage / mb, "peak_private_mb": pmc.PeakPagefileUsage / mb}


def environment() -> dict:
    return {
        "blender_version": bpy.app.version_string,
        "blender_build_hash": bpy.app.build_hash.decode() if isinstance(bpy.app.build_hash, bytes) else bpy.app.build_hash,
        "blender_build_date": bpy.app.build_date.decode() if isinstance(bpy.app.build_date, bytes) else bpy.app.build_date,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
