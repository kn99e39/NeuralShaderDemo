"""Frozen released-8DNA evaluation of the locked teaset trajectory.

Refuses to run unless the locked protocol's hashes match and the project
tree is clean, so the reported commit is the code that produced the result.

Per locked state it renders the path-traced reference (locked seeds A/B, the
same as the GT-only design run) and frozen 8DNA in these query modes:
  attached  primary correspondence contract (all states)
  fixed     declared envelope sensitivity (all states)
  upstream  released code path; at T0 the canonical baseline, at moved
            states the declared coordinate-mismatch diagnostic (current-world
            queries into the canonical tri-plane -- not a transport result)
and reports, separately, the physical change |GT(Ti)-GT(T0)|, the model change
|8DNA(Ti)-8DNA(T0)| and the model error |GT(Ti)-8DNA(Ti)| per ROI.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import ednalib as L
import teaset_parts as T


def response(dn: np.ndarray, dg: np.ndarray) -> dict:
    """How much of the physical change the model reproduces (linear RGB, masked pixels)."""
    dn, dg = dn.reshape(-1), dg.reshape(-1)
    gg = float(dg @ dg)
    return {
        "gain_projection": float(dn @ dg / gg) if gg > 0 else None,
        "correlation": float(np.corrcoef(dn, dg)[0, 1]) if dn.std() > 0 and dg.std() > 0 else None,
        "residual_rms": float(np.sqrt(np.mean((dn - dg) ** 2))),
        "physical_change_rms": float(np.sqrt(np.mean(dg ** 2))),
        "model_change_rms": float(np.sqrt(np.mean(dn ** 2))),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--allow-dirty", action="store_true", help="development only; result is not evidence")
    args = ap.parse_args()
    proto = json.loads(open(args.protocol, encoding="utf-8").read())
    mi, dr = L.init_upstream()
    env = L.environment_record()
    if env["project_dirty"] and not args.allow_dirty:
        raise SystemExit(f"project tree is dirty; commit before the evidence run: {env['project_dirty'][:5]}")
    ck = L.sha256(L.checkpoint_path(proto["asset"]))
    if ck != proto["checkpoint_sha256"]:
        raise SystemExit("checkpoint hash differs from the locked protocol")
    out = L.RESULTS / proto["frozen_output"]
    gt_dir = L.RESULTS / proto["gt_output"]
    res, states, nr = proto["res"], proto["states"], proto["neural"]
    roi_names = ("interaction", "tray", "far", "mover")
    rois = {}
    for s_name in states:
        z = np.load(gt_dir / f"rois_{s_name}.npz")
        rois[s_name] = {k: (z[f"mask_{k}"].reshape(-1), z[f"idx0_{k}"]) for k in roi_names}

    scene0 = mi.load_dict(T.scene_dict(res, states["T0"]))
    bbox0 = scene0.shapes()[0].bbox()
    integ = L.load_neural_integrator(proto["asset"])
    base = integ.asset_models[0]

    gt, neural, times = {}, {}, {}
    for name, tr in states.items():
        scene = mi.load_dict(T.scene_dict(res, tr))
        a, _ = L.render_reference(scene, proto["gt"]["spp"], proto["gt"]["chunk"], proto["gt"]["seed_A"])
        b, _ = L.render_reference(scene, proto["gt"]["spp"], proto["gt"]["chunk"], proto["gt"]["seed_B"])
        design_a = L.load_exr(gt_dir / name / "gt_A.exr")
        gt[name] = (a, b, float(np.abs(a - design_a).max()))
        L.save_exr(out / name / "gt_A.exr", a)
        L.save_exr(out / name / "gt_B.exr", b)
        for mode in proto["modes"][name]:
            T.configure(integ, base, scene, mode, tr, bbox0)
            img, t = L.render_chunked(scene, integ, nr["spp"], nr["chunk"], nr["seed"])
            neural[(name, mode)], times[(name, mode)] = img, t
            L.save_exr(out / name / f"8dna_{mode}.exr", img)
            print(name, mode, f"{t:.0f}s", flush=True)
    T.configure(integ, base, scene0, "upstream")
    neural_b, _ = L.render_chunked(scene0, integ, nr["spp"], nr["chunk"], nr["seed_repeat"])
    L.save_exr(out / "T0" / "8dna_upstream_seedB.exr", neural_b)

    rec = {"protocol": proto, "environment": env, "checkpoint_sha256": ck,
           "gt_rerender_max_abs_diff_vs_design": {s: gt[s][2] for s in states},
           "neural_seconds": {f"{s}/{m}": t for (s, m), t in times.items()},
           "noise": {}, "states": {}}
    flat = lambda img: img.reshape(-1, 3)
    g0 = flat(0.5 * (gt["T0"][0] + gt["T0"][1]))
    for k in roi_names:
        m0 = rois["T0"][k][0]
        rec["noise"][k] = {"gt_repeat_A_vs_B": L.pixel_metrics(flat(gt["T0"][0])[m0], flat(gt["T0"][1])[m0]),
                           "neural_seed_repeat_T0": L.pixel_metrics(flat(neural_b)[m0], flat(neural[("T0", "upstream")])[m0])}
    rec["noise"]["full_image"] = {"gt_repeat_A_vs_B": L.metrics(gt["T0"][0], gt["T0"][1]),
                                  "neural_seed_repeat_T0": L.metrics(neural_b, neural[("T0", "upstream")])}
    for name in states:
        g_img = 0.5 * (gt[name][0] + gt[name][1])
        g = flat(g_img)
        st = {"translations": states[name], "role": proto["roles"][name], "modes": {}}
        for mode in proto["modes"][name]:
            n = flat(neural[(name, mode)])
            n0 = flat(neural[("T0", mode)])
            reg = {"full_image": {"model_error_8DNA_vs_GT": L.metrics(neural[(name, mode)], g_img)}}
            for k in roi_names:
                m, i0 = rois[name][k]
                reg[k] = {
                    # every change is paired: state pixel vs its corresponding T0 pixel
                    "physical_change_GT_vs_GT0": L.pixel_metrics(g[m], g0[i0]),
                    "model_change_8DNA_vs_8DNA0": L.pixel_metrics(n[m], n0[i0]),
                    "model_error_8DNA_vs_GT": L.pixel_metrics(n[m], g[m]),
                    "model_error_at_T0_same_surface": L.pixel_metrics(n0[i0], g0[i0]),
                    "response_linear": response(n[m] - n0[i0], g[m] - g0[i0]),
                }
            st["modes"][mode] = reg
        rec["states"][name] = st
    L.write_json(out / "frozen_eval.json", rec)
    print("written", L.rel(out / "frozen_eval.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
