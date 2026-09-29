"""Runtime diagnostic: DrJit LoopRecord vs wavefront reference renders on this GPU.

Six-way table (regime x LoopRecord x lanes per kernel) of teaset image means
with the C++ path integrator, 4 seeds each.  Compare with probe_llvm_reference.py
(CPU LLVM backend, WSL), which is independent of the GPU JIT."""

import json
import numpy as np, ednalib as L, teaset_parts as T
mi, dr = L.init_upstream()
proto = json.load(open(L.EXPERIMENT/'protocol/teaset_cross_backbone_locked.json'))
for regime, light in (('sun', proto['lighting']), ('env', None)):
    scene = mi.load_dict(T.scene_dict(128, {}, light))
    integ = mi.load_dict({'type': 'path', 'max_depth': -1, 'rr_depth': 5})
    for rec in (False, True):
        for chunk in (64, 256, 1024):
            dr.set_flag(dr.JitFlag.LoopRecord, rec)
            m = []
            for k in range(4):
                img = sum(np.array(mi.render(scene, integrator=integ, spp=chunk, seed=1000*k + c)) for c in range(1024 // chunk)) / (1024 // chunk)
                m.append(float(img.mean()))
            print(f'GPU {regime} path LoopRecord={rec} chunk={chunk} lanes=2^{int(np.log2(128*128*chunk))}: {np.mean(m):.5f} +- {np.std(m)/2:.5f}', flush=True)
    dr.set_flag(dr.JitFlag.LoopRecord, False)
