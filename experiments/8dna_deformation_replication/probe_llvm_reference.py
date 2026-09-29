"""Runtime diagnostic: CPU LLVM reference means for the teaset (run in WSL:
/opt/venvs/8dna26/bin/python probe_llvm_reference.py).  Independent of the GPU
JIT; see probe_loop_record.py."""
import sys, json, os, math
import numpy as np
import mitsuba as mi
mi.set_variant('llvm_ad_rgb')
mi.set_variant = lambda *a, **k: None          # upstream scene modules call set_variant('cuda_ad_rgb') on import
sys.path.insert(0, '/mnt/c/Projects/NeuralShaderDemo/external/8dna26')
from scenes.teaset import get_scene
lit = json.load(open('/mnt/c/Projects/NeuralShaderDemo/experiments/8dna_deformation_replication/protocol/teaset_cross_backbone_locked.json'))['lighting']
def scene(regime):
    d = get_scene(128)
    if regime == 'sun':
        del d['background']
        tl = np.asarray(lit['to_light'], float); tl /= np.linalg.norm(tl)
        d['sun'] = {'type': 'directional', 'direction': [float(v) for v in -tl], 'irradiance': {'type': 'rgb', 'value': 5.0}}
    d['integrator'] = {'type': 'path', 'max_depth': -1, 'rr_depth': 5}
    return mi.load_dict(d)
for regime in ('sun', 'env'):
    s = scene(regime)
    m = [float(np.array(mi.render(s, spp=1024, seed=60 + k)).mean()) for k in range(4)]
    print(f'LLVM {regime} path: {np.mean(m):.5f} +- {np.std(m)/2:.5f}  {np.round(m,5)}', flush=True)
