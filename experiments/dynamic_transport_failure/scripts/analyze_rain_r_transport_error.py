"""Compare frozen-RNA error in a direct-visibility transition against controls."""
from __future__ import annotations
import argparse, json, pathlib
import numpy as np
import pyexr
from PIL import Image
from scipy.spatial import cKDTree

def load(path: pathlib.Path) -> np.ndarray:
    return pyexr.open(str(path)).get().astype(np.float32)

def codes(image: np.ndarray) -> np.ndarray:
    rgb=image[..., :3]; out=np.full(rgb.shape[:2],-1,np.int8); good=rgb.max(-1)>0.1
    out[good & (rgb[...,0]>=rgb[...,1])]=0; out[good & (rgb[...,1]>rgb[...,0])]=1
    return out

def stats(value: np.ndarray) -> dict:
    return {"pixels":int(len(value)),"mae_mean":float(value.mean()),"mae_median":float(np.median(value)),"mae_p95":float(np.quantile(value,.95))}

def main() -> None:
    p=argparse.ArgumentParser()
    for prefix in ("r0","r4"):
        p.add_argument(f"--{prefix}-reference",type=pathlib.Path,required=True)
        p.add_argument(f"--{prefix}-rna",type=pathlib.Path,required=True)
        p.add_argument(f"--{prefix}-canonical",type=pathlib.Path,required=True)
        p.add_argument(f"--{prefix}-provenance",type=pathlib.Path,required=True)
        p.add_argument(f"--{prefix}-direct",type=pathlib.Path,required=True)
    p.add_argument("--output-dir",type=pathlib.Path,required=True); p.add_argument("--match-radius",type=float,default=.005)
    a=p.parse_args()
    def frame(k):
        ref,rna,can,prov,direct=(load(getattr(a,f"{k}_{x}")) for x in ("reference","rna","canonical","provenance","direct"))
        return ref,rna,can[...,:3],codes(prov),direct[...,:3].sum(-1)>0
    r0, n0, c0, s0, v0=frame("r0"); r4,n4,c4,s4,v4=frame("r4")
    valid0=(r0[...,3]>0)|(n0[...,3]>0); flat=np.flatnonzero(valid0.reshape(-1)&(s0.reshape(-1)>=0)); pts=c0.reshape(-1,3)[flat]; surface=s0.reshape(-1)[flat]; vis0=v0.reshape(-1)[flat]
    near=np.full(len(flat),np.inf,np.float32); indices=np.full(len(flat),-1,np.int64)
    for code in (0,1):
        mask=surface==code; candidates=np.flatnonzero((s4.reshape(-1)==code))
        d,local=cKDTree(c4.reshape(-1,3)[candidates]).query(pts[mask]); near[mask]=d; indices[mask]=candidates[local]
    matched=near<=a.match_radius; vis4=np.zeros(len(flat),bool); vis4[matched]=v4.reshape(-1)[indices[matched]]
    transition=matched & (surface==0) & vis0 & ~vis4
    control=matched & (surface==0) & vis0 & vis4
    if not transition.any() or control.sum()<transition.sum(): raise ValueError("insufficient matched scarf transition/control pixels")
    control_idx=np.flatnonzero(control)[np.linspace(0,control.sum()-1,transition.sum(),dtype=np.int64)]
    err4=np.abs(r4[...,:3]-n4[...,:3]).mean(-1).reshape(-1); affected_idx=indices[transition]
    visual=np.zeros((*s4.shape,3),np.uint8); vf=visual.reshape(-1,3); vf[affected_idx]=(255,220,0); vf[indices[control_idx]]=(0,220,0)
    out=a.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True); Image.fromarray(visual).save(out/"R4_scarf_transport_transition_vs_control.png")
    payload={"label":"RAIN REDESIGN1 R0-TO-R4 TRANSPORT ERROR ATTRIBUTION","regime":"frozen RNA; GT direct AOV; no refit","correspondence":"R0 canonical nearest-neighbour into R4 constrained by stable scarf provenance AOV; not triangle/UV tracking","match_radius":a.match_radius,"transition":"matched scarf R0 visible -> R4 occluded","control":"equal-count deterministic matched scarf R0 visible -> R4 visible","transition_error":stats(err4[affected_idx]),"control_error":stats(err4[indices[control_idx]]),"transition_to_control_mae_ratio":float(err4[affected_idx].mean()/err4[indices[control_idx]].mean()),"matched_scarf_pixels":int((matched&(surface==0)).sum()),"transition_pixels":int(transition.sum())}
    (out/"R4_scarf_transport_transition_error.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
if __name__=="__main__": main()
