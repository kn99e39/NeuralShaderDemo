"""Worklog-35: reproduced training curves and wall-clock per segment (RNA venv)."""
import json
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

W = "/mnt/c/Projects/NeuralShaderDemo/results/rna_reproduction/work/checkpoints"
runs = {"reproduced HQ": [f"{W}/tog-lego/version_1", f"{W}/tog-lego/version_2"], "reproduced small": [f"{W}/tog-lego_rt/version_0"]}
out, timing = {}, {}
for name, dirs in runs.items():
    vp, segs = [], []
    for d in dirs:
        a = EventAccumulator(d, size_guidance={"scalars": 0}); a.Reload()
        v = a.Scalars("val_psnr"); l = a.Scalars("loss")
        vp += [[x.step, x.value] for x in v]
        segs.append({"dir": d.split("/")[-2] + "/" + d.split("/")[-1], "epochs_validated": len(v), "first_step": l[0].step, "last_step": v[-1].step,
                     "wall_s": v[-1].wall_time - l[0].wall_time})
    # segment 1 of HQ ends after epoch 64; the resumed segment repeats nothing (it starts at step 26000)
    out[name] = {"val_psnr": vp}
    timing[name] = {"segments": segs, "total_wall_h": sum(s["wall_s"] for s in segs) / 3600,
                    "best_val_psnr": max(x[1] for x in vp), "final_val_psnr": vp[-1][1]}
json.dump(out, open(sys.argv[1], "w"))
json.dump(timing, open(sys.argv[2], "w"), indent=1)
print(json.dumps(timing, indent=1))
