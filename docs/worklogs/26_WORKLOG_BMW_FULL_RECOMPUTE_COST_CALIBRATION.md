# Worklog — Physical Full-Recomputation Cost Calibration on BMW27 and BMW Garage XL (2026-10-06)

## Session question

`docs/Architecture.md` (added remotely in `5cb61ed`) argues that a future
method should reuse persistent appearance state and update only
configuration-dependent GI state, and that this only matters if the update is
much cheaper than rebuilding the current transport (§10, §15.1). Before any
method exists, how expensive is it to get *physically current* GI again after
a meaningful rigid geometry change, in a real Cycles scene, on the RTX 5080?
How does that cost grow with samples per pixel, bounce depth, and scene scale?

This is a calibration batch. No neural method was implemented, trained or
timed. Nothing here validates `Architecture.md`. RNA/8DNA code, the teaset
experiment and earlier worklogs were not touched.

## Repository state

- Local `main` held Worklog 25 (`aa7e374`, committed, clean tree) two
  commits ahead of `origin/main`; remote `main` held `5cb61ed`
  (`docs/Architecture.md`). Merged without conflict: `4b0db56`.
- Benchmark implementation, protocol and tests committed before any
  evidence run: **`3060f1d`**. All evidence runs (run id `v1_3060f1d`) were
  produced by the measurement code of that commit.
- `93e04af` added only orchestration (`--resume` for the driver), the report
  script and an untimed preview renderer; measurement code unchanged. It was
  committed before the one resumed condition (see "Interrupted condition").
- Nothing was pushed.

## IMPLEMENTATION FACT

### Scene source

| item | value |
|---|---|
| source | official Blender demo/benchmark file, `https://download.blender.org/demo/test/BMW27_2.blend.zip` (BMW27, Mike Pan) |
| archive sha256 | `74f5dc6d718fc565e0ff50e355be7f8b1a58983cc9ae79775608d8905c269ee4` |
| file used | `bmw27/bmw27_gpu.blend`, sha256 `172a6b1c6f25acf59d17cb2682866a6b7a33718b92ad15f4073d55d16ff3af74` (`bmw27_cpu.blend`, `4f0a12ee…`, differs only in tiling and is unused) |
| stored at | `external/bmw27_official/` (gitignored), never saved over |
| file version | 2.77; Blender 5.1.0 converts it on load (one harmless `blend.doversion` warning about a UI region). The converted state is never written back. |

Scene as loaded: two BMW 1M cars (the front car is real objects; the rear car
is an empty `1M` instancing the same collection), a cyclorama floor with an
anisotropic + sheen material, one emission panel (`Light`, strength 5,
camera-invisible), a blue-grey world (strength 1), headlight emitters, camera
`Camera` (75 mm, f/0.75 depth of field focused at 12.03 m). Cycles settings as
loaded: max 32 / diffuse 16 / glossy 16 / transmission 32 / volume 32 bounces,
transparent 128, caustics on, no clamping, light tree off, Gaussian 2.0 px
filter.

### Environment

| item | value |
|---|---|
| OS | Windows 11 Pro 10.0.26200 |
| CPU | AMD Ryzen 9 9950X3D (16 C / 32 T) |
| RAM | 63.6 GB |
| GPU | NVIDIA GeForce RTX 5080, 16,303 MiB, driver 596.49 |
| Blender | 5.1.0, build hash `adfe2921d5f3` (2026-03-17), `--factory-startup -b` |
| Cycles device | GPU, OptiX (precompiled `kernel_sm_120` cubin, OptiX pipeline from the warm disk cache) |

All numbers below are from this one Blender build. They are not comparable
to other builds without a label.

### Protocol and code

- Protocol `experiments/full_recompute_benchmark/protocol/bmw_full_recompute_v1.json`,
  written before any timing at protocol settings. It records two
  pre-protocol observations honestly: a 16-spp smoke timing and the scene's
  triangle/memory accounting were seen first, and G1 was chosen from a 32-spp
  half-resolution preview plus a top view (collision and visibility check).
  The protocol was not changed after the evidence runs started.
- Render contract: 1920×1080, Cycles GPU/OptiX, denoising off, adaptive
  sampling off, seed 0, everything else as loaded. **Deviation declared in
  the protocol:** the file's compositor (glare, blur, bokeh-image overlay,
  colour balance) is switched off in timed runs because it is image
  post-processing, not transport. Its cost was measured once (below).
- Mover: object `1M` (the rear car). G0 = as loaded; G1 = G0 + (−1.0, −2.0,
  0) m, rotation unchanged. It brings the rear car ~2 m from the front car and
  partly under the emission panel.
- One Blender process per condition: open the file (A), cold render in G0,
  2 warm-up frames, 10 steady frames (5 with persistent data off) alternating
  G1/G0 so every steady frame follows a mover change, then 3 static re-renders
  with no change (control).
- Phase timing per render comes from that render's slice of the Cycles debug
  log (`--log cycles --log-level debug`), which flushes per render. Python
  timers measure C and the bpy render call.
- Two update paths, both "physical full recomputation" of GI (every frame
  path-traces from scratch; nothing is cached across frames except scene
  data):
  - **persistent data on** (`render.use_persistent_data`): the Cycles session
    survives, and only changed data is re-synchronised. This is Blender's
    incremental final-render path. Primary.
  - **persistent data off**: a fresh session per frame, with full export and
    full BVH build.
- Code: `frb_records.py` (schema, log parser, statistics, FPS/gap),
  `blender/bench_common.py`, `blender/bench_timing.py`,
  `blender/build_xl.py`, `blender/render_evidence.py`,
  `blender/render_preview.py`, `run_benchmark.py` (host driver, nvidia-smi
  poller), `make_report.py`.

### What Blender exposes, and what it does not

| phase | source | separable? |
|---|---|---|
| A file load | `wm.open_mainfile` wall time | yes |
| B cold setup | cold render: Cycles total − path tracing | yes (aggregate) |
| C transform + depsgraph | Python timer around `view_layer.update()` | yes |
| D Blender→Cycles sync | Cycles "Total time spent synchronizing data" | yes |
| E Cycles device update | Cycles total − render-without-sync − D | **only as an aggregate**: BVH/IAS build or refit together with every other device manager (lights, shaders, images, attributes). The debug log does not split them. Trace level adds per-kernel spam that would perturb timing, and gives no manager split either. The log does report how many acceleration-structure builds a render performed ("Tasks handled"), which is used below. |
| F path tracing | Cycles render-scheduler "Path Tracing" wall time | yes |
| G frame total | C + bpy render-call wall time (includes Blender's render-pipeline overhead outside the Cycles session, reported separately) | yes |

GPU memory: per-process usage is not available under WDDM, so whole-GPU
`memory.used` peak minus the pre-launch baseline is reported (it includes
the CUDA/OptiX context). Alongside it, the per-render maximum of Cycles'
own `Mem:` stats figure is reported (Cycles-tracked device allocations).

### BMW Garage XL

The protocol's predeclared trigger ("lightweight" = under 5 M logical
triangles **and** Cycles scene memory under 10% of 16 GB, i.e. workload
properties, not timing) was met: 0.70 M logical / 0.23 M unique triangles,
about 77 MB of geometry plus textures. So the XL series was built. It is
saved separately under `results/full_recompute_benchmark/scenes/`.

- Unit = one bay, i.e. the original layout (front car, rear car, one
  emission panel). Bay (0,0) is the untouched original scene and keeps the
  mover. Added bays sit on a 13 m pitch.
- Front and rear copies are collection instances of `1M`. Panel copies are
  linked duplicates of `Light`. The cyclorama floor is scaled in x/y to keep
  the original margin. No material, shader, light strength, world or render
  setting is changed.
- **instanced** series: 1×1, 2×2, 4×4, 8×8, 16×16 bays (2 to 512 cars).
  **realized** series (added instances converted with `duplicates_make_real`,
  each keeping its modifier stack, so evaluated geometry is no longer
  shared): 2×2, 4×4, 8×8, 12×12 (8 to 288 cars).
- Camera `XLCamera` (one rule for every XL level, including 1×1): a copy of
  the original camera data, original azimuth, 30° down, 35 mm, at the
  smallest distance that frames every car. Depth of field is kept at f/0.75,
  focused on the grid centre.
- Every car is camera-ray visible at every level (9 probe rays per car,
  skipping camera-invisible emission panels): 2/2, 8/8, 32/32, 128/128,
  288/288, 512/512.
- Build working set: about 0.3 GB for every instanced level; realized
  1.1 / 4.1 / 16.2 / 36.3 GB. The 48 GB stop rule was never hit, and no level
  failed to build or render.

| variant | cars | logical tris | unique tris | mesh instances | unique evaluated meshes | file | sha256 (12) |
|---|---|---|---|---|---|---|---|
| original BMW27 | 2 | 0.70 M | 0.23 M | 104 | 35 | 3.1 MB | `172a6b1c6f25` |
| xl 1×1 instanced | 2 | 0.70 M | 0.23 M | 104 | 35 | 5.4 MB | `8113acb95767` |
| xl 2×2 instanced | 8 | 2.8 M | 0.23 M | 413 | 35 | 5.5 MB | `32f4d710b221` |
| xl 4×4 instanced | 32 | 11.2 M | 0.23 M | 1,649 | 35 | 5.5 MB | `6080f1e8ad33` |
| xl 8×8 instanced | 128 | 44.8 M | 0.23 M | 6,593 | 35 | 5.8 MB | `25813b0a18d1` |
| xl 16×16 instanced | 512 | 179.1 M | 0.23 M | 26,369 | 35 | 6.9 MB | `865e3562df49` |
| xl 2×2 realized | 8 | 2.8 M | 1.85 M | 413 | 281 | 6.5 MB | `cde798b49afd` |
| xl 4×4 realized | 32 | 11.2 M | 8.32 M | 1,649 | 1,265 | 10.8 MB | `5b01fb3b0059` |
| xl 8×8 realized | 128 | 44.8 M | 34.2 M | 6,593 | 5,201 | 28.0 MB | `0daa4bb76ef3` |
| xl 12×12 realized | 288 | 100.8 M | 77.4 M | 14,833 | 11,761 | 56.7 MB | `4ba040cb53ef` |

All variants: 23 materials, 6 images (18.4 M texels, most of them the 4k
`sidewall.jpg`), one emission panel per bay.

### Tests

Focused only. Production neural code was untouched, so the full project
suite was not run.

- `tests/test_records.py` (stdlib): FPS/gap arithmetic; parsing of two real
  Blender 5.1 Cycles log slices (a cold render and a render after a mover
  change); rejection of zero- or two-render slices; derived fields; summary
  keeps the first value and reports the median; schema validation; lossless
  JSON round trip with NaN refused. **All pass.**
- `tests/test_blender_side.py` (inside Blender): accounting on a synthetic
  scene with known counts; deterministic switching on BMW27 (G0→G1→G0 restores
  all 129 instance matrices bit-for-bit, and in G1 exactly the 61 mover
  instances move, each by the protocol translation, while 68 others are
  bit-identical); render-contract application; a live 1-spp render's log
  slice parses. **All pass.** One expected value in the first draft of the
  accounting test was wrong (I assumed a 2-subdivision icosphere has 320
  faces; Blender's has 80). The test, not the accounting code, was corrected.

### Smoke-run fixes before the evidence runs (not evidence)

- Blender 5.1 selects multilayer EXR through `image_settings.media_type`.
- `scene.ray_cast` hits the camera-invisible emission panels, which made 2 of
  8 cars look occluded in the 2×2 visibility check. The probe now steps past
  camera-invisible objects.
- The Cycles stats string carries `Mem:` (no `Peak:`). The parser was
  switched to the per-render maximum of `Mem:`.
- The poller's stop event was renamed so it no longer shadows `Thread._stop`.

### Interrupted condition

The XL stage would have outlasted the driver's 2-hour background limit during
its last condition (`xl12_realized_spp064_poff`, about 38 min). That process
was stopped 10 s after it started, and its log was kept as
`*.interrupted_*.log`. After `93e04af` it was rerun alone with `--resume`,
which skips existing records. No other record was re-run or overwritten.

## MEASUREMENT

All frame numbers are the **median of the steady frames**, each rendered right
after a mover change. Min–max spreads are in the table export. Static
re-renders are the no-change control.

### 1. BMW27 SPP curve (original camera, original bounces)

Persistent data **on**:

| spp | frame ms (median) | min–max | C | D | E | F path trace | FPS | gap vs 30 FPS | gap vs 60 FPS |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 38.3 | 37.4–39.2 | 0.1 | 0.3 | 1.6 | 23.3 | 26.1 | 1.1× | 2.3× |
| 4 | 84.7 | 83.4–86.7 | 0.1 | 0.3 | 1.7 | 69.7 | 11.8 | 2.5× | 5.1× |
| 16 | 218.8 | 216.3–220.3 | 0.1 | 0.3 | 1.6 | 203.6 | 4.57 | 6.6× | 13.1× |
| 64 | 763.0 | 755.6–771.3 | 0.1 | 0.3 | 1.7 | 747.9 | 1.31 | 22.9× | 45.8× |
| 256 | 2977.4 | 2959.1–2995.0 | 0.1 | 0.3 | 1.8 | 2961.8 | 0.34 | 89.3× | 178.6× |

Persistent data **off** (full export and BVH build every frame):

| spp | frame ms | D sync | E device update | F | Blender overhead | FPS | gap 30 | gap 60 |
|---|---|---|---|---|---|---|---|---|
| 1 | 487.2 | 137.3 | 251.1 | 23.7 | 46.2 | 2.05 | 14.6× | 29.2× |
| 4 | 529.8 | 138.4 | 251.7 | 68.8 | 47.7 | 1.89 | 15.9× | 31.8× |
| 16 | 663.6 | 137.8 | 255.9 | 203.0 | 47.5 | 1.51 | 19.9× | 39.8× |
| 64 | 1199.2 | 137.6 | 246.0 | 746.6 | 48.1 | 0.83 | 36.0× | 72.0× |
| 256 | 3424.8 | 137.1 | 253.9 | 2959.8 | 49.5 | 0.29 | 102.7× | 205.5× |

- Path tracing costs about 11.5 ms per additional sample at 1080p, on top of
  a ~12 ms first-sample floor: (2961.8 − 23.3) / 255.
- A file load: 0.02 s.
- Cold first render: 0.48 s (1 spp) to 3.44 s (256 spp) with persistent data
  on. That is the same full sync plus BVH (≈0.39 s) as every persistent-off
  frame.
- With persistent data on, steady frames performed 0 acceleration-structure
  builds. With it off, every frame performed 36 (one per unique mesh).
- Static re-render (no change), persistent on: 37.7 / 83.4 / 220.2 / 764.6 /
  2991.9 ms. This is indistinguishable from the after-change frames.
- GPU: +1.8–1.9 GiB over baseline. The Cycles `Mem:` maximum is 824 MB, most
  of it render buffers and path state, plus a fixed 690 MB local-memory
  reservation. Process working set is about 1.0 GB.
- Repeatability: the 64-spp persistent-on configuration ran twice in
  separate processes (SPP stage 763.0 ms, bounce stage 755.2 ms), a 1.0%
  spread.

### 2. Bounce depth (64 spp, persistent on)

| preset | bounces (max/diff/gloss/trans) | frame ms | F |
|---|---|---|---|
| low | 2/2/2/2 | 736.5 | 720.8 |
| medium | 8/4/4/8 | 745.4 | 730.3 |
| original | 32/16/16/32 | 755.2 | 739.9 |

### 3. Compositor (64 spp, persistent on)

Frame time is 832.9 ms with the file's compositor on, against about 755–763 ms
off. Blender pipeline overhead is 76.9 ms (on) against 1.0 ms (off).

### 4. Scale series (XLCamera, after a mover change)

| variant | logical tris | 64 spp pon | 64 spp poff | 1 spp pon | 1 spp poff | 1 spp pon: C / D / E | GPU +MiB | peak working set |
|---|---|---|---|---|---|---|---|---|
| xl 1×1 inst | 0.70 M | 624 | 1,058 | 33.6 | 472 | 0.1 / 0.3 / 1.8 | 1,805 | 1.0 GB |
| xl 2×2 inst | 2.8 M | 640 | 1,074 | 36.5 | 473 | 0.1 / 0.7 / 3.3 | 1,816 | 1.0 GB |
| xl 4×4 inst | 11.2 M | 604 | 1,039 | 43.4 | 480 | 0.1 / 1.9 / 7.4 | 1,813 | 1.0 GB |
| xl 8×8 inst | 44.8 M | 500 | 948 | 62.2 | 513 | 0.2 / 6.6 / 24.6 | 1,835 | 1.0 GB |
| xl 16×16 inst | 179.1 M | 504 | 980 | 158.1 | 621 | 0.5 / 26.7 / 99.8 | 1,921 | 1.2 GB |
| xl 2×2 real | 2.8 M | 641 | 3,395 | 38.1 | 2,803 | 0.3 / 1.1 / 3.3 | 2,185 | 3.9 GB |
| xl 4×4 real | 11.2 M | 669 | 13,193 | 48.3 | 12,878 | 1.5 / 3.6 / 8.1 | 3,335 | 15.2 GB |
| xl 8×8 real | 44.8 M | 563 | 54,076 | 88.5 | 54,121 | 5.5 / 14.9 / 28.8 | 7,787 | 50.1 GB |
| xl 12×12 real | 100.8 M | 623 | 183,175 | 169.9 | 198,235 | 15.1 / 37.4 / 63.5 | 12,326 | 55.5 GB |

(Frame times in ms; "pon"/"poff" = persistent data on/off.)

- Persistent off, realized: almost all of the frame is D (Blender→Cycles
  export: 2.2 s, 10.7 s, 45.7 s, 129.3 s at 1 spp) plus E (0.5, 1.8, 7.0,
  63.9 s), with 306, 1,386, 5,706 and 12,906 acceleration-structure builds per
  frame.
- Persistent on: still 0 builds per steady frame at every level. The
  non-path-tracing part of a frame after the change (frame − F, 1 spp) grows
  from ~15 ms (2 cars) to ~142 ms (512 instanced cars) and ~148 ms (288
  realized cars).
- Change cost vs no change (persistent on, 1 spp, frame − static): +0.6 ms
  (original), +1 ms (xl 1×1), +24 ms (8×8 inst), +100 ms (16×16 inst),
  +29 ms (8×8 real), +60 ms (12×12 real).
- Cold first render with persistent on, which is the full build every
  incremental session pays once: 0.5 s (instanced, any level), 2.9 / 12.9 /
  54.7 / 150.5 s (realized 2×2 … 12×12).
- Path tracing at a fixed spp falls as the grid grows (64 spp:
  609 → 455 → 365 ms for instanced 1×1 → 8×8 → 16×16), because the XL camera
  pulls back to frame more cars.

### 5. G0/G1 transport evidence (original scene, 64 spp)

Renders: G0 seed 0, G1 seed 0, G0 seed 1, each Combined + Depth. A pixel is
*stationary* when its depth is equal in G0 and G1 (same seed, relative
1e-4): its first hit is unchanged.

| quantity | value |
|---|---|
| stationary pixels | 1,798,022 of 2,073,600 (86.7%); first hit changed on 275,578, i.e. the rear car's two positions spread by depth of field |
| mean \|ΔL\| G1−G0 on stationary pixels (per pixel) | 0.00123 (mean stationary luminance 0.162) |
| mean \|ΔL\| G0 seed 0 − seed 1 on stationary pixels | 0.0110 |
| 9×9 box-filtered, whole stationary mask: change / seed noise | 0.00119 / 0.00138 (0.86×) |
| stationary pixels whose box-filtered change exceeds 3× the box-filtered seed noise **at the same pixel** | 198,709 (11.1%) |
| within those: mean box-filtered change / noise | 0.0099 / 0.00034 (≈29×; ≈6% of mean stationary luminance) |

The per-pixel same-seed difference cancels most sampling noise, so comparing
it with independent-seed noise is conservative. Averaged over the whole
frame, the change is the size of the noise, because it is local. The change
map (`results/evaluation/26/08_*`) puts it on the floor around and left of
the rear car (its moved shadow and contact region), in the gap between the
cars, and on small parts of the front car near the rear car. Most of the
front car's body changes little. The move is real and changes transport on
surfaces that did not move; it is a local, not a global, change in this
scene.

## OBSERVATION

- With Blender's incremental path (persistent data), a rigid move of a whole
  car changes **nothing measurable** in the BMW27 frame cost. Geometry sync
  (D) is 0.3 ms, device update (E) 1.7 ms, and no BLAS is rebuilt. Path
  tracing is 61% of the frame at 1 spp, 93% at 16 spp, 98% at 64 spp and
  99.5% at 256 spp. The rest at 1 spp, about 15 ms, is mostly fixed
  per-render session work (render-buffer write-out and scheduling, about
  12 ms), not geometry. The cost of physically current GI here is almost
  entirely the cost of sampling, and it is linear in spp.
- Without persistent data, the same move costs a fixed ≈0.39 s of export plus
  BVH on BMW27, about 17× the 1-spp path tracing. This fixed cost grows
  roughly linearly with unique geometry (realized 8 → 128 cars: 2.8 s → 54 s)
  and super-linearly at 288 cars (198 s at 1 spp, 183 s at 64 spp: about 8%
  process-to-process spread in the rebuild itself). The 288-car process ran at
  55–56 GB working set on a 63.6 GB machine, so memory pressure may inflate
  that point. It is not separated here.
- Instancing makes even the full rebuild cheap: 512 instanced cars rebuild
  in ≈0.6 s per frame at 1 spp. The incremental path's per-change overhead
  instead grows with instance count (IAS rebuild and per-instance sync):
  ~100 ms at 26 k instances.
- Bounce depth barely matters in this open studio scene: 2 → 32 bounces adds
  2.5%. Most paths leave the scene early. This is a property of BMW27's
  layout, not of indirect light in general.
- The scale series does not show path-tracing cost rising with scene size.
  At fixed spp, path tracing *drops*, because the per-variant camera makes
  every car smaller on screen.

## INTERPRETATION

- On the RTX 5080, physically current GI after a meaningful rigid
  configuration change in BMW27 costs, per frame, at 1080p without
  denoising:
  - **38 ms (26 FPS) at 1 spp**, which is a noisy, unconverged image;
  - **0.22 s (4.6 FPS) at 16 spp**;
  - **0.76 s (1.3 FPS) at 64 spp**;
  - **3.0 s (0.34 FPS) at 256 spp**.

  These figures use Blender's most favourable exposed path, persistent data.
  Without it, each frame costs 0.39 s more. That is 1.1×–89× the 30-FPS
  budget and 2.3×–179× the 60-FPS budget across the measured spp range. One
  spp setting alone does not decide real-time feasibility; the curve does.
- In large non-instanced scenes, a fresh rebuild of scene data, as opposed to
  Blender's incremental session, costs tens to hundreds of seconds per frame
  (54 s at 128 cars, 198 s at 288 cars), even before sampling matters. With
  an incremental session the non-sampling overhead is 0.07–0.15 s per frame,
  plus sampling. So "full
  recomputation" spans three orders of magnitude depending on whether scene
  data persists. A future comparison must name which one it uses, as
  `Architecture.md` §10 already separates physical recomputation from neural
  re-encoding.
- What this implies for the reuse argument, as an order of magnitude and not
  as a method claim:
  - Geometry synchronisation is *not* the expensive part of physically current
    GI when scene data persists. The expensive part is drawing enough samples
    to converge transport again: about 11.5 ms per spp at 1080p here.
  - An incremental neural transport update is therefore practically
    meaningful only if update plus inference fits a frame budget,
    **~1–10 ms for the update** within a 16.7–33 ms frame. That is 1–2
    orders of magnitude below the 0.2–3 s of 16–256 spp physical
    recomputation. It also needs quality clearly above what 1–4 spp physical
    frames give (38–85 ms per frame).
  - An update costing ~0.1–1 s would compete with roughly 8–85 spp of physical
    rendering on this hardware, and the reuse argument would be weak.
  - Against a non-persistent rebuild of a large non-instanced scene
    (seconds to minutes), almost any local update wins. That is the weaker,
    less interesting baseline.
- **Batch question 1, measured cost and FPS of physically current GI after the
  change (RTX 5080, Blender 5.1.0, 1080p, no denoising):**
  - BMW27: 38 ms / 85 ms / 219 ms / 763 ms / 2,977 ms at 1 / 4 / 16 / 64 /
    256 spp, i.e. 26.1 / 11.8 / 4.6 / 1.3 / 0.34 FPS, with persistent data.
    Without it, add ≈0.39 s per frame (2.05 … 0.29 FPS).
  - BMW Garage XL, 64 spp with persistent data: 0.50–0.67 s per frame up to
    179 M logical / 77 M unique triangles. The incremental update overhead
    grows to ~0.15 s.
  - BMW Garage XL, without persistent data: about 1 s per frame (instanced),
    and 3.4 s / 13 s / 54 s / 183–198 s for realized 8 / 32 / 128 / 288 cars.
- **Batch question 2, order of magnitude a future incremental update must
  reach:** ~1–10 ms per change, inside a 16.7–33 ms frame together with
  inference. That is ≥1–2 orders of magnitude below 16–256 spp physical
  recomputation (0.2–3 s), and it must beat the quality of 1–4 spp physical
  frames (38–85 ms). An update in the 0.1–1 s range buys nothing over 8–85 spp
  of path tracing on this GPU.
- None of this tests whether the proposed method can reach such a budget, or
  at what quality. It also does not compare against denoised low-spp
  rendering, which the protocol excludes from the timing curve.

## UNRESOLVED QUESTION

- **Denoising.** A denoised 1–16 spp frame would move the physical baseline's
  quality/cost frontier substantially. The protocol excluded it from the main
  curve, so it was not measured. It is the most important missing baseline
  for the "quality at a budget" comparison.
- **Converged-quality target.** No reference-quality spp (where the error
  stops mattering) was established for BMW27. The FPS gaps are per spp, not
  per quality level.
- **Scene-scale vs framing.** The XL series changes the camera distance with
  scale, so per-pixel transport complexity and on-screen object size are
  confounded. A fixed-camera series (adding only visible bays inside a fixed
  frustum) would separate them.
- **288-car rebuild super-linearity.** Is it intrinsic or memory pressure
  (55–56 GB of 63.6 GB)? Not separated.
- **E attribution.** BVH work cannot be separated from the other Cycles
  device managers at debug log level. Only the build count ("Tasks handled")
  is known.
- **Interactive viewport path.** Blender's interactive viewport rendering
  (progressive, incremental by design) was not measured. Headless Blender
  has no viewport.
- **Generality.** One scene family (an open studio with two car models), one
  rigid mover, one GPU, one Blender build. Interior or closed-room scenes with
  deep indirect transport may weigh bounce depth and path-tracing cost
  differently.

## Artifacts

- Machine-readable records: `results/full_recompute_benchmark/runs/v1_3060f1d/`
  (one `<condition>.json` per process, each with environment, scene hash,
  accounting, mover transforms, every render's phase record, summary
  statistics, FPS/gaps and host GPU memory/clock samples; the raw Cycles log
  of each process sits beside it).
- XL scenes and build manifests: `results/full_recompute_benchmark/scenes/`.
- Report: `results/full_recompute_benchmark/report/v1_3060f1d/`.
- Reviewer exports with provenance (`manifest.json`, sha256 of every file and
  every source record): `results/evaluation/26/`.

## Commits

- `4b0db56` merge of `origin/main` (`5cb61ed`, `docs/Architecture.md`)
- `3060f1d` benchmark protocol, Blender timing, XL builder, tests (measurement code for all runs)
- `93e04af` driver `--resume`, report and preview scripts
- this worklog, together with `make_report.py` additions made after the runs
  (pixelwise stationary-change statistic, residual segment in the breakdown
  plot, label placement); analysis only, no measurement code changed
