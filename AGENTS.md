# Working On This Repo

## Selective_Recompute research gate — Corrected object-level intent (2026-10-12)

**User-confirmed definition:** Selective recomputation means **recompute the ENTIRE neural transport model for each scene object whose learned transport was invalidated by a geometry edit**, while keeping **entire models** belonging to transport-valid objects unchanged.

Never reinterpret this as selecting only triplane cells, local features, a subset of MLP layers, or another intra-object parameter mask.

**Geometry-dirty D_G and transport-dirty D_T differ.** When a milk pot moves, stationary teapot/tray object models can be transport-invalid because their indirect GI changes. A mover-only rebuild does not necessarily suffice.

**Active research decision frontier: O0/O1 object-level feasibility.** First show high-quality separately-owned neural GI object models can be trained and composed against full-scene GI (O0); then, with an offline oracle object-dirty set, fully refit only invalid object models (O1). Only after this evidence and a new user decision may work on deployable object-to-object transport dependency/hierarchy proceed.

**Matched controls:** frozen T0 object models; all-object scratch reconstruction; all-object full-parameter warm-start refit; directly geometry-changed object(s) only; oracle transport-dirty object(s) full-model refit; optional simple spatial object selection. Use the same scene partition, model family, reference supervision and refit contract. **Every parameter of each selected object model is eligible to update.**

**Historical scope:** Worklog 30 validly tested selective triplane-cell masking inside a single 8DNA all-parts asset while shared layers were frozen; its negative result is **not** a test of this clarified object-level approach. Keep Worklogs 21–30 unchanged. Superseded intra-object hypothesis archived in docs/history/ARCHITECTURE_INTRA_OBJECT_FEATURE_REFIT_2026-10-08.md; previous lifecycle-separation hypothesis archived in docs/history/ARCHITECTURE_LIFECYCLE_SEPARATION_2026-10-07.md.

**Do NOT** resume intra-object feature masking as the main experiment, assume the historical single-asset 8DNA teaset is already decomposed into independent object models, ignore stationary transport-dirty receivers, implement an object hierarchy before proving oracle whole-object update viability, or automatically advance to larger implementation after a result.

Keep actual code changes and future experiments subject to user research decisions. Doc correction is not approval to implement O0/O1.

---

## Research context bootstrap

Before substantial research implementation, experiment design, or architecture interpretation, read:

1. `docs/RESEARCH_CENTRIC_TOPIC.md`
2. `docs/RESEARCH_ROADMAP.md`
3. `docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md`
4. `docs/Architecture.md`

In particular, do not confuse implementation/development baselines with scientific replication or architecture-contrast baselines, and do not generalize a negative result from one model/asset pair into project-level falsification without the evidence breadth defined in the strategy document.

## Living-document synchronization

Worklogs are historical evidence; the central documents above are the current research state.

When a substantive batch causes a **major decision** or **phase transition**—for example a gate closes, a working hypothesis is adopted or rejected, a baseline role changes, a kill condition is triggered, the next architecture question changes, or implementation is authorized—update the relevant living document(s) in the same batch after the evidence is interpreted.

At minimum consider:

- `docs/Architecture.md` for representation ownership, selected working architecture, update contract, or implementation frontier;
- `docs/RESEARCH_ROADMAP.md` for milestone status, current decision frontier, next task, or stop condition;
- `docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md` for baseline roles, evidence hierarchy, or generality scope.

Do not rewrite prior worklogs to make them agree with the new interpretation. Preserve them as append-only history and record the new current interpretation in the living documents.

Do not update living documents for trivial implementation details or every diagnostic result; update them when the durable research direction or decision state actually changes.

## Worklog conventions

`docs/worklogs/` is the base location for human-readable worklogs — Korean or English
narrative explaining a JSON result's purpose, verdict and limits.
Machine-reproducible verdicts and pipeline input/output stay in JSON; a
worklog is not a place to paste one wholesale (an agent once filed
`docs/worklogs/44` as raw JSON — recognize that as a mistake, not a pattern to repeat).

Create a numbered worklog for a substantive session: an experiment, analysis,
implementation, structural change, or decision whose rationale and limits need
to be traceable. Do not create one for minor housekeeping such as a small path,
formatting, or configuration adjustment unless it materially changes an
experiment's execution or interpretation.

Worklog history is one numbered worklog per qualifying session, never a single mutable
handoff doc that gets overwritten (the old `docs/worklogs/11_SESSION_HANDOFF_*.md`
pattern was retired 2026-08-24 for exactly that reason):

- Find the highest-numbered file already in `docs/worklogs/` and use the next
  integer, even if that file doesn't otherwise look like a normal worklog
  (wrong extension, missing `_WORKLOG_` infix, JSON instead of Markdown) —
  fix that file's own format if asked to, but don't reuse its number.
- Name the file for what the session actually did, not its date —
  `docs/worklogs/12_WORKLOG_YAW_TAIL_ATTRIBUTION_AND_A11.md`, not
  `docs/worklogs/12_WORKLOG_2026-08-25.md`. Every worklog filename carries
  `_WORKLOG_` as an infix. The date belongs inside the document, on its
  title line (`# Worklog — <Descriptive Name> (<date>)`), not primarily in
  the filename.
- Never edit or delete a previous worklog to "update" it. A worklog is
  append-only; a correction to an earlier one's claims is a new worklog
  that says so explicitly (e.g. docs/worklogs/38 correcting docs/worklogs/37's wording,
  docs/worklogs/40 correcting docs/worklogs/39's), not a silent rewrite.
- Scope a worklog to session narrative: what was done, why, which commits,
  what's still open. Durable facts that keep being updated over many
  sessions — experiment tables, data-contract fingerprints, gate
  definitions, environment/server setup — belong in their own living
  document instead of being duplicated into every worklog that touches
  them (`docs/10_TEMPORAL_LIFTER_IMPROVEMENT_ABLATION.md` for the lifter
  ablation table, `docs/06_SERVER_AI_AGENT_TRAINING_RUNBOOK.md` for server
  setup) — reference it, don't repeat it.
- A worklog is bound by the same discipline as everything else here: label
  every result by evaluation regime and never compare across regimes
  without the label, and never state a gate/case verdict stronger than what
  was actually measured that session.

## Evaluation artifact conventions

Human-reviewable intermediate and final artifacts (images, GIFs, videos, and
their evaluation manifest entries) belong under a folder keyed by the session
worklog number: `results/evaluation/<number>/`. For example, artifacts for
`docs/worklogs/17_WORKLOG_*.md` go in `results/evaluation/17/`.

- Create the matching evaluation folder for every substantive session that
  produces reviewable output, even if earlier numbers have no evaluation
  folder. Number gaps are intentional; never reuse another worklog's folder.
- Keep source experiment outputs in their experiment directories. Copy only
  reviewer-facing artifacts into the matching evaluation folder and record
  their provenance/hash in that folder's manifest.
- Do not move, rename, overwrite, or retrospectively reorganize artifacts in
  a prior numbered evaluation folder unless the user explicitly asks. A new
  worklog/session gets a new numbered folder.

## Running long jobs in local WSL

When the `wsl.exe` client that started a job is killed, the job dies with it.
It gets SIGHUP, and WSL then idles the distro down about 15 s later. That is
what the old "WSL DXG restart" reports were
(`docs/worklogs/16_WORKLOG_WSL_DXG_ROOT_CAUSE_DIAGNOSIS.md`). An agent's
shell-tool timeout is exactly such a kill.

- Never run a WSL render, training, dataset generation or loader job that
  may outlast the tool timeout as a foreground tool call. Treat anything
  over about a minute as at risk.
- Run such a job either as a background task, so the client is not subject
  to the tool timeout, or through
  `experiments/dynamic_transport_failure/scripts/wsl_run_detached.sh`.
  Poll the launcher's `<job>.status` file from short, separate `wsl.exe`
  calls.
- Do not rely on `systemd-run`, a bare `nohup ... &`, or a `setsid` job
  whose launching client is later killed. WSL's idle shutdown kills all of
  them.
- A job that ends without its completion marker is not evidence of a GPU,
  DXG or RNA fault until the launching client is shown to have survived.
  To check, rerun it under
  `experiments/dynamic_transport_failure/scripts/capture_wsl_gpu_pv_failure.ps1`
  and read `termination` in its `summary.json`. Only `job_failed` or
  `killed_without_exit` points at the job or the runtime;
  `client_teardown` means the client was lost again
  (`docs/worklogs/20_WORKLOG_WSL_CAPTURE_CLIENT_TEARDOWN_CLASSIFICATION.md`).
