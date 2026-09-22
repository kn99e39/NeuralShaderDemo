# Working On This Repo

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
