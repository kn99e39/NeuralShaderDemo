# Worklog — RNA Static Baseline Quality: Root-Cause Attribution (2026-10-14 – 2026-10-15)

## SESSION QUESTION

User prompt "RNA Static Baseline Quality — Root-Cause Attribution Batch" (2026-10-14). This is a
diagnostic batch, not architecture Batch 2.

> Is the existing RNA baseline a valid reference for evaluating Surface-Owned Neural Transport, and
> if not, which specific implementation, data, optimisation or representation factor prevents it
> from being one?

**Short answer:** _(filled in at the end of the session; see VERDICT)_

## RELATION TO EARLIER WORK

- **Worklog 32** stopped Batch 1 on its predeclared static-credibility gate: original RNA (A), val
  PSNR ≥ 25 dB, pooled, final weights. C1 scored 17.97 dB, C2 24.92 dB.
- **Earlier RNA numbers** come from different contracts: worklogs 9/15/17 (Rain), 22, 23, 27. Section
  1 below puts every one of them on a common footing.
- **Preserved unchanged:** worklogs 17–32, every checkpoint, dataset and the upstream source
  (`git diff` empty at `66b5b098`). Everything new lives under `experiments/rna_baseline_quality/`
  and `results/rna_baseline_quality/`.
- **Protocol:** `experiments/rna_baseline_quality/protocol/rbq_v1.json`, committed in `120e321`
  before any C1 control-C, D1 or B2 result existed. The 25 dB value is not changed.

_(sections below are completed at the end of the session)_
