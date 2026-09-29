---
status: reference
updated: 2026-09-26T21:29:45-04:00
scope: Reading the experiments
---

# Reading the experiments

The [catalog](EXPERIENCES.md) and [sortable CSV](experiences.csv) list tracked runs as of their generation time. For the accepted research direction, read [current status](../docs/CURRENT_STATUS.md). Live metrics may be newer than catalog rows or cards. Each run has a timestamped folder under `runs/` and a `FICHE.md` with parameters, results, interpretation, and retention advice. Shared evaluations live under `comparisons/`; early heuristic benchmarks are under `references/`.

## Run directory

Interactive policy-inspection sessions are stored separately under `inspections/`, with model/rule metadata, a decision event log and optional recurrent-state exports. They are not benchmark runs and do not enter the training catalog. See the [viewer guide](../docs/POLICY_INSPECTOR.md).

```text
runs/YYYY-MM-DD_HHMMSS_description/
  FICHE.md                   readable run summary
  metadata/config.json       frozen choices
  metadata/method.json       effective method and limitations
  metadata/provenance.json   start time, source hashes, optional Git commit
  metadata/source/           source snapshot from the start
  metadata/resumes/          source snapshots for resumed segments
  CONTINUATIONS.md           index of sessions made with the continuation launcher
  metadata/continuations/   per-session summary, terminal log and prior results
  metadata/review.json       relevance and retention judgment, when reviewed
  data/metrics.csv           per-update learning and validation metrics
  data/episodes.csv          completed training episodes and seeds (recurrent runs)
  models/                    selected model and complete checkpoints
  evaluation/                per-game evaluations and comparison link
  report/report.html         standalone charts
```

Earlier runs may lack some files. Their cards state the limits of their provenance. Reviews are separate from measured results; `pending review` means no research judgment has been entered. The catalog never deletes a run.

## Current recurrent runs

To extend an existing run after the current batch finishes, use the [continuation launcher](../docs/CONTINUING_RUNS.md). `resume_reproduction.ps1 -List` and `-Preview` are read-only; `-AdditionalSteps` specifies an extra training budget. Every executed continuation gets a terminal log, summary and archive of prior results.

The [v2 reference](../docs/REPRODUCTION_V2.md) explains the initial experiment design. The following commands create new models; they do not continue the existing batch. Choose experiments using [current status](../docs/CURRENT_STATUS.md).

```powershell
.\run_reproduction_v2.ps1 -Preview
.\run_reproduction_v2.ps1
```

Open the path printed for a run in a second terminal to watch live charts:

```powershell
.\.venv-repro\Scripts\python.exe src/dashboard.py results\runs\YOUR_RUN_DIRECTORY
```

Then open `http://127.0.0.1:8765/`. The completed chart is `report/report.html` inside the run folder. To continue the last complete recurrent checkpoint, where `--steps` means **additional** transitions:

```powershell
.\.venv-repro\Scripts\python.exe -u src/train_reproduction.py --resume results\runs\YOUR_RUN_DIRECTORY --steps 2000000
```

If interrupted during a rollout, continuation returns to the last fully saved update. The run keeps its training seed and absolute-step learning-rate schedule. Revision-1 checkpoints cannot resume under the revised engine; use their archived source for historical continuation. The comparison at the end of the launcher uses validation games; the final-test seed range remains reserved.

Historical MLP runs used `scripts/legacy/run_experiments.ps1`, `scripts/legacy/run_next_experiments.ps1`, and `scripts/legacy/run_article_experiments.ps1`. Their settings and findings remain in their run cards and comparison folders. Refresh only the catalog with `.\.venv-repro\Scripts\python.exe src/experiment_catalog.py --catalog-only`; this leaves per-run cards and evidence untouched. A normal invocation also regenerates run cards.
