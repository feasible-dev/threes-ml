---
status: historical
updated: 2026-09-26T21:29:45-04:00
scope: Code quality audit — September 26, 2026
---

# Code quality audit — September 26, 2026

> Historical stage only. Earlier recommendations and agent instructions are superseded by [current status](../CURRENT_STATUS.md).

This audit covers the earlier MLP training pipeline. The newer recurrent training code and its limitations are documented in [REPRODUCTION.md](../REPRODUCTION.md).

## Earlier weak points

- The PPO configurations repeated a flat list of fields. Training, the catalog, and the dashboard each interpreted them separately, without complete validation or a shared schema.
- Metric tracking found the run directory by walking two levels up from the CSV path. Moving that file could have sent checkpoints to the wrong place.
- `train.py` mixed directory creation, resuming, learning, and completion in one function. A directly interrupted run could lack a summary or HTML report.
- The dashboard's HTML, CSS, and JavaScript lived inside a long Python string that was hard to review.
- The catalog loaded every row of each CSV even though it needed only the latest update and evaluation summaries.

## Changes made

- [`experiment_config.py`](../../src/experiment_config.py) defines and validates the accepted fields while reading historical flat JSON files. New configurations and frozen run copies have clear sections; see the [configuration guide](../../configs/README.md).
- [`train.py`](../../src/train.py) separates run preparation, model restoration, and training. The callback receives its run directory explicitly. A summary is written before learning; the report and catalog are updated after an interruption too.
- The old `--save` option, which wrote a duplicate model, was removed. Each run already keeps `models/model.zip`.
- [`dashboard.html`](../../src/templates/dashboard.html) is separate from the Python server and included in each new source snapshot. Its JavaScript and CSS are formatted for review.
- [`experiment_catalog.py`](../../src/experiment_catalog.py) streams metric rows when building summaries, instead of holding the entire log in memory.
- Artifact paths remain centralized in [`run_layout.py`](../../src/run_layout.py). Historical results and configurations were not rewritten by this refactor.

## Current assessment

The separation between game engine, RL environment, training, and reporting is sound for short and medium experiments. Responsibilities are easier to follow, and tests cover both old and new configuration formats. No unused PPO configuration fields were found: every visible field has a role. Implicit library settings are recorded in `method.json` instead of being presented as configurable.

Two limits matter before training for hundreds of millions of transitions: the dashboard still rereads the full metrics CSV on each refresh, and every evaluation keeps a checkpoint. Their costs should be measured, followed by incremental chart updates or sampling and an explicit checkpoint retention policy. Resuming preserves weights and optimizer state, but does not reconstruct the game's exact random state in the middle of an episode.

## Verification

All 45 unit tests passed. A 128-transition run and its resume validated the new JSON, report, catalog, checkpoints, and source snapshot; the technical run was then removed. The dashboard JavaScript passes `node --check`.
