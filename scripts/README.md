---
status: reference
updated: 2026-09-28T18:08:32-04:00
scope: Script entry points
---

# Script entry points

Run commands from the project root. The [current status](../docs/CURRENT_STATUS.md) owns experiment selection; this page explains entry points.

## Current commands

| Command | Purpose | Execution |
| --- | --- | --- |
| `watch_policy.ps1` | Play with advice, autoplay, activation viewer | CPU window; user starts it |
| `resume_reproduction.ps1 -List` / `-Preview` | Inspect complete checkpoints and continuation compatibility | Read-only |
| `resume_reproduction.ps1 -Run PATH -AdditionalSteps N` | Continue one existing v2 run; frequent progress remains, and terminal validation lines show reach 192/384/768 | Visible training; extra budget |
| `run_long_reproduction.ps1 -Preview` | Inspect the former equal-budget pair plan | Read-only; plan superseded September 27 |
| `run_long_reproduction.ps1` | Historical equal-budget pair launcher | Do not launch under the current decision; adds its budget again |
| `run_reproduction_v2.ps1 -Preview` | Describe a fresh v2 batch | Read-only |
| `run_reproduction_v2.ps1` | Start fresh v2 models | Does not resume existing runs |
| `run_reproduction_forks.ps1 -ParentRun PATH -ParentSteps N -Preview` | Inspect the four isolated masking/LR arms | Read-only |
| `run_reproduction_forks.ps1 -ParentRun PATH -ParentSteps N -All` | Run four fixed-budget forks, then paired development comparison | Visible long training; original pilot completed |
| `run_reproduction_forks.ps1 -ResumeRun PATH` | Resume one interrupted fork to its allocated endpoint | Visible training; remaining budget only |
| `extend_reproduction_fork.ps1 -ParentFork PATH -AdditionalSteps N` | Branch a completed masking/LR fork into a new fixed-budget child run | Visible long training; preserves the completed parent |
| `run_masked_lr_probe.ps1 -ParentFork PATH -AdditionalSteps N` | Run masked LR `0.00005` and `0.00010` from one completed fork, then paired development comparison | Visible training/evaluation; current short probe |
| `run_masked_lr_probe.ps1 -ResumeRun PATH` / `-CompareGroup GROUP` | Recover a custom-LR probe arm or its paired comparison | Visible training or evaluation; inspect with `-Preview` |
| `extend_masked_lr_probe.ps1 -ParentFork PATH -AdditionalSteps N` | Continue the completed masked LR `0.00010` probe in a new fixed-budget run | Visible long training; current selected continuation |
| `extend_masked_lr_probe.ps1 -ResumeRun PATH` | Resume an interrupted long custom-LR child to its allocated endpoint | Visible training; inspect with `-Preview` |
| `run_plateau_lr_probe.ps1 -Preview` / `run_plateau_lr_probe.ps1` | Assess the 330M endpoint, then train and compare matched LR `0.00010` versus `0.00005` children | Visible evaluation and training; current selected pilot |
| `run_plateau_lr_probe.ps1 -ResumeRun PATH` / `-CompareGroup GROUP` | Resume an interrupted plateau arm or finish its comparison | Visible training or evaluation; inspect with `-Preview` |
| `setup_reproduction.ps1` | Install CUDA training environment | Changes dependencies; setup operation |
| `.\.venv\Scripts\python.exe src/check_repository.py` | Check maintained docs, metadata and local links | Read-only, standard library |
| `.\.venv-repro\Scripts\python.exe src/experiment_catalog.py --catalog-only` | Refresh catalog without rewriting per-run cards | Writes catalog Markdown/CSV |

See [continuation/recovery](../docs/CONTINUING_RUNS.md) before selecting an additional budget. Do not start a second learner while the user's batch is using the GPU.

## Historical launchers

These moved out of the root to avoid presenting completed experiments as the next task:

| Path | Historical experiment |
| --- | --- |
| `scripts/legacy/run_experiments.ps1` | Generic CPU MaskablePPO series |
| `scripts/legacy/run_next_experiments.ps1` | Completed reward/update sweep, seeds 42/43 |
| `scripts/legacy/run_article_experiments.ps1` | Completed rank-merge MLP sweep, seeds 52/53 |
| `scripts/legacy/run_reproduction.ps1` | Initial recurrent base/shaped configurations, seed 71 |

All accept `-Preview`; the generic MLP launcher also requires `-Series`, `-Seeds` and `-Steps`. They resolve the project root from their location. They execute **current** source: in particular the old recurrent launcher now sees engine v2. Exact historical reruns require the source and dependencies frozen with the original run. Original commands inside frozen evidence retain their original paths.
