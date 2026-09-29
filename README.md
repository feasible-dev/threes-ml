---
status: current
updated: 2026-09-26T21:29:45-04:00
scope: Threes ML
---

# Threes ML

A playable Threes-inspired game and a research project for training stronger agents.

**Start here:** [Current research status](docs/CURRENT_STATUS.md). For agents, read [AGENTS.md](AGENTS.md). The [documentation index](docs/README.md) distinguishes accepted decisions, proposals, references and historical evidence.

## Current capabilities

- GPU recurrent PPO from random weights: a 13,995,453-parameter GELU/shared-LSTM network whose released-weight inference computation was verified against the reference ONNX model.
- Engine v2, checked against 10,000 controlled C/Python transitions. Full published training equivalence and published playing strength have not been reproduced.
- Complete checkpoint continuation, visible terminal progress, per-run source/configuration provenance, comparisons and reports.
- CPU policy viewer with manual play, AI advice, autoplay, undo and live activation inspection.

Architecture, rewards and limits are in the [implemented method](docs/REPRODUCTION.md) and [fidelity audit](docs/FIDELITY_AUDIT.md). Research progress and the next budget are recorded only in [current status](docs/CURRENT_STATUS.md), with links to dated evidence.

## Useful commands

Run from the project root in PowerShell:

```powershell
# CPU viewer; published pretrained reference, initially paused.
.\watch_policy.ps1 -Features

# Read-only checkpoint listing and continuation preview.
.\resume_reproduction.ps1 -List
.\run_long_reproduction.ps1 -Preview

# Documentation metadata and local links; no training.
.\.venv\Scripts\python.exe src/check_repository.py
```

Use the [script index](scripts/README.md) and [continuation guide](docs/CONTINUING_RUNS.md) for execution commands. Continuation budgets are additional: rerunning a batch does not attach to the existing process. The user starts long jobs in a visible terminal.

## Repository map

| Location | Purpose |
| --- | --- |
| [docs/](docs/README.md) | Current status, method, operations and dated decisions |
| [docs/archive/](docs/archive/README.md) | Earlier plans, MLP experiments and historical engine audits |
| [src/](src/README.md) | Current recurrent pipeline, inspector and retained historical game |
| [configs/](configs/README.md) | Current v2 and historical parameter templates; run copies are frozen separately |
| [scripts/](scripts/README.md) | Command index and archived launchers; current user entry points stay at the root |
| [results/](results/README.md) | Timestamped runs, comparisons, evidence snapshots and generated catalog |
| [external_models/kiok/](external_models/kiok/README.md) | Downloaded reference provenance; distinct from our trained models |
| `tests/` | Engine, learning, continuation, reporting and inspector checks |
| `.external/` | Ignored upstream checkout, compiled oracle, scratch diagnostics and continuation lock |

`.venv-repro` is the CUDA training environment; `.venv` serves the CPU viewer and historical pipeline. `config.json` configures the historical game, not recurrent training. Old experiment launchers are under `scripts/legacy/`; exact historical reruns use each run's source snapshot.

## Verification

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Use focused tests during active training. Neither documentation cleanup nor catalog refresh establishes new model-performance evidence.
