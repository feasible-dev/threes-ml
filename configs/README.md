---
status: reference
updated: 2026-09-26T21:29:45-04:00
scope: Experiment configurations
---

# Experiment configurations

These are parameter templates, not instructions to launch every experiment. See [current status](../docs/CURRENT_STATUS.md) and the [script index](../scripts/README.md). Effective run settings live in each run's frozen `metadata/config.json`.

## Current recurrent reproduction

Use `reproduction_v2_control.json`, `reproduction_v2_curriculum.json`, and `reproduction_v2_curriculum_decay.json` through `run_reproduction_v2.ps1`. Optional `reproduction_v2_decay.json` completes the two-by-two design. V2 adds explicit reward scale, timeout bootstrapping, and an absolute-transition learning-rate schedule. The default batch uses 10 million transitions per run. See the [v2 guide](../docs/REPRODUCTION_V2.md).

### Historical initial pilot configurations

[`reproduction_base.json`](reproduction_base.json) and [`reproduction_shaped.json`](reproduction_shaped.json) use the same 1,024-unit LSTM, PPO settings, seed protocol, and transition budget. The shaped version adds top-row, corner, and monotonicity rewards. Both begin from random weights; scaffolding was disabled in the [completed first comparison](../results/comparisons/2026-09-26_153305_recurrent/comparison.md).

| Section | Controls |
| --- | --- |
| `model` | LSTM and encoder width |
| `environment` | Structural reward, optional scaffolding, terminal rank |
| `ppo` | Environment count, rollout, minibatch, epochs, learning settings |
| `training` | Run seed, transitions, device, checkpoint interval |
| `evaluation` | Periodic validation games, seeds, and batching |

The launcher overrides the seed and requested transition count, then copies the effective configuration into each run. See the [reproduction guide](../docs/REPRODUCTION.md) for the reward, architecture, evidence, and known differences from the publication.

## Historical MLP configurations

`ppo*.json` and `article_merge_*.json` describe completed MaskablePPO experiments on the historical simulator. Their run cards preserve effective settings and results. They remain available for auditing those results, while the recurrent configurations define the current work.

Historical JSON paths remain stable for the retained MLP code and launchers. They are not active candidates merely because they share this directory. Changing a template does not change an already frozen run configuration.
