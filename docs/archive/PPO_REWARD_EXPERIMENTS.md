---
status: historical
updated: 2026-09-26T21:29:45-04:00
scope: Archived PPO reward and update experiment
---

# Archived PPO reward and update experiment

> Historical stage only. Earlier recommendations and agent instructions are superseded by [current status](../CURRENT_STATUS.md).

**Status: completed September 26, 2026.** This was the next experiment after the first 200,000-transition MLP runs. The [six-run comparison](../../results/comparisons/2026-09-26_133136_ppo_reward_updates/comparison.md) contains measured results and per-game data. The current research program is the [recurrent reproduction](../REPRODUCTION.md).

The experiment used the same 35-input, 64–64 Tanh network and requested 200,000 transitions per run. Each condition used training seeds 42 and 43:

| Condition | PPO rollout / batch / epochs | Reward |
| --- | --- | --- |
| Baseline | 128 / 64 / 2 | Logarithmic official-score gain |
| Stronger updates | 512 / 128 / 8 | Same as baseline |
| Score and survival | 128 / 64 / 2 | Score gain / 27 + 0.05 per legal move |

Periodic validation used 100 games beginning at seed 100042. The comparison used 300 games beginning at seed 300000. Stronger updates modestly improved mean score on both training seeds without improving tile-192 reach. The score-and-survival reward did worse. Two seeds are too few for a firm general conclusion. The comparison seeds have been inspected and should be treated as validation data in later decisions.

Configurations, source snapshots, metrics, model files, and run reviews remain under [`results/runs/`](../../results/runs). The original script `scripts/legacy/run_next_experiments.ps1` is retained for historical reproducibility.
