# 2026-09-26_115920_ppo_200k_s42

> Generated: 2026-09-28T19:13:15-04:00. Training cutoff: 200,064 transitions. Snapshot, not a live status monitor.

**Status:** complete. **Question:** Does 200,000-transition PPO improve on the prototype and catch up with the heuristics?

## Provenance and code

- Started: 2026-09-26T11:59:20-04:00
- Git commit: unavailable (no accessible Git repository)
- Code: The exact code for this historical run was not archived; current files may differ.
- [Provenance and hashes](metadata/provenance.json)

## Parameters

| Item | Value |
| --- | --- |
| Method | MaskablePPO |
| Policy network | 35 → 64-64 → 4, activation Tanh |
| Value network | 35 → 64-64 → 1 |
| Trainable parameters | 13253 |
| Reward | log_score_gain_v1: log1p(official score gain) for a legal move; -1 otherwise |
| Training seed | 42 |
| Initial budget / cumulative transitions | 200000 / 200064 |
| PPO: rollout / batch / epochs | 128 / 64 / 2 |
| PPO: learning rate / γ / GAE λ | 0.0003 / 0.99 / 0.95 |
| PPO: clip / value weight / entropy weight | 0.2 / 0.5 / 0.0 |
| Periodic evaluation | 100 games from seed 100042, every 8192 steps |

4 legal-action indicators supplied separately from the observation.

## Results

| Measure | Value |
| --- | ---: |
| Cumulative duration | 125.6 s |
| Latest periodic mean | 495.6 |
| Best periodic mean | 495.6 |
| Held-out mean | 411.4 (300 games) |
| Reference on these games | 1,378.5 |
| Share of reference | 29.8% |

Periodic evaluations track training; held-out games support the final comparison.

## Interpretation and retention

**Relevance:** medium. 
**Finding:** The final model beats random play but remains well below the position and expected-spawn heuristics. It also illustrates variation across training seeds.

**Suggested retention:** trim later. 
Keep the configuration, metrics, held-out scores, and at least the final model until the next comparison. The 26 checkpoints can be reviewed for cleanup.

## Files

- [Frozen configuration](metadata/config.json)
- [Method description](metadata/method.json)
- [Per-update metrics](data/metrics.csv)
- [Charts](report/report.html)
- [Final model](models/model.zip)
- [Checkpoints](models/checkpoints/)
- [Editable review](metadata/review.json)
- [Held-out evaluation](evaluation/holdout.csv)
