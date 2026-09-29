# 2026-09-25_183728_visual_demo_20k

> Generated: 2026-09-28T19:13:15-04:00. Training cutoff: 20,224 transitions. Snapshot, not a live status monitor.

**Status:** complete. **Question:** Are the dashboard, checkpoints, and resume workflow usable on a short run?

## Provenance and code

- Started: 2026-09-25T18:37:28-04:00
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
| Initial budget / cumulative transitions | 20000 / 20224 |
| PPO: rollout / batch / epochs | 128 / 64 / 2 |
| PPO: learning rate / γ / GAE λ | 0.0003 / 0.99 / 0.95 |
| PPO: clip / value weight / entropy weight | 0.2 / 0.5 / 0.0 |
| Periodic evaluation | 20 games from seed 100042, every 2048 steps |

4 legal-action indicators supplied separately from the observation.

## Results

| Measure | Value |
| --- | ---: |
| Cumulative duration | 12.1 s |
| Latest periodic mean | 304.2 |
| Best periodic mean | 486.3 |
| Held-out mean | — (— games) |
| Reference on these games | — |
| Share of reference | — |

Periodic evaluations track training; held-out games support the final comparison.

## Interpretation and retention

**Relevance:** medium. 
**Finding:** Tracking worked over 20,224 transitions and 11 evaluations; variation across 20 games prevents a performance conclusion.

**Suggested retention:** trim later. 
Keep the configuration, metrics, and report as a tracking example. Decide later whether every intermediate checkpoint is needed.

## Files

- [Frozen configuration](metadata/config.json)
- [Method description](metadata/method.json)
- [Per-update metrics](data/metrics.csv)
- [Charts](report/report.html)
- [Final model](models/model.zip)
- [Checkpoints](models/checkpoints/)
- [Editable review](metadata/review.json)
- Held-out evaluation unavailable
