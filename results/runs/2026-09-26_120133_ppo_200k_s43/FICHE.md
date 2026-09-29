# 2026-09-26_120133_ppo_200k_s43

> Generated: 2026-09-28T19:13:15-04:00. Training cutoff: 200,064 transitions. Snapshot, not a live status monitor.

**Status:** complete. **Question:** Does the same PPO configuration produce a similar result with another training seed?

## Provenance and code

- Started: 2026-09-26T12:01:33-04:00
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
| Training seed | 43 |
| Initial budget / cumulative transitions | 200000 / 200064 |
| PPO: rollout / batch / epochs | 128 / 64 / 2 |
| PPO: learning rate / γ / GAE λ | 0.0003 / 0.99 / 0.95 |
| PPO: clip / value weight / entropy weight | 0.2 / 0.5 / 0.0 |
| Periodic evaluation | 100 games from seed 100042, every 8192 steps |

4 legal-action indicators supplied separately from the observation.

## Results

| Measure | Value |
| --- | ---: |
| Cumulative duration | 127.1 s |
| Latest periodic mean | 523.7 |
| Best periodic mean | 558.2 |
| Held-out mean | 532.5 (300 games) |
| Reference on these games | 1,378.5 |
| Share of reference | 38.6% |

Periodic evaluations track training; held-out games support the final comparison.

## Interpretation and retention

**Relevance:** high. 
**Finding:** The best final neural model so far, but still below the expected-spawn heuristic. Its best intermediate evaluation occurred early and did not persist.

**Suggested retention:** complete. 
Keep the final model, best intermediate checkpoint, configuration, and curves for subsequent controlled experiments.

## Files

- [Frozen configuration](metadata/config.json)
- [Method description](metadata/method.json)
- [Per-update metrics](data/metrics.csv)
- [Charts](report/report.html)
- [Final model](models/model.zip)
- [Checkpoints](models/checkpoints/)
- [Editable review](metadata/review.json)
- [Held-out evaluation](evaluation/holdout.csv)
