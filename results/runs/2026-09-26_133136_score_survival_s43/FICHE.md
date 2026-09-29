# 2026-09-26_133136_score_survival_s43

> Generated: 2026-09-28T19:13:15-04:00. Training cutoff: 200,064 transitions. Snapshot, not a live status monitor.

**Status:** complete. **Question:** Does scaled score gain plus a survival bonus improve seed 43?

## Provenance and code

- Started: 2026-09-26T13:42:51-04:00
- Git commit: unavailable (no accessible Git repository)
- Code: Source copy archived in `metadata/source/` with SHA-256 hashes.
- [Provenance and hashes](metadata/provenance.json)

## Parameters

| Item | Value |
| --- | --- |
| Method | MaskablePPO |
| Policy network | 35 → 64-64 → 4, activation Tanh |
| Value network | 35 → 64-64 → 1 |
| Trainable parameters | 13253 |
| Reward | scaled_score_survival_v1: official score gain / 27 + 0.05 per legal move; -1 otherwise |
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
| Cumulative duration | 134.6 s |
| Latest periodic mean | 389.0 |
| Best periodic mean | 435.6 |
| Held-out mean | 367.4 (300 games) |
| Reference on these games | 1,414.2 |
| Share of reference | 26.0% |

Periodic evaluations track training; held-out games support the final comparison.

## Interpretation and retention

**Relevance:** high. 
**Finding:** Held-out mean 367.4, 150.2 points below its control; tile-192 rate falls from 2.0% to 0.3%. Explained variance rises to about 0.25 late in training, showing that easier value prediction alone did not improve the policy.

**Suggested retention:** trim later. 
Keep the final model and diagnostics for this informative negative result; review intermediate checkpoints during cleanup.

## Files

- [Frozen configuration](metadata/config.json)
- [Method description](metadata/method.json)
- [Per-update metrics](data/metrics.csv)
- [Charts](report/report.html)
- [Final model](models/model.zip)
- [Checkpoints](models/checkpoints/)
- [Editable review](metadata/review.json)
- [Held-out evaluation](evaluation/holdout.csv)
