# 2026-09-26_133136_baseline_s43

> Generated: 2026-09-28T19:13:15-04:00. Training cutoff: 200,064 transitions. Snapshot, not a live status monitor.

**Status:** complete. **Question:** What does the current PPO setup achieve with training seed 43 under the archived source?

## Provenance and code

- Started: 2026-09-26T13:33:45-04:00
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
| Cumulative duration | 125.9 s |
| Latest periodic mean | 523.7 |
| Best periodic mean | 558.2 |
| Held-out mean | 517.6 (300 games) |
| Reference on these games | 1,414.2 |
| Share of reference | 36.6% |

Periodic evaluations track training; held-out games support the final comparison.

## Interpretation and retention

**Relevance:** medium. 
**Finding:** Held-out mean 517.7 and tile-192 rate 2.0% over 300 games. This control reproduces the seed-dependent variation seen in the earlier PPO comparison.

**Suggested retention:** trim later. 
Keep the final model, source snapshot, metrics, and evaluation; review redundant intermediate checkpoints during cleanup.

## Files

- [Frozen configuration](metadata/config.json)
- [Method description](metadata/method.json)
- [Per-update metrics](data/metrics.csv)
- [Charts](report/report.html)
- [Final model](models/model.zip)
- [Checkpoints](models/checkpoints/)
- [Editable review](metadata/review.json)
- [Held-out evaluation](evaluation/holdout.csv)
