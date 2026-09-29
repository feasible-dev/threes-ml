# 2026-09-26_133136_stronger_updates_s42

> Generated: 2026-09-28T19:13:15-04:00. Training cutoff: 200,192 transitions. Snapshot, not a live status monitor.

**Status:** complete. **Question:** Do larger rollouts and more PPO passes improve seed 42 at the same transition budget?

## Provenance and code

- Started: 2026-09-26T13:35:55-04:00
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
| Training seed | 42 |
| Initial budget / cumulative transitions | 200000 / 200192 |
| PPO: rollout / batch / epochs | 512 / 128 / 8 |
| PPO: learning rate / γ / GAE λ | 0.0003 / 0.99 / 0.95 |
| PPO: clip / value weight / entropy weight | 0.2 / 0.5 / 0.0 |
| Periodic evaluation | 100 games from seed 100042, every 8192 steps |

4 legal-action indicators supplied separately from the observation.

## Results

| Measure | Value |
| --- | ---: |
| Cumulative duration | 138.0 s |
| Latest periodic mean | 489.6 |
| Best periodic mean | 639.1 |
| Held-out mean | 496.6 (300 games) |
| Reference on these games | 1,414.2 |
| Share of reference | 35.1% |

Periodic evaluations track training; held-out games support the final comparison.

## Interpretation and retention

**Relevance:** high. 
**Finding:** Held-out mean 496.6, 35.8 points above its seed-matched control; tile-192 rate remains 1.3%. The gain is modest and uncertain at two training seeds. Best periodic mean 639.1 exceeds the final periodic mean 489.6.

**Suggested retention:** complete. 
Keep the final model and the best periodic checkpoint, metrics, configuration, and source snapshot for a controlled follow-up.

## Files

- [Frozen configuration](metadata/config.json)
- [Method description](metadata/method.json)
- [Per-update metrics](data/metrics.csv)
- [Charts](report/report.html)
- [Final model](models/model.zip)
- [Checkpoints](models/checkpoints/)
- [Editable review](metadata/review.json)
- [Held-out evaluation](evaluation/holdout.csv)
