# 2026-09-26_133136_stronger_updates_s43

> Generated: 2026-09-28T19:13:15-04:00. Training cutoff: 200,192 transitions. Snapshot, not a live status monitor.

**Status:** complete. **Question:** Do larger rollouts and more PPO passes improve seed 43 at the same transition budget?

## Provenance and code

- Started: 2026-09-26T13:38:17-04:00
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
| Initial budget / cumulative transitions | 200000 / 200192 |
| PPO: rollout / batch / epochs | 512 / 128 / 8 |
| PPO: learning rate / γ / GAE λ | 0.0003 / 0.99 / 0.95 |
| PPO: clip / value weight / entropy weight | 0.2 / 0.5 / 0.0 |
| Periodic evaluation | 100 games from seed 100042, every 8192 steps |

4 legal-action indicators supplied separately from the observation.

## Results

| Measure | Value |
| --- | ---: |
| Cumulative duration | 140.5 s |
| Latest periodic mean | 612.7 |
| Best periodic mean | 612.7 |
| Held-out mean | 555.9 (300 games) |
| Reference on these games | 1,414.2 |
| Share of reference | 39.3% |

Periodic evaluations track training; held-out games support the final comparison.

## Interpretation and retention

**Relevance:** high. 
**Finding:** Held-out mean 555.9, 38.3 points above its seed-matched control; tile-192 rate remains 2.0%. This is the strongest neural result in the batch, still below the position heuristic. Periodic evaluation and held-out evaluation differ substantially.

**Suggested retention:** complete. 
Keep the final model, checkpoints, metrics, configuration, and source snapshot while examining update strength and longer training.

## Files

- [Frozen configuration](metadata/config.json)
- [Method description](metadata/method.json)
- [Per-update metrics](data/metrics.csv)
- [Charts](report/report.html)
- [Final model](models/model.zip)
- [Checkpoints](models/checkpoints/)
- [Editable review](metadata/review.json)
- [Held-out evaluation](evaluation/holdout.csv)
