# 2026-09-26_142800_article_merge_64_s53

> Generated: 2026-09-28T19:13:15-04:00. Training cutoff: 300,032 transitions. Snapshot, not a live status monitor.

**Status:** complete. **Question:** Is the 64-64 merge-depth policy result reproducible across training seeds?

## Provenance and code

- Started: 2026-09-26T14:31:02-04:00
- Git commit: unavailable (no accessible Git repository)
- Code: Source copy archived in `metadata/source/` with SHA-256 hashes.
- [Provenance and hashes](metadata/provenance.json)

## Parameters

| Item | Value |
| --- | --- |
| Method | MaskablePPO |
| Policy network | 35 → 64-64 → 4, activation ReLU |
| Value network | 35 → 64-64 → 1 |
| Trainable parameters | 13253 |
| Reward | rank_weighted_merges_v1: sum of merged tile levels (3=1, 6=2, 12=3, ...) per legal move; -1 otherwise |
| Training seed | 53 |
| Initial budget / cumulative transitions | 300000 / 300032 |
| PPO: rollout / batch / epochs | 512 / 128 / 8 |
| PPO: learning rate / γ / GAE λ | 0.0003 / 0.995 / 0.95 |
| PPO: clip / value weight / entropy weight | 0.2 / 0.5 / 0.0 |
| Periodic evaluation | 100 games from seed 120000, every 24576 steps |

4 legal-action indicators supplied separately from the observation.

## Results

| Measure | Value |
| --- | ---: |
| Cumulative duration | 178.0 s |
| Latest periodic mean | 472.9 |
| Best periodic mean | 476.5 |
| Held-out mean | 416.0 (300 games) |
| Reference on these games | 1,358.7 |
| Share of reference | 30.6% |

Periodic evaluations track training; held-out games support the final comparison.

## Interpretation and retention

**Relevance:** medium. 
**Finding:** Held-out mean 416.0 on 300 games, 89.1 below the seed-52 counterpart; no game reached tile 384. Tracking stayed near 473 points late in training. Mean policy entropy fell from about 1.27 to 0.03 nats, suggesting an almost deterministic policy without a clear late score gain.

**Suggested retention:** trim later. 
Keep per-game data and learning curves to document training-seed variability; periodic checkpoints are candidates for later cleanup.

## Files

- [Frozen configuration](metadata/config.json)
- [Method description](metadata/method.json)
- [Per-update metrics](data/metrics.csv)
- [Charts](report/report.html)
- [Final model](models/model.zip)
- [Checkpoints](models/checkpoints/)
- [Editable review](metadata/review.json)
- [Held-out evaluation](evaluation/holdout.csv)
