# 2026-09-26_142800_article_merge_256_s53

> Generated: 2026-09-28T19:13:15-04:00. Training cutoff: 300,032 transitions. Snapshot, not a live status monitor.

**Status:** complete. **Question:** Does widening the merge-depth policy from 64-64 to 256-256 help at training seed 53?

## Provenance and code

- Started: 2026-09-26T14:37:26-04:00
- Git commit: unavailable (no accessible Git repository)
- Code: Source copy archived in `metadata/source/` with SHA-256 hashes.
- [Provenance and hashes](metadata/provenance.json)

## Parameters

| Item | Value |
| --- | --- |
| Method | MaskablePPO |
| Policy network | 35 → 256-256 → 4, activation ReLU |
| Value network | 35 → 256-256 → 1 |
| Trainable parameters | 151301 |
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
| Cumulative duration | 198.6 s |
| Latest periodic mean | 473.6 |
| Best periodic mean | 549.2 |
| Held-out mean | 464.8 (300 games) |
| Reference on these games | 1,358.7 |
| Share of reference | 34.2% |

Periodic evaluations track training; held-out games support the final comparison.

## Interpretation and retention

**Relevance:** medium. 
**Finding:** Held-out mean 464.8 on 300 games versus 416.0 for the 64-64 seed-53 policy, but still below spawn (1,358.7) and prior stronger-updates seed 43 (560.6). No game reached tile 384. The width benefit did not repeat at seed 52.

**Suggested retention:** trim later. 
Keep the paired-seed width comparison and per-game scores; review redundant periodic checkpoints later.

## Files

- [Frozen configuration](metadata/config.json)
- [Method description](metadata/method.json)
- [Per-update metrics](data/metrics.csv)
- [Charts](report/report.html)
- [Final model](models/model.zip)
- [Checkpoints](models/checkpoints/)
- [Editable review](metadata/review.json)
- [Held-out evaluation](evaluation/holdout.csv)
