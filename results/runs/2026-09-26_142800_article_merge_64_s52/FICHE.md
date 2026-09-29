# 2026-09-26_142800_article_merge_64_s52

> Generated: 2026-09-28T19:13:15-04:00. Training cutoff: 300,032 transitions. Snapshot, not a live status monitor.

**Status:** complete. **Question:** Does a 64-64 ReLU PPO policy with merge-depth reward outperform earlier PPO and heuristics?

## Provenance and code

- Started: 2026-09-26T14:28:01-04:00
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
| Training seed | 52 |
| Initial budget / cumulative transitions | 300000 / 300032 |
| PPO: rollout / batch / epochs | 512 / 128 / 8 |
| PPO: learning rate / γ / GAE λ | 0.0003 / 0.995 / 0.95 |
| PPO: clip / value weight / entropy weight | 0.2 / 0.5 / 0.0 |
| Periodic evaluation | 100 games from seed 120000, every 24576 steps |

4 legal-action indicators supplied separately from the observation.

## Results

| Measure | Value |
| --- | ---: |
| Cumulative duration | 177.0 s |
| Latest periodic mean | 517.5 |
| Best periodic mean | 540.8 |
| Held-out mean | 505.1 (300 games) |
| Reference on these games | 1,358.7 |
| Share of reference | 37.2% |

Periodic evaluations track training; held-out games support the final comparison.

## Interpretation and retention

**Relevance:** medium. 
**Finding:** Held-out mean 505.1 on 300 games; no game reached tile 384. The final tracking mean was 517.5 after 300,032 transitions. This is below the same-seed spawn heuristic (1,358.7) and prior stronger-updates seed 43 (560.6).

**Suggested retention:** trim later. 
Keep metrics, method, source snapshot, and held-out scores as the compact-network control; consider removing redundant periodic checkpoints during cleanup.

## Files

- [Frozen configuration](metadata/config.json)
- [Method description](metadata/method.json)
- [Per-update metrics](data/metrics.csv)
- [Charts](report/report.html)
- [Final model](models/model.zip)
- [Checkpoints](models/checkpoints/)
- [Editable review](metadata/review.json)
- [Held-out evaluation](evaluation/holdout.csv)
