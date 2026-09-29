---
status: historical
updated: 2026-09-26T21:29:45-04:00
scope: Archived article-inspired MLP experiment
---

# Archived article-inspired MLP experiment

> Historical stage only. Earlier recommendations and agent instructions are superseded by [current status](../CURRENT_STATUS.md).

**Status: four runs completed September 26, 2026.** The [comparison](../../results/comparisons/2026-09-26_142800_article_merge/comparison.md) and run cards contain the results. This experiment was inspired by [Josiah Kiok's article](https://medium.com/@josiah-kiok/beating-threes-with-reinforcement-learning-ae074dd28a68); it did **not** reproduce the article's recurrent architecture or training scale. The current work is described in the [reproduction guide](../REPRODUCTION.md).

| Condition | Network | Training |
| --- | --- | --- |
| `article_merge_64` | 35-input, 64–64 ReLU MLP | MaskablePPO, 300,000 requested transitions, seeds 52 and 53 |
| `article_merge_256` | 35-input, 256–256 ReLU MLP | Same PPO and seeds |

Both trained from random weights in the historical local game with legal-action masks. Their reward summed merged-tile ranks before spawning; gamma was 0.995, rollout length 512, batch size 128, and epochs 8. Validation used seeds 120000–120099; the 300-game comparison used 500000–500299. Individual training episode seeds were not logged in this older pipeline.

Mean comparison scores were roughly **416–505**, versus **1,359** for the spawn heuristic on those initial seeds. Widening helped one seed but not the other. These results did not establish that longer training of this MLP would approach the published agent. The comparison range has been inspected and is now validation data.

The historical launch script `scripts/legacy/run_article_experiments.ps1`, configurations, source snapshots, model files, and per-run cards remain for provenance. Current experiments use the matched GELU/LSTM network and a separate audited simulator.
