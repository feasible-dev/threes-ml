---
status: historical
updated: 2026-09-26T21:29:45-04:00
scope: Early project milestones
---

# Early project milestones

> Historical stage only. Earlier recommendations and agent instructions are superseded by [current status](../CURRENT_STATUS.md).

Historical record through the initial recurrent pilot and v2 design on September 26, 2026. Repeated descriptions of the then-current plan have been removed; see [current status](../CURRENT_STATUS.md) for accepted work.

## Completed milestones

| Period | Result | Evidence |
| --- | --- | --- |
| September 25 | Playable terminal and GUI game, seeded rules, random/merge/position/spawn heuristics | [Rules audit](LEGACY_RULES_AUDIT.md), [reference results](../../results/references) |
| September 25–26 | Historical 35-input MaskablePPO training, reports, checkpointing, and experiment catalog | [Model guide](MLP_MODEL.md), [catalog](../../results/EXPERIENCES.md) |
| September 26 | Six-run reward/update comparison: stronger updates gave modest score gains; score-plus-survival regressed | [Comparison](../../results/comparisons/2026-09-26_133136_ppo_reward_updates/comparison.md) |
| September 26 | Four article-inspired MLP runs on seeds 52 and 53 scored roughly 416–505 mean versus 1,359 for the spawn heuristic on the same 300 initial seeds | [Comparison](../../results/comparisons/2026-09-26_142800_article_merge/comparison.md) |
| September 26 | Downloaded agent integrated as a separate reference; local historical-game differences audited | [Compatibility audit](LEGACY_MODEL_COMPATIBILITY.md) |
| September 26 | Reconstructed ONNX graph, native simulator, CUDA training, and complete recurrent checkpoint continuation; engineering tests passed | [Reproduction guide](../REPRODUCTION.md), [validation artifacts](../../results/references) |

The earlier MLP results provided baselines and showed that simply making a feed-forward network wider did not close the gap. Their experiment proposals are archived in [NEXT_EXPERIMENTS.md](PPO_REWARD_EXPERIMENTS.md) and [ARTICLE_INSPIRED_EXPERIMENTS.md](ARTICLE_INSPIRED_EXPERIMENTS.md). The [initial plan](INITIAL_PROPOSAL.md) and [code audit](MLP_CODE_AUDIT.md) remain historical context. The [catalog](../../results/EXPERIENCES.md) records measured values separately from relevance and retention judgments. No runs are deleted automatically.
