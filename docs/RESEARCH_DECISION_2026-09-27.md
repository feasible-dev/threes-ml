---
status: decision
updated: 2026-09-27T10:02:25-04:00
scope: Interim LR comparison and early-stop decision for the constant-LR run
---

# September 27 interim learning-rate decision

This decision replaces **section 2's equal-100M-budget continuation and 1,000-game comparison** in the [September 26 decision](RESEARCH_DECISION_2026-09-26.md). The user chose to stop the second long run early and move to the next experiment. At this document update, the constant-LR process was still recorded as in progress; stopping it is a user action in its visible terminal. Preserve its checkpoints, metrics, evaluations and continuation history.

## Measured evidence at the decision cutoff

The two engine-v2 curriculum runs use training seed 72, the same architecture, simulator, reward, PPO settings, training mask state (off) and core source hashes. Their frozen configurations differ only in the LR milestone: the [decay run](../results/runs/2026-09-26_162446_recurrent_v2_curriculum_decay_s72/metadata/config.json) halves `0.0003` to `0.00015` after 5M transitions; the [constant-LR run](../results/runs/2026-09-26_162446_recurrent_v2_curriculum_s72/metadata/config.json) remains at `0.0003`. Both use legal-action filtering during natural-game evaluation. Neither trains with legal-action masking. These are separate random-weight training trajectories, not a pretrained branch or a different learning algorithm.

The constant-LR metrics reached **57,008,128 logged attempted transitions** at this cutoff. The completed decay run reached **100,007,936**. Compare them at approximately equal interim budgets, not at their unequal final counts:

| Validation window | Decay mean of periodic 100-game means | Constant-LR mean | Decay reach 384 | Constant-LR reach 384 |
| --- | ---: | ---: | ---: | ---: |
| 50–55M transitions, 10 evaluations each | 2,759.9 | 1,435.4 | 10.3% | 1.7% |
| 55–57M transitions, 4 evaluations each | 2,749.1 | 1,451.2 | 8.7% | 1.8% |

The nearest periodic development evaluations are [decay at 56,737,792](../results/runs/2026-09-26_162446_recurrent_v2_curriculum_decay_s72/evaluation/step_000056737792.csv) and [constant LR at 56,770,560](../results/runs/2026-09-26_162446_recurrent_v2_curriculum_s72/evaluation/step_000056770560.csv), only 32,768 transitions apart. On the same 100 natural-game seeds 700000–700099, decay averaged **2,876.76** versus **1,458.60** and reached tile 384 in **11/100** versus **2/100** games. The paired decay-minus-constant score difference was **+1,418.16**; decay won 72 games and lost 28. A 20,000-draw paired bootstrap percentile interval was **[926.28, 1,938.40]**, with bootstrap RNG seed 720927.

This is strong interim development evidence for prioritizing a different intervention. It is still one training seed and reused validation games, with correlated checkpoints. It does not prove a universal LR effect, and an early stop forfeits the planned equal-100M-budget comparison. No reserved final-test seeds were used.

## Planned intervention after the visible process stops

1. Stop the constant-LR continuation with Ctrl+C in its visible terminal. Keep the interrupted run as a truncated comparison condition; do not relabel its logged rows as a completed 100M endpoint. Inspect the final complete checkpoint marker separately from the last logged row.
2. Use the complete **100,007,936-transition decay endpoint** as the fixed parent for the already implemented [masking × LR forks](REPRODUCTION_FORKS.md). This replaces further long unmasked training. The four isolated arms each add 5,005,312 attempted transitions and cross training masking on/off with fixed effective LR `0.00015`/`0.00005`. Parent weights, optimizer, simulator and recurrent state are restored identically; each arm then collects fresh rollouts.
3. Compare fixed-budget fork endpoints on paired natural development games, seeds 820000–820299, and report selected-best checkpoints separately. Continue the remaining reward-component audit without changing the active parent's frozen source. Promote a promising arm only after examining paired uncertainty, valid moves, wall time and a further training-seed replication.

The [fork launcher preview](REPRODUCTION_FORKS.md) already resolves the decay endpoint and reports no source compatibility issues. The GPU fork treatments and throughput have not been measured. Launching them remains conditional on the constant-LR worker having stopped; the user starts long jobs in a visible terminal.
