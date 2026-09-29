# Held-out model comparison

Each agent played 300 complete games on initial seeds 300000–300299. Training did not use these seeds.
Different policies can encounter different draws after making different moves.

| Run | Mean score | Median | Maximum | Tile ≥192 | Tile ≥384 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026-09-26_133136_baseline_s42 | 460.8 | 333.0 | 3,207 | 1.3% | 0.0% |
| 2026-09-26_133136_baseline_s43 | 517.6 | 409.5 | 3,015 | 2.0% | 0.0% |
| 2026-09-26_133136_stronger_updates_s42 | 496.6 | 381.0 | 2,808 | 1.3% | 0.0% |
| 2026-09-26_133136_stronger_updates_s43 | 555.9 | 441.0 | 2,829 | 2.0% | 0.0% |
| 2026-09-26_133136_score_survival_s42 | 444.1 | 328.5 | 3,396 | 0.3% | 0.0% |
| 2026-09-26_133136_score_survival_s43 | 367.4 | 279.0 | 2,586 | 0.3% | 0.0% |
| Heuristic: random | 292.7 | 222.0 | 2,697 | 0.3% | 0.0% |
| Heuristic: immediate merge | 530.2 | 420.0 | 2,769 | 0.7% | 0.0% |
| Heuristic: position | 967.1 | 912.0 | 3,510 | 9.3% | 0.0% |
| Heuristic: expected spawn | 1,414.2 | 1,056.0 | 7,704 | 22.3% | 2.0% |

Model per-game data: [holdout_models.csv](holdout_models.csv).
Heuristic per-game data: [holdout_heuristics.csv](holdout_heuristics.csv).

## Interpretation — September 26, 2026

Across the two training seeds, the baseline models average **489.2**, stronger updates **526.2**, and score-plus-survival reward **405.8** over their 600 held-out games per condition. The stronger-update models improve over their seed-matched controls by **35.8** and **38.3** points. A paired bootstrap over the 300 initial seeds gives illustrative 95% intervals of **−32 to 105** and **−35 to 113** points, respectively. Both intervals include zero; two training seeds do not establish a reliable improvement. The tile-192 reach rates remain **1.3%** and **2.0%**, unchanged from their controls. No neural model reached tile 384. The position and expected-spawn heuristics still have a large lead.

The changed reward produces **−16.7** and **−150.2** points relative to the respective controls and reduces the tile-192 rate to **0.3%** for both seeds. It is a useful negative result. Its value-network explained variance rises to about **0.25** late in training, whereas it stays near zero for the other conditions. Learning to predict this reward somewhat better did not make the policy better at the official score.

The baseline PPO logs contain 1,563 updates per run; the larger-rollout condition has 391 updates but more optimization passes per rollout. Late approximate KL is around **0.00002–0.00003** for baseline and **0.001–0.002** for stronger updates, consistent with larger policy changes. Value loss does not decrease steadily, and its absolute scale differs between reward formulas. PPO gathers new, changing data as the policy learns, so this loss is not a fixed supervised-learning objective. Official score and tile reach on separate games remain the decision metrics.

**Next direction:** preserve the stronger-update models and their best periodic checkpoints. Select a checkpoint using only periodic evaluation, then test it on fresh seeds. For the next substantive learning method, investigate an afterstate value learner with a small expectimax lookahead, following the Threes! literature, and benchmark simulation throughput before any much longer PPO training. The current 200,000-transition runs are informative screening experiments, not a training budget expected to reach tile 6,144.
