# Matched LR-halving plateau probe

Both masked arms restored the same 330,039,296-transition checkpoint and each added 10,002,432 attempted transitions. Only the learning rate differs: 0.00010 versus 0.00005.
Natural development games: 860000–860299. All policies use legal greedy moves and fresh recurrent state per game.
These are fixed-policy game-level comparisons from one training lineage, not independent training-seed replications or final-test results.

| Policy | Mean score | Median | >=384 | >=768 | >=1536 | Mean valid moves |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| parent_endpoint | 6521.2 | 7161.0 | 51.7% | 3.0% | 0.0% | 234.4 |
| masked_lr_00010 | 7512.3 | 8004.0 | 61.3% | 4.7% | 0.0% | 254.9 |
| masked_lr_00005 | 8287.6 | 8058.0 | 62.7% | 7.7% | 0.0% | 269.1 |

| Paired score contrast (right minus left) | Mean difference | 95% bootstrap interval | Wins / losses |
| --- | ---: | ---: | ---: |
| control_minus_parent | 991.0 | [217.7, 1752.7] | 170 / 130 |
| treatment_minus_parent | 1766.3 | [908.0, 2638.7] | 172 / 128 |
| lr_00005_minus_00010 | 775.3 | [-66.4, 1596.0] | 165 / 135 |

Per-game scores are in [evaluation/](evaluation/); machine-readable results are in [comparison.json](comparison.json).
