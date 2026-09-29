# Post-run plateau assessment

The selected-best checkpoint was chosen on the reused 100-game periodic set; it is assessed separately from the fixed endpoint here.
Natural development games: 850000–850299. All policies use legal greedy moves and fresh recurrent state per game.
These are fixed-policy game-level comparisons from one training lineage, not independent training-seed replications or final-test results.

| Policy | Mean score | Median | >=384 | >=768 | >=1536 | Mean valid moves |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| probe_parent_230m | 5972.0 | 4342.5 | 43.3% | 2.3% | 0.0% | 227.1 |
| fixed_endpoint_330m | 6502.4 | 7173.0 | 52.3% | 2.3% | 0.0% | 236.7 |
| selected_best_307m | 6821.4 | 7381.5 | 52.7% | 3.7% | 0.0% | 242.0 |

| Paired score contrast (right minus left) | Mean difference | 95% bootstrap interval | Wins / losses |
| --- | ---: | ---: | ---: |
| 330m_minus_230m | 530.4 | [-142.2, 1207.5] | 166 / 134 |
| selected_best_minus_330m | 319.0 | [-373.5, 1038.2] | 154 / 146 |

Per-game scores are in [evaluation/](evaluation/); machine-readable results are in [comparison.json](comparison.json).
