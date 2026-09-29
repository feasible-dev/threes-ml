# Held-out model comparison

Each agent played 300 complete games on initial seeds 500000–500299. Training did not use these seeds.
Different policies can encounter different draws after making different moves.

| Run | Mean score | Median | Maximum | Tile >=192 | >=384 | >=3072 | >=6144 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-09-26_133136_stronger_updates_s42 | 477.6 | 408.0 | 2,781 | 1.0% | 0.0% | 0.0% | 0.0% |
| 2026-09-26_133136_stronger_updates_s43 | 560.6 | 454.5 | 2,988 | 1.0% | 0.0% | 0.0% | 0.0% |
| Heuristic: random | 283.4 | 216.0 | 2,337 | 0.3% | 0.0% | 0.0% | 0.0% |
| Heuristic: immediate merge | 540.4 | 417.0 | 2,652 | 1.3% | 0.0% | 0.0% | 0.0% |
| Heuristic: position | 976.4 | 930.0 | 3,426 | 9.3% | 0.0% | 0.0% | 0.0% |
| Heuristic: expected spawn | 1,358.7 | 1,125.0 | 3,681 | 25.3% | 0.0% | 0.0% | 0.0% |

Model per-game data: [prior_models.csv](prior_models.csv).
Heuristic per-game data: [heuristics.csv](heuristics.csv).

Interpretation and retention decisions remain pending review.
