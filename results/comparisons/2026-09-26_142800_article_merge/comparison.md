# Held-out model comparison

Each agent played 300 complete games on initial seeds 500000–500299. These seeds were reserved for evaluation, but individual training episode seeds were not logged or checked for overlap. Because this comparison informed the next research decision, this range is now validation data rather than an untouched final test set.
Different policies can encounter different draws after making different moves.

| Run | Mean score | Median | Maximum | Tile >=192 | >=384 | >=3072 | >=6144 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-09-26_142800_article_merge_64_s52 | 505.1 | 396.0 | 2,889 | 1.0% | 0.0% | 0.0% | 0.0% |
| 2026-09-26_142800_article_merge_64_s53 | 416.0 | 327.0 | 2,703 | 0.3% | 0.0% | 0.0% | 0.0% |
| 2026-09-26_142800_article_merge_256_s52 | 502.5 | 405.0 | 2,640 | 1.0% | 0.0% | 0.0% | 0.0% |
| 2026-09-26_142800_article_merge_256_s53 | 464.8 | 337.5 | 2,664 | 2.0% | 0.0% | 0.0% | 0.0% |
| Heuristic: random | 283.4 | 216.0 | 2,337 | 0.3% | 0.0% | 0.0% | 0.0% |
| Heuristic: immediate merge | 540.4 | 417.0 | 2,652 | 1.3% | 0.0% | 0.0% | 0.0% |
| Heuristic: position | 976.4 | 930.0 | 3,426 | 9.3% | 0.0% | 0.0% | 0.0% |
| Heuristic: expected spawn | 1,358.7 | 1,125.0 | 3,681 | 25.3% | 0.0% | 0.0% | 0.0% |

Model per-game data: [models.csv](models.csv).
Heuristic per-game data: [heuristics.csv](heuristics.csv).

## Interpretation

The four new policies averaged 416.0 to 505.1 points and none reached tile 384 in 300 games. The expected-spawn heuristic averaged 1,358.7 on the same initial seeds. The earlier stronger-updates PPO checkpoints, re-evaluated on these seeds, averaged 477.7 and 560.6; see [their per-game scores](prior_models.csv). Thus the new training package has not established a performance gain over our prior PPO, and it remains far below the heuristic.

Network width is the controlled difference within this batch. The 256-256 model scored 2.6 points less than 64-64 at training seed 52 and 48.8 points more at seed 53. That mixed result does not establish that width helps. Reward, activation, discount, and training budget also changed from the older PPO runs, so this batch cannot isolate the effect of the merge-depth reward.

The value network learned a partial prediction signal (final explained variance about 0.45 to 0.53), while periodic evaluation stayed around 400 to 550 points late in training. Mean policy entropy over the last 100 updates fell to about 0.03 to 0.22 nats from about 1.23 to 1.27 over the first 20 updates. This suggests increasingly deterministic policies; it does not by itself identify why score stalled. Value loss alone is not a performance measure. Further blind scaling of this configuration is not justified by these results. Keep the logs and comparison; review periodic checkpoints for trimming later. No files are deleted automatically.
