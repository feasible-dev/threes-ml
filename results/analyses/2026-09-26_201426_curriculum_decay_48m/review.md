# Interim learning review: 48.61M transitions

Source: `results\runs\2026-09-26_162446_recurrent_v2_curriculum_decay_s72`. The CSV snapshot and summary JSON freeze this analysis while training continues.

## Observations

| Transition window (M) | Evaluations | Average validation mean | Average tile-384 rate |
| --- | ---: | ---: | ---: |
| 10-15.00 | 11 | 1368.8 | 1.1% |
| 15-20.00 | 10 | 1896.5 | 3.7% |
| 20-25.00 | 10 | 2089.3 | 4.3% |
| 25-30.00 | 10 | 2361.0 | 6.6% |
| 30-35.00 | 10 | 2441.3 | 7.4% |
| 35-40.00 | 10 | 2625.5 | 9.6% |
| 40-45.00 | 9 | 2576.4 | 9.3% |
| 45-48.61 | 8 | 2725.2 | 9.9% |

These window averages summarize repeated evaluation on the same 100 seeds; they are not independent replications.

The recent curve shows a noisy plateau rather than a sustained downward trend. Validation at 46.58M (3325.86, median 2976) nearly matched the selected 37.44M peak (3368.34, median 2914.5). The saved best model remains selected until a higher mean is measured. Selection of the best among many evaluations makes that peak optimistic.

## Interpretation and next decision

PPO changes both the policy and collected training positions. Optimizing its sampled surrogate objective does not guarantee a monotonically increasing game score. Value MSE uses changing bootstrapped targets; it is not the evaluation objective. At 38.45M, a lower value loss (3.077) coincided with 64.5% invalid actions and lower validation (2255). Entropy around 0.4 means a concentrated distribution, not a direct score of playing strength.

Training samples all four directions and penalizes invalid attempts by -0.05; evaluation chooses the highest-logit legal direction. Persistent invalid attempts consume substantial training budget and create a training/evaluation mismatch. This is a candidate limitation, not a proven cause of the plateau. The displayed curriculum fraction is the fraction of rollout actions in artificial-start games, not a performance score or the configured reset probability.

Recommendation: keep the current method fixed for now and reassess near 60M using smoothed natural-game scores, medians and tile rates. If it remains around 2500-2800 mean and 10% tile-384 reach, prioritize a separately tracked action-masking experiment and an isolated lower-learning-rate branch. Masking must be consistent in rollout sampling and PPO log-probability/entropy calculations; it is not an evaluation-only switch.

Before claiming regression or a stronger method, evaluate frozen candidate checkpoints on a larger common validation set. The 100 periodic seeds have already influenced selection. Reserved final-test seeds remain unused.

Sources: [PPO](https://arxiv.org/abs/1707.06347), [invalid action masking](https://arxiv.org/abs/2006.14171).
