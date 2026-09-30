# 2026-09-29_210238_fresh_corner_initial_lr30_s74

> Generated: 2026-09-30T06:43:50-04:00. Training cutoff: 100,007,936 transitions.
> This card is a snapshot, not a live process monitor. Comparisons have their own checkpoint and evaluation provenance.

**Status:** complete. **Method:** recurrent PPO with a reconstruction of the published ONNX architecture.

Initialization: random; no imported weights. 

| Setting | Value |
| --- | --- |
| Network | 18 x 8 tile embedding + 17 x 8 position embedding; bag 4 -> 32; encoder 168 -> 2048 -> 1024 -> 1024; shared LSTM; two separate hidden heads; GELU |
| Parameters | 13,995,453 |
| Training run seed | 74 |
| Transitions | 100,007,936 |
| Environments / rollout / batch / epochs | 128 / 64 / 2048 / 4 |
| Initial learning rate / gamma / entropy coefficient | 0.0003 / 0.999 / 0.01 |
| Last recorded effective learning rate | 0.00015 |
| Shaping / curriculum probability | True / 0.5 |
| Engine revision / reward scale / timeout bootstrap | 2 / 1 / False |
| Learning-rate schedule (absolute transitions) | {'milestones': [5000000], 'factor': 0.5} |
| Device | cuda |
| Latest / best validation mean | 3951.33 / 4201.35 |
| Independent comparison mean / games |  /  |
| Published reference mean, same rules and seeds |  |

Training game seeds are logged and lie above 2^63. Validation seeds lie below 2^63. Validation uses natural initial boards and legal greedy actions, with no curriculum.

The catalog's historical `holdout` column contains independent validation for this pipeline, not an untouched final test. [Latest comparison](evaluation/).

## Interpretation and retention

**Relevance:** pending review. **Retention:** undecided.

Review learning curves and independent evaluation.

No cleanup happens automatically. Record research judgments in `metadata/review.json`.

## Files

- [Configuration](metadata/config.json), [method and limitations](metadata/method.json), [source provenance](metadata/provenance.json)
- [Metrics](data/metrics.csv), [training episodes and seeds](data/episodes.csv), [HTML curves](report/report.html)
- [Evaluation games](evaluation/), [checkpoints](models/checkpoints/)
