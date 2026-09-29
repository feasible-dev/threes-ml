# 2026-09-26_162446_recurrent_v2_control_s72

> Generated: 2026-09-28T19:13:15-04:00. Training cutoff: 10,002,432 transitions.
> This card is a snapshot, not a live process monitor. Comparisons have their own checkpoint and evaluation provenance.

**Status:** complete. **Method:** recurrent PPO with a reconstruction of the published ONNX architecture.

Initialization: random; no imported weights. 

| Setting | Value |
| --- | --- |
| Network | 18 x 8 tile embedding + 17 x 8 position embedding; bag 4 -> 32; encoder 168 -> 2048 -> 1024 -> 1024; shared LSTM; two separate hidden heads; GELU |
| Parameters | 13,995,453 |
| Training run seed | 72 |
| Transitions | 10,002,432 |
| Environments / rollout / batch / epochs | 128 / 64 / 2048 / 4 |
| Initial learning rate / gamma / entropy coefficient | 0.0003 / 0.999 / 0.01 |
| Last recorded effective learning rate | 0.0003 |
| Shaping / curriculum probability | True / 0 |
| Engine revision / reward scale / timeout bootstrap | 2 / 1 / False |
| Learning-rate schedule (absolute transitions) | {'milestones': [], 'factor': 0.5} |
| Device | cuda |
| Latest / best validation mean | 1156.56 / 1156.56 |
| Independent comparison mean / games | 959.6 / 300 |
| Published reference mean, same rules and seeds | 271896.76 |

Training game seeds are logged and lie above 2^63. Validation seeds lie below 2^63. Validation uses natural initial boards and legal greedy actions, with no curriculum.

The catalog's historical `holdout` column contains independent validation for this pipeline, not an untouched final test. [Latest comparison](../../comparisons/2026-09-26_162446_recurrent_v2/comparison.md).

## Interpretation and retention

**Relevance:** pending review. **Retention:** undecided.

Review learning curves and independent evaluation.

No cleanup happens automatically. Record research judgments in `metadata/review.json`.

## Files

- [Configuration](metadata/config.json), [method and limitations](metadata/method.json), [source provenance](metadata/provenance.json)
- [Metrics](data/metrics.csv), [training episodes and seeds](data/episodes.csv), [HTML curves](report/report.html)
- [Evaluation games](evaluation/), [checkpoints](models/checkpoints/)
