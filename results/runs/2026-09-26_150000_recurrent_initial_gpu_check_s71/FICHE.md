# 2026-09-26_150000_recurrent_initial_gpu_check_s71

> Generated: 2026-09-28T19:13:15-04:00. Training cutoff: 24,576 transitions.
> This card is a snapshot, not a live process monitor. Comparisons have their own checkpoint and evaluation provenance.

**Status:** complete. **Method:** recurrent PPO with a reconstruction of the published ONNX architecture.

Initialization: random; no imported weights. 

| Setting | Value |
| --- | --- |
| Network | 18 x 8 tile embedding + 17 x 8 position embedding; bag 4 -> 32; encoder 168 -> 2048 -> 1024 -> 1024; shared LSTM; two separate hidden heads; GELU |
| Parameters | 13,995,453 |
| Training run seed | 71 |
| Transitions | 24,576 |
| Environments / rollout / batch / epochs | 128 / 64 / 2048 / 4 |
| Initial learning rate / gamma / entropy coefficient | 0.0003 / 0.999 / 0.01 |
| Last recorded effective learning rate | not recorded |
| Shaping / curriculum probability | False / 0.0 |
| Engine revision / reward scale / timeout bootstrap | 1 / 1 / True |
| Learning-rate schedule (absolute transitions) | constant |
| Device | cuda |
| Latest / best validation mean | 804.375 / 804.375 |
| Independent comparison mean / games |  /  |
| Published reference mean, same rules and seeds |  |

Training game seeds are logged and lie above 2^63. Validation seeds lie below 2^63. Validation uses natural initial boards and legal greedy actions, with no curriculum.

The catalog's historical `holdout` column contains independent validation for this pipeline, not an untouched final test. [Latest comparison](evaluation/).

## Interpretation and retention

**Relevance:** medium. **Retention:** trim later.

Initial full-size CUDA integration and sequence-fusion throughput check only. Eight-game scores do not establish playing strength. This checkpoint format predates explicit recurrent-memory restoration and cannot be continued exactly. Superseded for continuation by the later recurrent_gpu_check run; retain engineering records and review bulky checkpoints during cleanup.

No cleanup happens automatically. Record research judgments in `metadata/review.json`.

## Files

- [Configuration](metadata/config.json), [method and limitations](metadata/method.json), [source provenance](metadata/provenance.json)
- [Metrics](data/metrics.csv), [training episodes and seeds](data/episodes.csv), [HTML curves](report/report.html)
- [Evaluation games](evaluation/), [checkpoints](models/checkpoints/)
