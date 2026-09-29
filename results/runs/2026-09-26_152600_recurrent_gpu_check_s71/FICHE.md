# 2026-09-26_152600_recurrent_gpu_check_s71

> Generated: 2026-09-28T19:13:15-04:00. Training cutoff: 32,768 transitions.
> This card is a snapshot, not a live process monitor. Comparisons have their own checkpoint and evaluation provenance.

**Status:** complete. **Method:** recurrent PPO with a reconstruction of the published ONNX architecture.

Initialization: random; no imported weights. 

| Setting | Value |
| --- | --- |
| Network | 18 x 8 tile embedding + 17 x 8 position embedding; bag 4 -> 32; encoder 168 -> 2048 -> 1024 -> 1024; shared LSTM; two separate hidden heads; GELU |
| Parameters | 13,995,453 |
| Training run seed | 71 |
| Transitions | 32,768 |
| Environments / rollout / batch / epochs | 128 / 64 / 2048 / 4 |
| Initial learning rate / gamma / entropy coefficient | 0.0003 / 0.999 / 0.01 |
| Last recorded effective learning rate | not recorded |
| Shaping / curriculum probability | False / 0.0 |
| Engine revision / reward scale / timeout bootstrap | 1 / 1 / True |
| Learning-rate schedule (absolute transitions) | constant |
| Device | cuda |
| Latest / best validation mean | 353.625 / 792.75 |
| Independent comparison mean / games | 317.625 / 8 |
| Published reference mean, same rules and seeds | 277453.125 |

Training game seeds are logged and lie above 2^63. Validation seeds lie below 2^63. Validation uses natural initial boards and legal greedy actions, with no curriculum.

The catalog's historical `holdout` column contains independent validation for this pipeline, not an untouched final test. [Latest comparison](../../comparisons/2026-09-26_153200_recurrent_integration/comparison.md).

## Interpretation and retention

**Relevance:** high. **Retention:** complete.

Engineering validation of the full 13,995,453-parameter model on RTX 3060: 16,384 transitions followed by two 8,192-transition checkpoint continuations, finite optimization metrics, about 3,400-3,650 transitions/s including tracking. Separate CPU tests verify exact next-update equality after restoration. Eight-game validation is insufficient for a policy-performance conclusion. Keep as a known working pipeline checkpoint; proceed to the user-launched controlled pilots.

No cleanup happens automatically. Record research judgments in `metadata/review.json`.

## Files

- [Configuration](metadata/config.json), [method and limitations](metadata/method.json), [source provenance](metadata/provenance.json)
- [Metrics](data/metrics.csv), [training episodes and seeds](data/episodes.csv), [HTML curves](report/report.html)
- [Evaluation games](evaluation/), [checkpoints](models/checkpoints/)
