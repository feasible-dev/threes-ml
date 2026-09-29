# 2026-09-26_153305_recurrent_base_s71

> Generated: 2026-09-28T19:13:15-04:00. Training cutoff: 2,007,040 transitions.
> This card is a snapshot, not a live process monitor. Comparisons have their own checkpoint and evaluation provenance.

**Status:** complete. **Method:** recurrent PPO with a reconstruction of the published ONNX architecture.

Initialization: random; no imported weights. 

| Setting | Value |
| --- | --- |
| Network | 18 x 8 tile embedding + 17 x 8 position embedding; bag 4 -> 32; encoder 168 -> 2048 -> 1024 -> 1024; shared LSTM; two separate hidden heads; GELU |
| Parameters | 13,995,453 |
| Training run seed | 71 |
| Transitions | 2,007,040 |
| Environments / rollout / batch / epochs | 128 / 64 / 2048 / 4 |
| Initial learning rate / gamma / entropy coefficient | 0.0003 / 0.999 / 0.01 |
| Last recorded effective learning rate | not recorded |
| Shaping / curriculum probability | False / 0.0 |
| Engine revision / reward scale / timeout bootstrap | 1 / 1 / True |
| Learning-rate schedule (absolute transitions) | constant |
| Device | cuda |
| Latest / best validation mean | 410.16 / 548.61 |
| Independent comparison mean / games | 459.41 / 300 |
| Published reference mean, same rules and seeds | 284310.88 |

Training game seeds are logged and lie above 2^63. Validation seeds lie below 2^63. Validation uses natural initial boards and legal greedy actions, with no curriculum.

The catalog's historical `holdout` column contains independent validation for this pipeline, not an untouched final test. [Latest comparison](../../comparisons/2026-09-26_153305_recurrent/comparison.md).

## Interpretation and retention

**Relevance:** high. **Retention:** complete.

A necessary from-scratch control for the published-architecture pipeline. After 2,007,040 transitions, the selected checkpoint averaged 459.41 on 300 shared validation games, reached tile 192 in 5/300 and tile 384 in 0/300. Periodic 100-game validation peaked at 548.61 at 1,531,904 transitions and ended at 410.16; the cumulative training invalid-action rate was 25.10%. The value explained variance was about 0.66 at the end, so the value objective learned some signal while playing strength remained far below the published reference. Keep model, source, and metrics as the paired baseline; do not extrapolate this curve to the publication result.

No cleanup happens automatically. Record research judgments in `metadata/review.json`.

## Files

- [Configuration](metadata/config.json), [method and limitations](metadata/method.json), [source provenance](metadata/provenance.json)
- [Metrics](data/metrics.csv), [training episodes and seeds](data/episodes.csv), [HTML curves](report/report.html)
- [Evaluation games](evaluation/), [checkpoints](models/checkpoints/)
