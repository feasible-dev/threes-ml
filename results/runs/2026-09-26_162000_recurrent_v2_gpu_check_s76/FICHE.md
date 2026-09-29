# 2026-09-26_162000_recurrent_v2_gpu_check_s76

> Generated: 2026-09-28T19:13:15-04:00. Training cutoff: 16,384 transitions.
> This card is a snapshot, not a live process monitor. Comparisons have their own checkpoint and evaluation provenance.

**Status:** complete. **Method:** recurrent PPO with a reconstruction of the published ONNX architecture.

Initialization: random; no imported weights. 

| Setting | Value |
| --- | --- |
| Network | 18 x 8 tile embedding + 17 x 8 position embedding; bag 4 -> 32; encoder 168 -> 2048 -> 1024 -> 1024; shared LSTM; two separate hidden heads; GELU |
| Parameters | 13,995,453 |
| Training run seed | 76 |
| Transitions | 16,384 |
| Environments / rollout / batch / epochs | 128 / 64 / 2048 / 4 |
| Initial learning rate / gamma / entropy coefficient | 0.0003 / 0.999 / 0.01 |
| Last recorded effective learning rate | 0.00015 |
| Shaping / curriculum probability | True / 0.5 |
| Engine revision / reward scale / timeout bootstrap | 2 / 1 / False |
| Learning-rate schedule (absolute transitions) | {'milestones': [4096], 'factor': 0.5} |
| Device | cuda |
| Latest / best validation mean | 297.0 / 382.5 |
| Independent comparison mean / games | 528.0 / 2 |
| Published reference mean, same rules and seeds | 213937.5 |

Training game seeds are logged and lie above 2^63. Validation seeds lie below 2^63. Validation uses natural initial boards and legal greedy actions, with no curriculum.

The catalog's historical `holdout` column contains independent validation for this pipeline, not an untouched final test. [Latest comparison](../../comparisons/2026-09-26_162000_recurrent_v2_engineering/comparison.md).

## Interpretation and retention

**Relevance:** high (engineering). **Retention:** keep reports and selected model; intermediate checkpoints eligible for later cleanup.

Engine-v2 CUDA smoke test: full 13,995,453-parameter network, curriculum, LR milestone at 4096, 8192 transitions then complete-state resume to 16384. LR remained 0.00015; curriculum appeared in 9% then 42% of actions. Two-game comparison verifies the reporting path only; it provides no playing-strength evidence. The CPU resume test checks exact next-update equality across an LR milestone.

No cleanup happens automatically. Record research judgments in `metadata/review.json`.

## Files

- [Configuration](metadata/config.json), [method and limitations](metadata/method.json), [source provenance](metadata/provenance.json)
- [Metrics](data/metrics.csv), [training episodes and seeds](data/episodes.csv), [HTML curves](report/report.html)
- [Evaluation games](evaluation/), [checkpoints](models/checkpoints/)
