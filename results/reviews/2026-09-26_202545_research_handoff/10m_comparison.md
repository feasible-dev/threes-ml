# Recurrent model comparison

300 games per policy; seeds 800000 through 800299.

All policies use the same native reproduction simulator, natural initial boards, legal greedy actions, and reset recurrent memory between games.

| Model | Mean | Median | >=3072 | >=6144 | 6144 95% Wilson CI | Truncated |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 2026-09-26_162446_recurrent_v2_control_s72 | 959.6 | 879.0 | 0.0% | 0.0% | 0.0%–1.3% | 0 |
| 2026-09-26_162446_recurrent_v2_curriculum_s72 | 993.3 | 876.0 | 0.0% | 0.0% | 0.0%–1.3% | 0 |
| 2026-09-26_162446_recurrent_v2_curriculum_decay_s72 | 1219.7 | 1005.0 | 0.0% | 0.0% | 0.0%–1.3% | 0 |
| published_pretrained_reference | 271896.8 | 242116.5 | 76.3% | 18.3% | 14.4%–23.1% | 0 |

These are validation results once inspected for model selection. They are not directly interchangeable with scores from our historical game variant.

The published reference, when included, uses downloaded weights. Our training runs start randomly; importing reference weights is restricted to verification/evaluation.

Retention decisions remain pending human/agent review of learning curves and this comparison. No files are deleted automatically.
