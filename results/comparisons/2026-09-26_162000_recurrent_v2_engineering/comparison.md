# Recurrent model comparison

2 games per policy; seeds 710110 through 710111.

All policies use the same native reproduction simulator, natural initial boards, legal greedy actions, and reset recurrent memory between games.

| Model | Mean | Median | >=3072 | >=6144 | 6144 95% Wilson CI | Truncated |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 2026-09-26_162000_recurrent_v2_gpu_check_s76 | 528.0 | 528.0 | 0.0% | 0.0% | 0.0%–65.8% | 0 |
| published_pretrained_reference | 213937.5 | 213937.5 | 100.0% | 0.0% | 0.0%–65.8% | 0 |

These are validation results once inspected for model selection. They are not directly interchangeable with scores from our historical game variant.

The published reference, when included, uses downloaded weights. Our training runs start randomly; importing reference weights is restricted to verification/evaluation.

Retention decisions remain pending human/agent review of learning curves and this comparison. No files are deleted automatically.
