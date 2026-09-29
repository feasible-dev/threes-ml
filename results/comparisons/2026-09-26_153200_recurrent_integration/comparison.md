# Recurrent model comparison

8 games per policy; seeds 710000 through 710007.

All policies use the same native reproduction simulator, natural initial boards, legal greedy actions, and reset recurrent memory between games.

| Model | Mean | Median | >=3072 | >=6144 | 6144 95% Wilson CI | Truncated |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 2026-09-26_152600_recurrent_gpu_check_s71 | 317.6 | 361.5 | 0.0% | 0.0% | 0.0%–32.4% | 0 |
| published_pretrained_reference | 277453.1 | 207309.0 | 75.0% | 25.0% | 7.1%–59.1% | 0 |

These are validation results once inspected for model selection. They are not directly interchangeable with scores from our historical game variant.

The published reference, when included, uses downloaded weights. Our training runs start randomly; importing reference weights is restricted to verification/evaluation.

Retention decisions remain pending human/agent review of learning curves and this comparison. No files are deleted automatically.
