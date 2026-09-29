# Experiment catalog

> Generated: 2026-09-28T19:13:16-04:00. Status: generated snapshot; training cutoffs are in each row's Steps column.
> This is not a live process monitor. Per-run metrics may be newer than this table or its linked cards.
> Research direction: [current status](../docs/CURRENT_STATUS.md). Comparisons describe their own selected checkpoints and evaluation budgets.

Each run has a summary in its directory. Numbers come from CSV files; the review in
`metadata/review.json` is written after we examine the results.
“Pending review” means no research judgment is recorded; the run may still be in progress.

## Configurations

| Run | Local start | Code | Steps | Seed | Network | Reward | PPO rollout/batch/epochs | Initial rate / γ |
| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |
| [2026-09-25_183513_visual_smoke](runs/2026-09-25_183513_visual_smoke/FICHE.md) | 2026-09-25T18:35:13-04:00 | incomplete history | 1664 | 42 | 64-64 | log_score_gain_v1 | 128/64/2 | 0.0003 / 0.99 |
| [2026-09-25_183728_visual_demo_20k](runs/2026-09-25_183728_visual_demo_20k/FICHE.md) | 2026-09-25T18:37:28-04:00 | incomplete history | 20224 | 42 | 64-64 | log_score_gain_v1 | 128/64/2 | 0.0003 / 0.99 |
| [2026-09-26_115920_ppo_200k_s42](runs/2026-09-26_115920_ppo_200k_s42/FICHE.md) | 2026-09-26T11:59:20-04:00 | incomplete history | 200064 | 42 | 64-64 | log_score_gain_v1 | 128/64/2 | 0.0003 / 0.99 |
| [2026-09-26_120133_ppo_200k_s43](runs/2026-09-26_120133_ppo_200k_s43/FICHE.md) | 2026-09-26T12:01:33-04:00 | incomplete history | 200064 | 43 | 64-64 | log_score_gain_v1 | 128/64/2 | 0.0003 / 0.99 |
| [2026-09-26_133136_baseline_s42](runs/2026-09-26_133136_baseline_s42/FICHE.md) | 2026-09-26T13:31:37-04:00 | snapshot archived | 200064 | 42 | 64-64 | log_score_gain_v1 | 128/64/2 | 0.0003 / 0.99 |
| [2026-09-26_133136_baseline_s43](runs/2026-09-26_133136_baseline_s43/FICHE.md) | 2026-09-26T13:33:45-04:00 | snapshot archived | 200064 | 43 | 64-64 | log_score_gain_v1 | 128/64/2 | 0.0003 / 0.99 |
| [2026-09-26_133136_score_survival_s42](runs/2026-09-26_133136_score_survival_s42/FICHE.md) | 2026-09-26T13:40:42-04:00 | snapshot archived | 200064 | 42 | 64-64 | scaled_score_survival_v1 | 128/64/2 | 0.0003 / 0.99 |
| [2026-09-26_133136_score_survival_s43](runs/2026-09-26_133136_score_survival_s43/FICHE.md) | 2026-09-26T13:42:51-04:00 | snapshot archived | 200064 | 43 | 64-64 | scaled_score_survival_v1 | 128/64/2 | 0.0003 / 0.99 |
| [2026-09-26_133136_stronger_updates_s42](runs/2026-09-26_133136_stronger_updates_s42/FICHE.md) | 2026-09-26T13:35:55-04:00 | snapshot archived | 200192 | 42 | 64-64 | log_score_gain_v1 | 512/128/8 | 0.0003 / 0.99 |
| [2026-09-26_133136_stronger_updates_s43](runs/2026-09-26_133136_stronger_updates_s43/FICHE.md) | 2026-09-26T13:38:17-04:00 | snapshot archived | 200192 | 43 | 64-64 | log_score_gain_v1 | 512/128/8 | 0.0003 / 0.99 |
| [2026-09-26_142800_article_merge_256_s52](runs/2026-09-26_142800_article_merge_256_s52/FICHE.md) | 2026-09-26T14:34:04-04:00 | snapshot archived | 300032 | 52 | 256-256 | rank_weighted_merges_v1 | 512/128/8 | 0.0003 / 0.995 |
| [2026-09-26_142800_article_merge_256_s53](runs/2026-09-26_142800_article_merge_256_s53/FICHE.md) | 2026-09-26T14:37:26-04:00 | snapshot archived | 300032 | 53 | 256-256 | rank_weighted_merges_v1 | 512/128/8 | 0.0003 / 0.995 |
| [2026-09-26_142800_article_merge_64_s52](runs/2026-09-26_142800_article_merge_64_s52/FICHE.md) | 2026-09-26T14:28:01-04:00 | snapshot archived | 300032 | 52 | 64-64 | rank_weighted_merges_v1 | 512/128/8 | 0.0003 / 0.995 |
| [2026-09-26_142800_article_merge_64_s53](runs/2026-09-26_142800_article_merge_64_s53/FICHE.md) | 2026-09-26T14:31:02-04:00 | snapshot archived | 300032 | 53 | 64-64 | rank_weighted_merges_v1 | 512/128/8 | 0.0003 / 0.995 |
| [2026-09-26_150000_recurrent_initial_gpu_check_s71](runs/2026-09-26_150000_recurrent_initial_gpu_check_s71/FICHE.md) | 2026-09-26T15:13:25-04:00 | snapshot archived | 24576 | 71 | LSTM-1024 | published_merge | 64/2048/4 | 0.0003 / 0.999 |
| [2026-09-26_152600_recurrent_gpu_check_s71](runs/2026-09-26_152600_recurrent_gpu_check_s71/FICHE.md) | 2026-09-26T15:26:30-04:00 | snapshot archived | 32768 | 71 | LSTM-1024 | published_merge | 64/2048/4 | 0.0003 / 0.999 |
| [2026-09-26_153305_recurrent_base_s71](runs/2026-09-26_153305_recurrent_base_s71/FICHE.md) | 2026-09-26T15:33:06-04:00 | snapshot archived | 2007040 | 71 | LSTM-1024 | published_merge | 64/2048/4 | 0.0003 / 0.999 |
| [2026-09-26_153305_recurrent_shaped_s71](runs/2026-09-26_153305_recurrent_shaped_s71/FICHE.md) | 2026-09-26T15:39:52-04:00 | snapshot archived | 2007040 | 71 | LSTM-1024 | published_merge_shaped | 64/2048/4 | 0.0003 / 0.999 |
| [2026-09-26_162000_recurrent_v2_gpu_check_s76](runs/2026-09-26_162000_recurrent_v2_gpu_check_s76/FICHE.md) | 2026-09-26T16:18:13-04:00 | snapshot archived | 16384 | 76 | LSTM-1024 | published_merge_shaped | 64/2048/4 | 0.0003 / 0.999 |
| [2026-09-26_162446_recurrent_v2_control_s72](runs/2026-09-26_162446_recurrent_v2_control_s72/FICHE.md) | 2026-09-26T16:24:47-04:00 | snapshot archived | 10002432 | 72 | LSTM-1024 | published_merge_shaped | 64/2048/4 | 0.0003 / 0.999 |
| [2026-09-26_162446_recurrent_v2_curriculum_decay_s72](runs/2026-09-26_162446_recurrent_v2_curriculum_decay_s72/FICHE.md) | 2026-09-26T17:28:46-04:00 | snapshot archived | 100007936 | 72 | LSTM-1024 | published_merge_shaped | 64/2048/4 | 0.0003 / 0.999 |
| [2026-09-26_162446_recurrent_v2_curriculum_s72](runs/2026-09-26_162446_recurrent_v2_curriculum_s72/FICHE.md) | 2026-09-26T16:55:50-04:00 | snapshot archived | 60317696 | 72 | LSTM-1024 | published_merge_shaped | 64/2048/4 | 0.0003 / 0.999 |
| [2026-09-27_100616_mask_lr_masked_lr_00005](runs/2026-09-27_100616_mask_lr_masked_lr_00005/FICHE.md) | 2026-09-27T10:54:21-04:00 | snapshot archived | 105013248 | 72 | LSTM-1024 | published_merge_shaped | 64/2048/4 | 5e-05 / 0.999 |
| [2026-09-27_100616_mask_lr_masked_lr_00015](runs/2026-09-27_100616_mask_lr_masked_lr_00015/FICHE.md) | 2026-09-27T10:20:03-04:00 | snapshot archived | 105013248 | 72 | LSTM-1024 | published_merge_shaped | 64/2048/4 | 0.00015 / 0.999 |
| [2026-09-27_100616_mask_lr_unmasked_lr_00005](runs/2026-09-27_100616_mask_lr_unmasked_lr_00005/FICHE.md) | 2026-09-27T10:38:43-04:00 | snapshot archived | 105013248 | 72 | LSTM-1024 | published_merge_shaped | 64/2048/4 | 5e-05 / 0.999 |
| [2026-09-27_100616_mask_lr_unmasked_lr_00015](runs/2026-09-27_100616_mask_lr_unmasked_lr_00015/FICHE.md) | 2026-09-27T10:06:17-04:00 | snapshot archived | 105013248 | 72 | LSTM-1024 | published_merge_shaped | 64/2048/4 | 0.00015 / 0.999 |
| [2026-09-27_112611_mask_lr_extension_masked_lr_00005](runs/2026-09-27_112611_mask_lr_extension_masked_lr_00005/FICHE.md) | 2026-09-27T11:26:11-04:00 | snapshot archived | 125018112 | 72 | LSTM-1024 | published_merge_shaped | 64/2048/4 | 5e-05 / 0.999 |
| [2026-09-27_125618_mask_lr_extension_masked_lr_00005](runs/2026-09-27_125618_mask_lr_extension_masked_lr_00005/FICHE.md) | 2026-09-27T12:56:19-04:00 | snapshot archived | 225026048 | 72 | LSTM-1024 | published_merge_shaped | 64/2048/4 | 5e-05 / 0.999 |
| [2026-09-27_185554_mask_lr_probe_masked_lr_00005](runs/2026-09-27_185554_mask_lr_probe_masked_lr_00005/FICHE.md) | 2026-09-27T18:55:55-04:00 | snapshot archived | 230031360 | 72 | LSTM-1024 | published_merge_shaped | 64/2048/4 | 5e-05 / 0.999 |
| [2026-09-27_185554_mask_lr_probe_masked_lr_00010](runs/2026-09-27_185554_mask_lr_probe_masked_lr_00010/FICHE.md) | 2026-09-27T19:13:13-04:00 | snapshot archived | 230031360 | 72 | LSTM-1024 | published_merge_shaped | 64/2048/4 | 0.0001 / 0.999 |
| [2026-09-27_193437_masked_lr_00010_long_masked_lr_00010](runs/2026-09-27_193437_masked_lr_00010_long_masked_lr_00010/FICHE.md) | 2026-09-27T19:34:37-04:00 | snapshot archived | 330039296 | 72 | LSTM-1024 | published_merge_shaped | 64/2048/4 | 0.0001 / 0.999 |
| [2026-09-28_181141_plateau_lr_probe_masked_lr_00005](runs/2026-09-28_181141_plateau_lr_probe_masked_lr_00005/FICHE.md) | 2026-09-28T18:42:34-04:00 | snapshot archived | 340041728 | 72 | LSTM-1024 | published_merge_shaped | 64/2048/4 | 5e-05 / 0.999 |
| [2026-09-28_181141_plateau_lr_probe_masked_lr_00010](runs/2026-09-28_181141_plateau_lr_probe_masked_lr_00010/FICHE.md) | 2026-09-28T18:11:52-04:00 | snapshot archived | 340041728 | 72 | LSTM-1024 | published_merge_shaped | 64/2048/4 | 0.0001 / 0.999 |

## Results and relevance

| Run | Latest tracking | Best tracking | Comparison mean (games) | Reference | Ratio | Relevance | Retention |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| [2026-09-25_183513_visual_smoke](runs/2026-09-25_183513_visual_smoke/FICHE.md) | 207.0 | 334.0 | — | — | — | low | trim later |
| [2026-09-25_183728_visual_demo_20k](runs/2026-09-25_183728_visual_demo_20k/FICHE.md) | 304.2 | 486.3 | — | — | — | medium | trim later |
| [2026-09-26_115920_ppo_200k_s42](runs/2026-09-26_115920_ppo_200k_s42/FICHE.md) | 495.6 | 495.6 | 411.4 (300) | 1,378.5 | 29.8 % | medium | trim later |
| [2026-09-26_120133_ppo_200k_s43](runs/2026-09-26_120133_ppo_200k_s43/FICHE.md) | 523.7 | 558.2 | 532.5 (300) | 1,378.5 | 38.6 % | high | complete |
| [2026-09-26_133136_baseline_s42](runs/2026-09-26_133136_baseline_s42/FICHE.md) | 495.6 | 495.6 | 460.8 (300) | 1,414.2 | 32.6 % | medium | trim later |
| [2026-09-26_133136_baseline_s43](runs/2026-09-26_133136_baseline_s43/FICHE.md) | 523.7 | 558.2 | 517.6 (300) | 1,414.2 | 36.6 % | medium | trim later |
| [2026-09-26_133136_score_survival_s42](runs/2026-09-26_133136_score_survival_s42/FICHE.md) | 432.9 | 563.9 | 444.1 (300) | 1,414.2 | 31.4 % | medium | trim later |
| [2026-09-26_133136_score_survival_s43](runs/2026-09-26_133136_score_survival_s43/FICHE.md) | 389.0 | 435.6 | 367.4 (300) | 1,414.2 | 26.0 % | high | trim later |
| [2026-09-26_133136_stronger_updates_s42](runs/2026-09-26_133136_stronger_updates_s42/FICHE.md) | 489.6 | 639.1 | 496.6 (300) | 1,414.2 | 35.1 % | high | complete |
| [2026-09-26_133136_stronger_updates_s43](runs/2026-09-26_133136_stronger_updates_s43/FICHE.md) | 612.7 | 612.7 | 555.9 (300) | 1,414.2 | 39.3 % | high | complete |
| [2026-09-26_142800_article_merge_256_s52](runs/2026-09-26_142800_article_merge_256_s52/FICHE.md) | 510.4 | 510.4 | 502.5 (300) | 1,358.7 | 37.0 % | medium | trim later |
| [2026-09-26_142800_article_merge_256_s53](runs/2026-09-26_142800_article_merge_256_s53/FICHE.md) | 473.6 | 549.2 | 464.8 (300) | 1,358.7 | 34.2 % | medium | trim later |
| [2026-09-26_142800_article_merge_64_s52](runs/2026-09-26_142800_article_merge_64_s52/FICHE.md) | 517.5 | 540.8 | 505.1 (300) | 1,358.7 | 37.2 % | medium | trim later |
| [2026-09-26_142800_article_merge_64_s53](runs/2026-09-26_142800_article_merge_64_s53/FICHE.md) | 472.9 | 476.5 | 416.0 (300) | 1,358.7 | 30.6 % | medium | trim later |
| [2026-09-26_150000_recurrent_initial_gpu_check_s71](runs/2026-09-26_150000_recurrent_initial_gpu_check_s71/FICHE.md) | 804.4 | 804.4 | — | — | — | medium | trim later |
| [2026-09-26_152600_recurrent_gpu_check_s71](runs/2026-09-26_152600_recurrent_gpu_check_s71/FICHE.md) | 353.6 | 792.8 | 317.6 (8) | 277,453.1 | 0.1 % | high | complete |
| [2026-09-26_153305_recurrent_base_s71](runs/2026-09-26_153305_recurrent_base_s71/FICHE.md) | 410.2 | 548.6 | 459.4 (300) | 284,310.9 | 0.2 % | high | complete |
| [2026-09-26_153305_recurrent_shaped_s71](runs/2026-09-26_153305_recurrent_shaped_s71/FICHE.md) | 528.6 | 528.6 | 520.0 (300) | 284,310.9 | 0.2 % | high | complete |
| [2026-09-26_162000_recurrent_v2_gpu_check_s76](runs/2026-09-26_162000_recurrent_v2_gpu_check_s76/FICHE.md) | 297.0 | 382.5 | 528.0 (2) | 213,937.5 | 0.2 % | high (engineering) | keep reports and selected model; intermediate checkpoints eligible for later cleanup |
| [2026-09-26_162446_recurrent_v2_control_s72](runs/2026-09-26_162446_recurrent_v2_control_s72/FICHE.md) | 1,156.6 | 1,156.6 | 959.6 (300) | 271,896.8 | 0.4 % | pending review | undecided |
| [2026-09-26_162446_recurrent_v2_curriculum_decay_s72](runs/2026-09-26_162446_recurrent_v2_curriculum_decay_s72/FICHE.md) | 2,132.7 | 3,579.7 | — | — | — | pending review | undecided |
| [2026-09-26_162446_recurrent_v2_curriculum_s72](runs/2026-09-26_162446_recurrent_v2_curriculum_s72/FICHE.md) | 1,286.4 | 1,982.2 | — | — | — | pending review | undecided |
| [2026-09-27_100616_mask_lr_masked_lr_00005](runs/2026-09-27_100616_mask_lr_masked_lr_00005/FICHE.md) | 4,212.9 | 4,212.9 | — | — | — | pending review | undecided |
| [2026-09-27_100616_mask_lr_masked_lr_00015](runs/2026-09-27_100616_mask_lr_masked_lr_00015/FICHE.md) | 3,333.2 | 3,560.2 | — | — | — | pending review | undecided |
| [2026-09-27_100616_mask_lr_unmasked_lr_00005](runs/2026-09-27_100616_mask_lr_unmasked_lr_00005/FICHE.md) | 3,225.2 | 3,618.8 | — | — | — | pending review | undecided |
| [2026-09-27_100616_mask_lr_unmasked_lr_00015](runs/2026-09-27_100616_mask_lr_unmasked_lr_00015/FICHE.md) | 2,946.9 | 3,106.1 | — | — | — | pending review | undecided |
| [2026-09-27_112611_mask_lr_extension_masked_lr_00005](runs/2026-09-27_112611_mask_lr_extension_masked_lr_00005/FICHE.md) | 4,067.4 | 5,205.1 | — | — | — | pending review | undecided |
| [2026-09-27_125618_mask_lr_extension_masked_lr_00005](runs/2026-09-27_125618_mask_lr_extension_masked_lr_00005/FICHE.md) | 6,630.5 | 7,393.2 | — | — | — | pending review | undecided |
| [2026-09-27_185554_mask_lr_probe_masked_lr_00005](runs/2026-09-27_185554_mask_lr_probe_masked_lr_00005/FICHE.md) | 7,369.1 | 7,817.9 | — | — | — | pending review | undecided |
| [2026-09-27_185554_mask_lr_probe_masked_lr_00010](runs/2026-09-27_185554_mask_lr_probe_masked_lr_00010/FICHE.md) | 6,164.9 | 7,107.7 | — | — | — | pending review | undecided |
| [2026-09-27_193437_masked_lr_00010_long_masked_lr_00010](runs/2026-09-27_193437_masked_lr_00010_long_masked_lr_00010/FICHE.md) | 6,298.1 | 7,968.9 | — | — | — | pending review | undecided |
| [2026-09-28_181141_plateau_lr_probe_masked_lr_00005](runs/2026-09-28_181141_plateau_lr_probe_masked_lr_00005/FICHE.md) | 8,359.6 | 8,359.6 | — | — | — | pending review | undecided |
| [2026-09-28_181141_plateau_lr_probe_masked_lr_00010](runs/2026-09-28_181141_plateau_lr_probe_masked_lr_00010/FICHE.md) | 7,412.2 | 7,752.4 | — | — | — | pending review | undecided |

Tracking and comparison columns use different seeds and may describe different checkpoints.
The historical CSV field name `holdout_mean` does not make these comparisons untouched final tests.
The rate column is the configured initial rate; the effective scheduled rate is in each run's metrics.
The same initial seed does not force two agents to see the same draws after they choose different moves.

## Read and sort

- Open a run summary for its parameters, question, finding, and files.
- Open [experiences.csv](experiences.csv) in a spreadsheet to filter or sort runs.
- `metadata/config.json` freezes chosen parameters; `metadata/method.json` describes the model and PPO.
- `metadata/provenance.json` records the commit or source hashes and code snapshot.
- `data/metrics.csv` has every update; `report/report.html` shows the charts.
- `metadata/review.json` holds our assessment. The catalog never deletes files.

## Relevance scale

- **high**: a result or diagnosis worth revisiting.
- **medium**: a useful comparison or validation, partly redundant.
- **low**: a technical check or result without a reliable conclusion.
Performance and relevance are separate: a weak run may reveal an important problem.

## Suggested retention

- **complete**: keep the model and detailed records for later work.
- **trim later**: keep results and the summary; review checkpoints during cleanup.
- **undecided**: wait until the run is interpreted.
These are recommendations; no files are deleted automatically.

## Early runs outside the table

The [first 20,096-transition PPO run](runs/2026-09-25_182257_ppo_20k_initial/FICHE.md) and the [PPO smoke test](runs/2026-09-25_182225_ppo_smoke/FICHE.md) predate automatic run tracking.
The first scored 344.9 points over 100 games. That historical measurement does not use
the held-out seeds from the runs in this table.
