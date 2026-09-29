# Masking × learning-rate pilot comparison

Measured: 2026-09-27T11:11:38-04:00. Parent: `2026-09-26_162446_recurrent_v2_curriculum_decay_s72` at 100,007,936 transitions. 
Each fork adds 5,005,312 attempted transitions to 105,013,248. Natural-game development seeds: 820000–820299. 
These are paired development games and one parent training trajectory, not a final test or independent training replication.

| Agent | Mean score | Median | Reach 384 | Reach 3072 | Mean valid moves |
| --- | ---: | ---: | ---: | ---: | ---: |
| parent_endpoint | 2502.7 | 1482.0 | 12.3% | 0.0% | 136.3 |
| unmasked_lr_00015_endpoint | 2629.0 | 1767.0 | 12.7% | 0.0% | 141.8 |
| unmasked_lr_00015_selected_best | 2659.7 | 1615.5 | 10.3% | 0.0% | 144.4 |
| masked_lr_00015_endpoint | 3132.9 | 2637.0 | 15.3% | 0.0% | 159.5 |
| masked_lr_00015_selected_best | 3008.3 | 2433.0 | 15.3% | 0.0% | 153.8 |
| unmasked_lr_00005_endpoint | 3535.2 | 2910.0 | 20.3% | 0.0% | 168.9 |
| unmasked_lr_00005_selected_best | 3324.8 | 2818.5 | 16.3% | 0.0% | 160.9 |
| masked_lr_00005_endpoint | 3926.7 | 2989.5 | 24.3% | 0.0% | 176.7 |
| masked_lr_00005_selected_best | 3926.7 | 2989.5 | 24.3% | 0.0% | 176.7 |

| Paired comparison (right minus left) | Mean score difference | Bootstrap 95% interval | Wins / losses |
| --- | ---: | ---: | ---: |
| unmasked_lr_00015_endpoint_vs_parent | 126.3 | [-249.6, 498.9] | 164 / 135 |
| masked_lr_00015_endpoint_vs_parent | 630.2 | [238.7, 1023.5] | 184 / 116 |
| unmasked_lr_00005_endpoint_vs_parent | 1032.5 | [615.2, 1443.9] | 191 / 108 |
| masked_lr_00005_endpoint_vs_parent | 1424.0 | [931.0, 1920.7] | 192 / 108 |
| mask_effect_lr_00015 | 503.9 | [99.8, 903.5] | 168 / 132 |
| mask_effect_lr_00005 | 391.5 | [-135.5, 925.1] | 145 / 154 |
| lr_effect_unmasked | 906.2 | [489.0, 1317.8] | 183 / 117 |
| lr_effect_masked | 793.9 | [314.2, 1288.1] | 165 / 135 |

Per-game CSV files and machine-readable results are in [evaluation/](evaluation/) and [comparison.json](comparison.json).
