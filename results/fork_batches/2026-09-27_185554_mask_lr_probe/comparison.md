# Masked learning-rate probe

Measured: 2026-09-27T19:29:11-04:00. Parent: `2026-09-27_125618_mask_lr_extension_masked_lr_00005` at 225,026,048 transitions. Each arm adds 5,005,312 attempted transitions to 230,031,360. Natural development seeds: 840000–840299.
Both arms restore the same parent weights, optimizer, simulator and RNG. This is one parent training trajectory, not an independent-seed replication or final test.

| Fixed-budget endpoint | Mean score | Median | Reach 384 | Mean valid moves |
| --- | ---: | ---: | ---: | ---: |
| parent_endpoint | 6945.7 | 7411.5 | 52.3% | 246.8 |
| masked_lr_00005 | 6819.9 | 7221.0 | 52.0% | 247.2 |
| masked_lr_00010 | 5798.9 | 4057.5 | 41.3% | 220.0 |

| Paired comparison (right minus left) | Mean difference | Bootstrap 95% interval | Wins / losses |
| --- | ---: | ---: | ---: |
| control_vs_parent | -125.8 | [-842.4, 583.7] | 148 / 152 |
| treatment_vs_parent | -1146.7 | [-1899.3, -385.7] | 125 / 175 |
| lr_00010_minus_00005 | -1020.9 | [-1751.1, -311.2] | 127 / 172 |

Per-game CSVs are in [evaluation/](evaluation/) and machine-readable results in [comparison.json](comparison.json).
