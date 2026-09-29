---
status: decision
updated: 2026-09-27T11:22:00-04:00
scope: Completed masking by LR pilot and bounded extension decision
---

# September 27 fork-pilot decision

This decision replaces the **four-fork pilot as the next action** in [step 3 of the earlier September 27 decision](RESEARCH_DECISION_2026-09-27.md). The pilot and paired comparison are now complete. It does not change the earlier decision to preserve the stopped constant-LR run.

## Measured pilot evidence

The [completed comparison](../results/fork_batches/2026-09-27_100616_mask_lr/comparison.md) was measured on 2026-09-27 at 11:11:38 -04:00. Four forks restored the same curriculum/LR-decay endpoint at 100,007,936 attempted transitions, then each added 5,005,312 attempted transitions. Evaluation used 300 paired natural development games, seeds 820000–820299. The parent training trajectory and training seed 72 were shared; these are treatment forks, not independent random-initialization replications.

| Fixed-budget endpoint | Mean score | Median score | Reach 384 |
| --- | ---: | ---: | ---: |
| Parent | 2,502.7 | 1,482.0 | 12.3% |
| Unmasked, LR 0.00015 | 2,629.0 | 1,767.0 | 12.7% |
| Masked, LR 0.00015 | 3,132.9 | 2,637.0 | 15.3% |
| Unmasked, LR 0.00005 | 3,535.2 | 2,910.0 | 20.3% |
| Masked, LR 0.00005 | **3,926.7** | **2,989.5** | **24.3%** |

The masked low-LR arm beat the parent by 1,424.0 mean points on these paired games (95% paired bootstrap interval 931.0 to 1,920.7). Its difference from the **unmasked arm at the same low LR** was 391.5 points, with interval -135.5 to 925.1 and 145 wins versus 154 losses. It has the highest observed mean in this pilot, while a masking-specific advantage at LR 0.00005 remains uncertain. Reducing LR from 0.00015 to 0.00005 improved the mean within both mask conditions: 906.2 points unmasked and 793.9 points masked, with both paired intervals above zero. Periodic 100-game validation of the masked low-LR arm also rose to 4,212.9 at its final 105,013,248-transition endpoint. Those repeated 100 games are development monitoring, not fresh confirmation.

## Next allocation

Continue the **completed masked LR 0.00005 fork** for a bounded 20,000,000 additional attempted transitions, rounded to 20,004,864 complete rollout transitions. Its new endpoint will be 125,018,112 total transitions. Branch into a new run from its exact 105,013,248-transition checkpoint; preserve the 5M pilot as completed evidence. Keep its training mask and fixed LR 0.00005 unchanged. The isolated [extension launcher](REPRODUCTION_FORKS.md) is implemented and previewed against the actual checkpoint without compatibility issues. The user starts the long job in a visible terminal.

This fourfold larger allocation tests whether the promising policy keeps improving. It does **not** establish that masking caused the gain or justify an open-ended/100M allocation. After the extension completes, evaluate its fixed endpoint on a fresh development block, seeds 830000–830299, and compare it with its 5M parent. That comparison measures additional training, not the mask effect. If the extension remains promising, run an equal-budget unmasked low-LR control or an independent training-seed replication before attributing the gain to masking. Keep reserved final-test seeds 900000–909999 unused until model selection is frozen.
