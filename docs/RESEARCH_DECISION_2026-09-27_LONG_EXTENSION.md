---
status: decision
updated: 2026-09-27T12:54:00-04:00
scope: Completed 20M masked fork extension and user-selected long allocation
---

# September 27 long-extension decision

This decision replaces the **20M masked low-LR extension as the next action** in the [fork-pilot decision](RESEARCH_DECISION_2026-09-27_FORK_PILOT.md). The user ran that bounded extension to completion and explicitly chose a longer run. The earlier 5M four-arm comparison remains the treatment evidence; neither extension is an independent training-seed replication.

## Measured evidence and limit

The completed [masked LR 0.00005 extension](../results/runs/2026-09-27_112611_mask_lr_extension_masked_lr_00005/) restored the 105,013,248-transition pilot endpoint and added 20,004,864 attempted transitions, ending at **125,018,112**. Its source/configuration and parent checkpoint are recorded in `metadata/`. Training continued with the legal-action mask on and fixed LR `0.00005`; the terminal log reports zero recent invalid attempts at the endpoint.

Periodic natural validation reused 100 development games, seeds 700000–700099. It reached a selected-best mean **5,205.09 at 120,766,464 transitions** (36% reached tile 384). The **fixed-budget endpoint mean was 4,067.43** (19% reached 384), compared with 4,212.87 at the 5M pilot endpoint on the same validation seeds. The selected-best peak supports further exploration, but the endpoint did not improve on this small repeated validation set. No fresh 300-game comparison of the 20M endpoint has been recorded at this decision cutoff. Do not use the selected-best peak as an unbiased estimate of the long-run outcome.

## User-selected next allocation

Branch the exact **125,018,112-transition endpoint** into a new, separately recorded run with **100,000,000 requested additional attempted transitions**, rounded to 100,007,936 complete rollouts. Its fixed endpoint will be **225,026,048** transitions. Keep the mask on, fixed LR `0.00005`, optimizer, simulator, recurrent state and RNG; the [extension launcher](REPRODUCTION_FORKS.md) has previewed this checkpoint with no source compatibility issues. The user launches the long job in a visible terminal. Preserve both completed parent runs and their evidence.

This larger allocation follows the user's explicit preference despite uncertain mask-specific benefit and a variable 20M validation curve. It tests continued training of the selected candidate; it cannot isolate masking from the lower LR. At completion, evaluate the fixed 125M and 225M endpoints on the same fresh development games, seeds 830000–830299, and report periodic selected-best models separately. Reserve final-test seeds 900000–909999 until model selection is frozen. A matched low-LR unmasked continuation or independent training-seed replication is still needed for a masking-causality claim.
