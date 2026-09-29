---
status: decision
updated: 2026-09-27T18:37:00-04:00
scope: Completed 100M masked continuation and matched LR probe decision
---

# September 27 masked learning-rate probe decision

This decision replaces the **100M masked continuation as the next action** in the [long-extension decision](RESEARCH_DECISION_2026-09-27_LONG_EXTENSION.md). That run completed; the user now requests a short test of LR `0.00010` instead of `0.00005`.

## Measured evidence at the long-run endpoint

The [long masked continuation](../results/runs/2026-09-27_125618_mask_lr_extension_masked_lr_00005/) restored the fixed 125,018,112-transition endpoint, kept training masking on and LR fixed at `0.00005`, and completed its requested 100M additional budget at **225,026,048 attempted transitions**. Periodic natural validation on the same 100 development games, seeds 700000–700099, measured a **6,630.54 mean at the fixed endpoint** (42% reached tile 384). The selected-best checkpoint measured **7,393.23 at 214,417,408 transitions** (56% reached 384). These are repeated small development evaluations, not an unbiased final test; the peak was selected from many checkpoints. The run's frozen `metadata/` and checkpoint markers record its provenance.

## Planned short intervention

Branch the **exact 225,026,048-transition endpoint** into two separate masked recurrent-PPO forks. Each restores the same weights, optimizer moments, live simulator, recurrent memory and RNG, then adds **5,005,312 attempted transitions** (5M requested, rounded to complete 8,192-transition rollouts). The fixed endpoint for both is **230,031,360**. The control keeps fixed LR `0.00005`; the treatment changes only the optimizer/schedule LR to fixed `0.00010`. Both retain legal-action masking, architecture, rewards, curriculum and simulator settings. The [isolated LR-probe launcher](MASKED_LR_PROBE.md) is implemented, previewed against the actual parent and CPU tested. The user launches the visible training job.

After both fixed endpoints complete, evaluate the parent and both branches on **300 paired natural development games, seeds 840000–840299**, and report the paired LR contrast with uncertainty. Selected-best checkpoints remain separate from the fixed-budget comparison. This test estimates the local effect of doubling LR at the current checkpoint over 5M attempted transitions; it does not establish a globally optimal LR or an independent-seed replication. Keep final-test seeds 900000–909999 unused until model selection is frozen. Do not change or resume the completed 100M run to perform this intervention.
