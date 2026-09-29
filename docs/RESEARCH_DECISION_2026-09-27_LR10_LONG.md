---
status: decision
updated: 2026-09-27T19:33:00-04:00
scope: Completed masked LR probe and user-selected long LR 0.00010 continuation
---

# September 27 long LR 0.00010 continuation decision

This decision replaces the **matched short LR probe as the next action** in the [earlier LR-probe decision](RESEARCH_DECISION_2026-09-27_LR_PROBE.md). The probe has completed. The user explicitly wants to continue the long-run lineage with LR `0.00010`, even though the short paired evidence favors retaining LR `0.00005`.

## Measured probe evidence

The [completed short comparison](../results/fork_batches/2026-09-27_185554_mask_lr_probe/comparison.md) was measured on 2026-09-27 at 19:29:11 -04:00. Both masked arms restored the exact 225,026,048-transition endpoint of the long LR `0.00005` run, then each added 5,005,312 attempted transitions. On 300 paired natural development games, seeds 840000–840299, the parent scored **6,945.7**, the LR `0.00005` control **6,819.9**, and the LR `0.00010` treatment **5,798.9** mean points. The treatment-minus-control paired difference was **−1,020.9 points**, with a 95% paired bootstrap interval of **−1,751.1 to −311.2** and 127 wins versus 172 losses. The treatment reached tile 384 in 41.3% of games versus 52.0% for the control. These are development results from one parent training trajectory, not a final test or independent training replication.

The short probe provides evidence **against** increasing LR at this checkpoint over the first 5M attempted transitions. It does not prove that a longer LR `0.00010` continuation cannot recover or improve later. A long continuation at LR `0.00010` alone will not measure the causal effect of LR over a long horizon without an equal-budget LR `0.00005` control.

## User-selected next allocation

Continue the completed **LR `0.00010` probe endpoint at 230,031,360 transitions**, preserving its already trained 5,005,312-transition segment. Branch it into a new run with **100,000,000 requested additional attempted transitions**, rounded to **100,007,936**, targeting **330,039,296 total transitions**. Keep legal-action masking, architecture, rewards, curriculum and simulator unchanged, and fixed LR `0.00010`. Restore weights, optimizer moments, live simulator, recurrent memory and RNG from the exact endpoint. The isolated [long-LR launcher](MASKED_LR10_LONG.md) is implemented, previewed without source compatibility issues and CPU tested. The user starts the long job in a visible terminal; neither completed parent is modified.

At completion, evaluate the fixed endpoint on fresh development games, seeds 850000–850299, and show selected-best checkpoints separately. Interpret the long-run result as performance of this chosen training path, not an isolated proof that increasing LR helped. Keep final-test seeds 900000–909999 unused until model selection is frozen.
