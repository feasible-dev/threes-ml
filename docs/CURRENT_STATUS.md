---
status: current
updated: 2026-09-28T20:44:57-04:00
scope: Current research status
---

# Current research status

This page owns the accepted research direction. The [documentation index](README.md) explains how to read other documents when their reasoning differs.

## Accepted direction

Decision in force: [September 28 long low-LR decision](RESEARCH_DECISION_2026-09-28_LOW_LR_LONG.md). It replaces the completed matched plateau LR pilot as the next action in the [earlier plateau decision](RESEARCH_DECISION_2026-09-28_PLATEAU.md). Earlier decisions remain dated reasoning; the equal-100M-budget second curriculum continuation is still superseded.

1. The curriculum/LR-decay run completed at 100,007,936 transitions. The constant-LR curriculum run was stopped early after the measured ~57M interim comparison and remains an interrupted condition. Do not run the superseded equal-budget continuation or its 1,000-game comparison.
2. Continue the recurrent PPO audit. Read-only checkpoint checks measured probability reconstruction, state/reset/padding handling, invalid probability mass and streaks, shared actor/critic gradients, and timeouts. Focused CPU checks now cover masked sampling, replay, padding, one-legal-action states and a real PPO update. Full training reward-component accounting remains.
3. The [matched plateau LR pilot](../results/fork_batches/2026-09-28_181141_plateau_lr_probe/comparison.md) completed from the 330,039,296-transition parent. At the matched 340,041,728-transition endpoints on 300 paired natural development games, LR `0.00005` averaged **8,287.6** versus **7,512.3** for LR `0.00010`; the paired +775.3-point interval was −66.4 to +1,596.0. The user launched a new, isolated long continuation of the complete LR `0.00005` endpoint, targeting **440,049,664** transitions. Its run status was `in progress` when checked on 2026-09-28 at 20:44:57 -04:00; that recorded status is a snapshot, not proof of a live process. The user runs long jobs in a visible terminal.
4. Keep the high-tile curriculum reset probability at 0.5 in this continuation. About half of the last 100,000 episodes of the previous long run were artificial starts, but they occupied about one fifth of actions. The optimal mixture remains unknown; a later curriculum-ratio test should change that factor alone. Inspect current actor/critic gradients and training reward components before deciding on another long extension. A separately labelled direct pretrained-weight fine-tuning branch remains planned. Transformers are deferred; no transformer is currently implemented or training.

The [independent reviewer](THREES_TRAINING_REVIEW.md) recommended deferring the queued constant-LR continuation. We initially retained it for an equal-budget comparison, then replaced that decision on September 27 after the constant-LR run stayed materially below decay at matched interim budgets. This resolves the earlier disagreement without claiming a completed equal-budget test.

## Implementation versus proposals

| Item | State |
| --- | --- |
| 13,995,453-parameter GELU encoder/shared LSTM, recurrent PPO, engine v2 | Implemented; local models start from random weights |
| Legal-action masking | Evaluation and inspector use legal actions; the core recurrent trainer remains unmasked. The isolated masked fork policy has trained successfully |
| Complete checkpoint continuation, visible launcher logs, provenance | Implemented |
| Live activation heatmaps and unit traces | Implemented; not identified causal features |
| Recurrent probability/gradient audit described above | Checkpoint and mask-path checks measured; full training reward-component accounting remains |
| Masking × LR forks | Four arms completed and compared on 300 paired development games |
| Completed-fork extension | 20M and 100M masked low-LR children completed through 225,026,048 transitions |
| Matched masked LR probe | Two 5M child arms completed and compared on 300 paired development games; LR `0.00010` underperformed |
| Long masked LR `0.00010` continuation | Completed at 330,039,296 transitions; 300-game post-run assessment completed |
| Matched LR-halving pilot from the 330M checkpoint | Completed, compared on 300 paired natural development games |
| Long masked LR `0.00005` continuation from the 340M pilot endpoint | Launched in a new run; status `in progress` at the dated check above |
| Shaping ablation, pretrained fine-tuning | Planned |
| Transformer training | Deferred |

For parameters and rewards, read the [method](REPRODUCTION.md) and each run's frozen `metadata/config.json`. The [v2 guide](REPRODUCTION_V2.md) describes the initial batch, not a new instruction to launch it again.

## Run and evidence pointers

- Long continuation: [curriculum + LR decay](../results/runs/2026-09-26_162446_recurrent_v2_curriculum_decay_s72/).
- Truncated comparison condition: [constant-LR curriculum](../results/runs/2026-09-26_162446_recurrent_v2_curriculum_s72/).
- Completed fork batch and [300-game paired comparison](../results/fork_batches/2026-09-27_100616_mask_lr/comparison.md); selected continuation parent: [masked LR 0.00005](../results/runs/2026-09-27_100616_mask_lr_masked_lr_00005/).
- Completed [20M masked low-LR extension](../results/runs/2026-09-27_112611_mask_lr_extension_masked_lr_00005/) at 125,018,112 transitions; selected parent for the long continuation.
- Completed [100M masked low-LR extension](../results/runs/2026-09-27_125618_mask_lr_extension_masked_lr_00005/) at 225,026,048 transitions; fixed endpoint selected as the matched LR-probe parent.
- Completed [matched LR-probe comparison](../results/fork_batches/2026-09-27_185554_mask_lr_probe/comparison.md); selected long-continuation parent: [masked LR 0.00010 endpoint](../results/runs/2026-09-27_185554_mask_lr_probe_masked_lr_00010/).
- Completed [long LR 0.00010 continuation](../results/runs/2026-09-27_193437_masked_lr_00010_long_masked_lr_00010/) at 330,039,296 transitions; its post-run assessment is linked below.
- Completed [post-run 300-game assessment](../results/comparisons/2026-09-28_181141_plateau_lr_probe_baseline/comparison.md) and [matched 10M LR pilot](../results/fork_batches/2026-09-28_181141_plateau_lr_probe/comparison.md); selected long-continuation parent: [masked LR 0.00005 endpoint](../results/runs/2026-09-28_181141_plateau_lr_probe_masked_lr_00005/).
- Launched [long masked LR 0.00005 continuation](../results/runs/2026-09-28_192644_mask_lr_extension_masked_lr_00005/) from the fixed 340M pilot endpoint; preserve its live artifacts.
- Completed 10M baseline stage: [three-condition comparison](../results/comparisons/2026-09-26_162446_recurrent_v2/comparison.md).
- Evidence used for the decision: [snapshot through 66,715,648 transitions](../results/reviews/2026-09-26_training_decision/). At that cutoff, the best 100-game validation mean was 3,480.81 at 63,340,544 transitions. This sentence is a dated observation, not a live maximum.
- [Read-only audit at the interrupted 74,235,904-transition checkpoint](../results/reviews/2026-09-26_recurrent_audit/README.md): PPO probabilities reconstructed closely on one fresh rollout; invalid-action mass and weighted critic gradients were substantial. This is measured engineering evidence, not a training-mask result.

Inspect `data/metrics.csv`, `models/best.json`, complete checkpoint markers and continuation logs for newer progress. Recorded `in progress`/`running` status is not proof of a live process. [Continuation instructions](CONTINUING_RUNS.md) explain read-only preview/list commands.

## What would change the decision

A demonstrated trainer defect takes precedence over the long continuation. Fresh ablation results can change the next intervention. Sustained growth on periodic validation is encouraging but does not alone establish reproducibility, published performance, or a reason for unlimited extensions. Reserve seeds 900000–909999 until model selection ends.
