---
status: reference
updated: 2026-09-27T19:33:00-04:00
scope: Isolated masking by learning-rate pilot forks of a completed recurrent run
---

# Recurrent masking by learning-rate forks

The [current status](CURRENT_STATUS.md) decides the next allocation. The four-arm pilot and [paired 300-game comparison](../results/fork_batches/2026-09-27_100616_mask_lr/comparison.md) completed on September 27, followed by 20M and 100M masked low-LR continuations and a short [masked LR probe](MASKED_LR_PROBE.md). The current [long-LR guide](MASKED_LR10_LONG.md) covers the next continuation. The original pilot used the completed decay run's exact 100,007,936-transition endpoint as its fixed parent, not a selected `best.zip` checkpoint.

## Design

Each arm restores the same complete parent checkpoint: weights, optimizer moments, active boards, bags, LSTM memory, environment counters and RNG. Every arm discards unfinished rollout data and collects fresh rollouts under its own action distribution. The parent run is read only. The fork runner and policy live in `src/reproduction_forks/`; the four core trainer files used by the active continuation are unchanged.

| Arm | Training action mask | Fixed effective LR | Additional attempted transitions |
| --- | --- | ---: | ---: |
| `unmasked_lr_00015` | Off | 0.00015 | 5,005,312 |
| `masked_lr_00015` | On | 0.00015 | 5,005,312 |
| `unmasked_lr_00005` | Off | 0.00005 | 5,005,312 |
| `masked_lr_00005` | On | 0.00005 | 5,005,312 |

The configured 5,000,000-transition request rounds to 611 complete 8,192-transition rollouts. The inherited LR milestone is removed in each fork, and optimizer parameter groups are set to the arm's effective LR. All other PPO, curriculum, reward, simulator and architecture settings are copied from the selected parent. The optimizer state is preserved in every arm. This is a four-treatment study from **one** training trajectory, not four independent random-initialization replications.

The masked policy applies the game's legal-move rule to the **pre-action observation** for sampling, stored old log probabilities, recomputed training log probabilities, entropy and prediction. The fork buffer also records each pre-action mask and checks it against the stored observations before every update. Padded zero observations receive a finite uniform dummy distribution; padded losses remain excluded by recurrent PPO's sample mask. Real no-legal-action states raise an error because terminal states must reset before sampling. A one-legal-action state has zero entropy and policy gradient.

## Original pilot launch and recovery

These commands reproduce the completed pilot design from the decay endpoint. Running them again without the original group creates another training allocation. Use a visible PowerShell terminal from the project root:

```powershell
.\run_reproduction_forks.ps1 -ParentRun results\runs\2026-09-26_162446_recurrent_v2_curriculum_decay_s72 -ParentSteps 100007936 -Preview
.\run_reproduction_forks.ps1 -ParentRun results\runs\2026-09-26_162446_recurrent_v2_curriculum_decay_s72 -ParentSteps 100007936 -All
```

The first command is read only. The second creates a timestamped group under `results/fork_batches/`, runs four arms sequentially in new `results/runs/` directories, and then evaluates the parent endpoint, all four fork endpoints and their selected-best checkpoints. It refuses an in-progress parent, source/version mismatch, or an active recurrent worker. It never resumes or changes the parent. The group name printed by the launcher identifies its plan and comparison artifacts.

For one arm, replace `-All` with `-Arm masked_lr_00015` (or another table name). If an arm is interrupted, preview and resume only its remaining allocated budget:

```powershell
.\run_reproduction_forks.ps1 -ResumeRun results\runs\GROUP_masked_lr_00015 -Preview
.\run_reproduction_forks.ps1 -ResumeRun results\runs\GROUP_masked_lr_00015
```

Resume archives pre-resume metrics, episodes, fork-update diagnostics, best weights, evaluation files and provenance. It restores the latest complete fork checkpoint, trims rows beyond that boundary and verifies the fork and core method hashes. Once an interrupted arm completes, rerun `-All -Group GROUP` with the same parent arguments. Completed arms are verified and skipped; missing arms run, then the common comparison is generated.

## Completed fork extensions

The selected masked low-LR pilot was extended from 105,013,248 to **125,018,112**, then to **225,026,048** transitions through separate completed child runs. The isolated `extend_reproduction_fork.ps1` launcher restores a completed fork endpoint into a new run while preserving its mask and fixed LR. `-Preview` reads a checkpoint and checks source compatibility. The exact commands and evidence for these completed extensions are in the [long-extension decision](RESEARCH_DECISION_2026-09-27_LONG_EXTENSION.md); the next LR-changing experiment uses the separate [matched probe launcher](MASKED_LR_PROBE.md).

The child inherits the same training mask and fixed LR. It restores weights, optimizer, live simulator, recurrent memory and RNG, then starts new rollouts. The launcher refuses a non-complete fork, a hash/version mismatch or a concurrent continuation. If interrupted, use `run_reproduction_forks.ps1 -ResumeRun PATH_TO_CHILD` to finish only the child's allocated budget. The group and child paths are printed when training starts. A same-LR extension does not automatically run the four-arm pilot comparison.

## Read the results

Each fork has frozen `metadata/config.json`, `metadata/fork.json`, `metadata/method.json`, source provenance, visible `metadata/terminal.log`, full checkpoints, periodic natural validation and `data/fork_updates.csv`. The supplemental CSV records actual PPO epochs, valid moves, invalid attempts, one-legal-action counts, KL and clip fraction per rollout. The standard metrics file continues to report validation and optimization curves. The terminal retains frequent progress lines and shows reach 192/384/768 on validation lines; `terminal.log` preserves the trainer's original output.

The group comparison uses 300 paired natural games, seeds **820000–820299**. It reports fixed-budget endpoints against the parent and paired mask/LR contrasts with bootstrap intervals. Selected-best checkpoints are shown separately because periodic 100-game validation (seeds 700000–700099) selected them. These are development results; final seeds 900000–909999 remain untouched. Interpret any promising treatment with its wall time, attempted transitions, valid moves and uncertainty. The [current decision](RESEARCH_DECISION_2026-09-27_LR10_LONG.md) retains replication as a requirement before attributing the gain to masking.

Short CPU tests cover legal sampling, stored-mask agreement, probability reconstruction through recurrent resets and padding, a one-legal-action state, parent optimizer restoration, a real PPO update, bitwise checkpoint-resume parity and extension planning. The pilot's treatment performance has been measured; it is still development evidence from one parent trajectory.
