---
status: decision
updated: 2026-09-28T19:22:39-04:00
scope: Completed plateau LR pilot and selected long low-LR continuation
---

# September 28 long low-LR continuation decision

This decision replaces the **matched 10M LR-halving pilot as the next action** in the [earlier plateau decision](RESEARCH_DECISION_2026-09-28_PLATEAU.md). The user completed the pilot and requested a long continuation at the lower learning rate. The completed pilot and its baseline assessment remain distinct evidence.

## Measured evidence

The [post-run baseline assessment](../results/comparisons/2026-09-28_181141_plateau_lr_probe_baseline/comparison.md) evaluated the 230M parent, fixed 330M endpoint and validation-selected 307M model on 300 common natural development games, seeds 850000–850299. Means were 5,972.0, 6,502.4 and 6,821.4 respectively. The 330M-minus-230M paired difference was +530.4 with a 95% game-level bootstrap interval of −142.2 to +1,207.5. The selected-best-minus-fixed-endpoint difference was +319.0 with interval −373.5 to +1,038.2. The selected model was chosen on the reused 100-game periodic set and remains separate from the fixed endpoint.

The [matched LR pilot](../results/fork_batches/2026-09-28_181141_plateau_lr_probe/comparison.md) restored the same 330,039,296-transition checkpoint in both masked arms and added 10,002,432 attempted transitions each. On 300 paired natural development games, seeds 860000–860299, the parent averaged 6,521.2, the LR `0.00010` control 7,512.3, and the LR `0.00005` treatment 8,287.6. Tile-768 reach was 3.0%, 4.7% and 7.7%. Treatment minus control was +775.3 mean points, with a 95% paired game-level bootstrap interval of **−66.4 to +1,596.0** and 165 wins versus 135 losses. The treatment is the stronger observed arm, but the interval includes zero. Both arms share training seed 72 and their parent; this is not independent-seed replication or final-test evidence.

The curriculum reset probability remained 0.5 in both arms. In the preceding long run, about half of the last 100,000 completed training episodes were artificial high-tile starts but they were shorter, occupying about 19.7% of attempted actions in the final 20M transitions. This proportion does not show whether the mixture is optimal. The [released project](https://github.com/pseudonam-gc/threes-web) confirms use of large-tile starts but does not publish a final mixture ratio. The [author's article](https://medium.com/@josiah-kiok/beating-threes-with-reinforcement-learning-ae074dd28a68) describes halving LR after plateaus, without providing a transferable final schedule.

## User-selected next allocation

Continue the completed **masked LR `0.00005` pilot endpoint at 340,041,728 attempted transitions** in a new child run for **100,007,936 additional attempted transitions** (100M requested, rounded to complete 8,192-transition rollouts), targeting **440,049,664**. Keep the legal-action mask and fixed LR `0.00005`. Keep the existing curriculum reset probability **0.5** for this continuation, along with rewards, PPO settings, architecture and simulator. This preserves the only measured low-LR method while collecting a longer trajectory; a later curriculum-ratio test can change that one factor from a fixed parent. Restore weights, optimizer moments, live simulator, recurrent memory and RNG from the exact complete endpoint. Preserve the completed pilot and earlier runs.

The existing `extend_reproduction_fork.ps1` launcher is implemented and previewed against this parent with no source/version compatibility issues. The user starts the long run in a visible terminal:

```powershell
.\extend_reproduction_fork.ps1 -ParentFork results\runs\2026-09-28_181141_plateau_lr_probe_masked_lr_00005 -AdditionalSteps 100000000 -Preview
.\extend_reproduction_fork.ps1 -ParentFork results\runs\2026-09-28_181141_plateau_lr_probe_masked_lr_00005 -AdditionalSteps 100000000
```

The continuation measures the chosen path's progress; it does not isolate a long-horizon LR or curriculum effect without matched controls. At completion, evaluate the fixed 340M and 440M endpoints on fresh natural development games, seeds 870000–870299, and show validation-selected peaks separately. Keep final-test seeds 900000–909999 unopened until method and checkpoint selection are frozen.
