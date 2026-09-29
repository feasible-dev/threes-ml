---
status: decision
updated: 2026-09-28T18:08:32-04:00
scope: Completed long LR 0.00010 continuation and next plateau experiment
---

# September 28 plateau decision

This decision replaces the **100M LR `0.00010` continuation as the next action** in [the September 27 decision](RESEARCH_DECISION_2026-09-27_LR10_LONG.md). That continuation completed at 330,039,296 attempted transitions. It does not replace or rewrite the completed run's evidence.

## Measured evidence and limits

The completed [long masked LR `0.00010` run](../results/runs/2026-09-27_193437_masked_lr_00010_long_masked_lr_00010/) restored the 230,031,360-transition short-probe endpoint, then added 100,007,936 attempted transitions with training masks and fixed LR `0.00010`. Its final periodic natural validation mean was **6,298.05**, with 51/100 games reaching tile 384 and 3/100 reaching 768. Its selected-best checkpoint was at 307,748,864 transitions, with **7,968.9** on the same reused 100 validation seeds. Across 198 periodic evaluations on seeds 700000–700099, the mean of evaluations in the first 20M segment was 6,078.2 and in the last complete 20M segment was 6,641.2. These correlated, reused-seed measurements show noisy, modest progress, not a statistical proof of a flat learning curve or of improvement over a different LR.

In the final 20M transitions, logged PPO updates averaged approximate KL **0.0147**, clip fraction **0.0407**, and explained variance **0.9276**; 5 of 2,447 updates used fewer than four epochs. The terminal value loss alone does not diagnose a critic problem. In the last 100,000 logged training episodes, 49,763 were curriculum episodes, averaging 57.6 valid moves; 98.3% finished with a tile at least 768. The 50,237 natural episodes averaged 231.6 moves, and 3.2% reached 768. Curriculum episodes occupied about 19.7% of logged actions in the final 20M. The episode window is defined by the last 100,000 rows, not by the final 20M transitions.

The [earlier matched 5M LR probe](../results/fork_batches/2026-09-27_185554_mask_lr_probe/comparison.md) favored `0.00005` over `0.00010` by 1,020.9 mean points on 300 paired development games. It is evidence about that checkpoint and horizon, not a controlled comparison of the completed 100M continuation. All these lineages share training seed 72 and are not independent-training-seed replications.

The [author's article](https://medium.com/@josiah-kiok/beating-threes-with-reinforcement-learning-ae074dd28a68) reports halving LR after plateaus and further progress; it does not publish the final successful schedule or full PPO settings. The [released project's README](https://github.com/pseudonam-gc/threes-web) describes adding 3–6 large tiles for curriculum. Our implementation already does this at eligible resets with probability 0.5, using the environment's natural-game rank frontier. The actual training state and evaluation distributions differ; additional arbitrary high-tile starts could further overweight short artificial endgames.

## Accepted next action

1. Evaluate the fixed 330,039,296 endpoint, its separately labeled selected-best checkpoint, and the 230,031,360 parent on the **same 300 natural development games, seeds 850000–850299**. Report means, medians, tile reach, paired game-score differences and uncertainty. This completes the planned post-run assessment. These seeds then become development data; retain 900000–909999 for final testing.
2. Run an isolated **matched 10M-transition LR pilot** from the complete 330,039,296 checkpoint. Both arms restore its weights, optimizer, live simulator, recurrent state and RNG, and each adds 10,002,432 attempted transitions, ending at 340,041,728. The control retains LR `0.00010`; the treatment halves LR to `0.00005`. Keep masking, curriculum probability 0.5, rewards, gamma, lambda, PPO clipping, value/entropy weights, rollout size, architecture and engine fixed. Evaluate fixed endpoints on 300 paired natural development games, seeds 860000–860299; keep validation-selected peaks separate. The [launcher](PLATEAU_LR_PROBE.md) is implemented and previewed; no training has been launched. The user starts long training/evaluation in a visible terminal.
3. Before a much longer extension, inspect current-checkpoint actor and critic gradient norms and clipping, PPO KL/clip fraction, and natural versus curriculum reward components. If the LR pilot improves natural-game score and 768 reach without a clear median loss, extend the better arm in a new run with a bounded budget. If it does not, prioritize a matched `vf_coef`/shared-gradient or shaping ablation according to the diagnostics. Do not infer from value loss alone which intervention wins.

Do **not** increase the share of artificial high-tile starts in the first pilot. The current curriculum already supplies abundant >=768 states, while natural games rarely reach them. A later curriculum-ratio test should change only the reset mixture and judge natural-start performance at equal attempted-transition budget. Gamma, GAE lambda, entropy coefficient, rollout length and PPO clip range remain candidates for later targeted pilots; their values are local choices, but changing them together with LR would obscure attribution.
