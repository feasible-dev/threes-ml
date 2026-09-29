---
status: proposal
updated: 2026-09-27T10:02:25-04:00
scope: Threes training review - next decisions
---

# Threes training review: next decisions

> Independent proposal, not the current plan. The [September 26 decision](RESEARCH_DECISION_2026-09-26.md) initially retained the constant-LR run; the [September 27 interim decision](RESEARCH_DECISION_2026-09-27.md) replaces that plan. See [current status](CURRENT_STATUS.md).

Review of the September 26, 2026, 20:25 EDT snapshot in `RESEARCH_HANDOFF(1).md` (the external handoff filename; that file is not present in this repository). The [frozen evidence bundle](../results/reviews/2026-09-26_202545_research_handoff/) preserves the supplied metrics and configurations.

Scope: the supplied handoff and the primary sources linked below. I have not inspected the local repository, checkpoints, full metrics or per-game CSVs, and have not run an experiment. Recommendations below are proposed interventions, not measured improvements. Read current run status before applying them; the snapshot is not live.

## Decision

The agent learned substantially, then progress slowed. There is no demonstrated numerical collapse or permanent performance ceiling. Reallocate the next training budget to the masking experiment after a short trainer audit. Do not automatically spend the next queued 90M transitions on the other unmasked condition before reviewing that result.

Keep the architecture for now. The released policy's strong results in this engine demonstrate that this architecture can represent strong play with this interface. They do not show that the local PPO recipe can discover those weights.

If the practical goal is a strong agent that we can train and improve, use a separate branch initialized directly from the released weights. Because the inference computation and weights are already mapped, imitation of an identical network is unnecessary for that first branch. Keep its provenance distinct from learning from random initialization.

## What the evidence supports

| Observation from the handoff | Interpretation |
| --- | --- |
| Periodic window mean rises from 1,369 at 10–15M to 2,725 at 45–48.61M | Real progress over the broader run |
| 35–40M window mean 2,626 versus 2,725 at 45–48.61M | Only about 3.8% further improvement; a slowdown worth investigating |
| Latest mean 2,949 versus best-of-many 3,368 | Insufficient evidence of regression: checkpoint selection is optimistic, and 100-game evaluations are noisy |
| Invalid training attempts often 25–40%, sometimes over 60% | Strongest measured actionable problem |
| KL and explained variance look reasonably stable | No obvious numerical blow-up; neither metric validates policy learning or recurrent implementation |
| Released reference mean 271,897; 55/300 reaching 6,144 | A strong policy exists for this engine/interface; no need to speculate about representational capacity first |

Averaging adjacent evaluations is descriptive, not independent replication. The snapshot supports “recent slow progress,” not a statistical proof of stagnation. Use per-game results and fixed-budget checkpoints to test the next decisions.

## Suspects, in priority order

1. **Action handling and recurrent training/evaluation mismatch.** Training permits invalid actions; evaluation takes the highest legal logit. Invalid attempts consume computation, discounting, timeout budget and LSTM updates without moving the board. Thus masking changes the experienced decision process as well as sample efficiency. At a fixed attempted-action count, removing a 25–40% invalid fraction mechanically gives roughly 1.33–1.67 times as many valid moves, before behavioral effects. Do not promise that this alone closes the enormous score gap. Action masking is a supported policy-gradient technique [1], but it must be implemented consistently.
2. **Reward incentives and shared-gradient optimization.** The optimized reward is a proxy for the selected metric. Equal-rank merge reward grows linearly with rank, whereas its score gain grows exponentially. Positional bonuses pay repeatedly for occupying favored configurations. This may help learning or encourage a limited strategy; neither is established. Also, critic gradients flow through the shared encoder/LSTM, intentionally differing from SB3's shared-LSTM default [3]. Value loss of 4–8 does not establish gradient domination. Measure gradients before diagnosing it.
3. **Curriculum and temporal targets.** Artificial starts can change the learned distribution, and a frontier based on past natural games makes it evolve. Measure natural and curriculum outcomes separately. Timeouts are artificial endings whose incidence in training is unknown. A 64-step recurrent sequence may limit learning, but longer sequences are not automatically the answer.

The LR remains a reasonable secondary suspect, not an established fault. The architecture is currently a lower-priority suspect than the training procedure and data distribution.

For the temporal distinction, GAE weights TD residuals by `(gamma * lambda)^k` [5]. Here `gamma * lambda = 0.94905`; the characteristic trace scale is about 19.6 attempts, and the weight at lag 64 is about 0.035. This does **not** mean the agent can plan only 20 moves ahead: a learned value function can carry information about much later rewards. Invalid attempts further reduce board progress per trace. Increasing rollout length while leaving inaccurate values untouched need not fix credit assignment.

## Short engineering gates before more long training

These are correctness and instrumentation checks, not a hyperparameter sweep.

- **Reconstruct rollout probabilities before any update.** With weights unchanged, replay stored observations/actions through the training minibatch path using the recorded initial recurrent states. Selected-action log probabilities must match those stored at collection: PPO ratios approximately one and KL approximately zero. Check the unmasked implementation first, then the masked one. Include episode resets, rollout boundaries and padded sequences. This is a standard high-information PPO check [4].
- **Audit masks throughout the PPO path.** Use the same legal distribution for sampling, stored old log probabilities, recomputed log probabilities and entropy. A mask belongs to the pre-action state. Exclude padding from every loss and reduction; provide a finite dummy distribution for padded entries. Test a one-legal-action state, whose categorical entropy and policy gradient are zero. Never sample an all-invalid terminal state: reset it first. SB3's standard MaskablePPO does not support recurrent policies [2]; this is not a class-name replacement.
- **Measure gradients on a few actual minibatches.** Separately measure policy-loss and weighted-value-loss gradients into the encoder/LSTM, their cosine similarity, and the global norm-clipping multiplier. Audit the intended gradient paths and `requires_grad` settings. Large scalar losses are not sufficient evidence of interference.
- **Break down the experience.** Log invalid probability mass and streaks; attempted actions, valid moves and natural games per second; natural/curriculum reward components; raw and clipped reward totals; clipping frequency; actual PPO epochs, KL and clip fraction; training timeouts and true game-over counts. Report legal entropy conditional on the number of legal actions.
- **Check timeout targets if timeouts occur.** Natural game-over is terminal. If an artificial collection timeout is intended only as a cap, continuing-task targets should bootstrap from the terminal observation with its correct recurrent state, never the reset observation. If the finite horizon is intended as the task, that is a different objective; hidden, history-dependent limits deserve explicit treatment. Do not change this inside the masking ablation unless correctness requires it; document and apply any prerequisite correction to all arms.
- **Account for reward on frozen policies.** Run the current agent and reference through reward accounting on natural games. Compare score, discounted return, undiscounted return and components. A mismatch is diagnostic, not proof of a causal reward bug. Positive correlation likewise does not prove good local learning incentives.

If the probability reconstruction fails, fix it before comparing hyperparameters. If training timeouts are frequent or critic gradients overwhelm the policy under global clipping, reconsider the order below.

## Experiment 1 — masking × learning rate

**Hypothesis:** unmasked training wastes useful interactions and learns behavior that differs materially from legal-action evaluation; LR reduction may independently stabilize adaptation.

I agree with the handoff's four-way experiment. Its approximately one-hour training cost is reasonable compared with another pair of long continuations. If engineering time is the constraint, implement masking and its correctness checks first, then run the four arms.

**Parent:** the latest complete checkpoint at the planned intervention boundary, after checking current status. Do not choose a lucky validation peak. Freeze its source/config hashes, optimizer, simulator and recurrent state. All arms start from the same complete parent. Preserve optimizer state consistently. Discard any unfinished rollout in all arms; collect fresh data under each arm's actual action distribution.

| Arm | Training mask | Fixed LR | Additional pilot budget |
| --- | --- | ---: | ---: |
| A | No | 0.00015 | 5M attempted transitions |
| B | Yes | 0.00015 | 5M |
| C | No | 0.00005 | 5M |
| D | Yes | 0.00005 | 5M |

Round budgets identically to complete rollouts. Keep reward, network, curriculum, gamma, lambda, rollout length, minibatch size, epochs, entropy coefficient, value coefficient and termination handling fixed. Correct LR schedule restoration explicitly so a resume callback does not overwrite the intervention.

**Runtime:** 20M total transitions divided by the reported 5,000–5,700/s is about 58–67 minutes of training. Allow roughly 1–1.5 hours provisionally for training overhead, then replace this with a short benchmark. Evaluation and implementation time are additional; benchmark evaluation separately. Log wall time, valid moves and completed natural games so the experiment measures both learning and useful interaction efficiency.

**Evaluation:** use the common protocol below. Primary comparisons are B versus A and D versus C for masking, and C versus A and D versus B for LR. Evaluate the parent once on the same validation set. Do not call four forks from one parent four independent training seeds.

**Decision:** zero invalid sampled actions is only an implementation gate. For a practical promotion criterion, predeclare approximately 20% improvement in mean natural-game score over the matched control, a game-level confidence interval favoring the candidate, no clear median deterioration and supportive tile-reach results. The 20% is a chosen engineering threshold, not a theorem. Smaller or uncertain effects justify an extension, not a victory claim. If all arms improve similarly, continued training explains the gain. If masks remove invalid actions but natural-game performance does not improve, the simple “masking quickly fixes it” hypothesis has failed at this budget, not for all possible training lengths.

**Risks:** inconsistent masked log probabilities, padded all-invalid distributions, changed recurrent histories, changed critic targets after masking, and inadvertently modifying reward or optimizer treatment. Keep separate experiment directories and immutable parent provenance.

## Experiment 2 — remove repeated structural bonuses

**Hypothesis:** repeated positional rewards favor a strategy that earns training return without sufficient improvement in natural-game score or large-tile success.

**When:** after the masking implementation is verified and a usable masked parent exists. If the preceding diagnostics expose a more direct gradient or target problem, address that before this experiment.

**Parent and change:** two forks from one fixed masked checkpoint. One retains the current reward. In the other, set the full-first-row, corner/second-maximum and monotonicity coefficients to zero. Leave the already-zero snake coefficient unchanged. Keep merge rewards, terminal penalty, clipping, masking, network, optimizer treatment, LR, curriculum and all PPO parameters identical. This isolates structural bonuses; it does not yet test raw-score optimization.

**Budget/runtime:** 5M additional transitions per arm, about 29–34 minutes at the existing throughput; provisionally 30–45 minutes of training plus evaluation. Evaluate after initial adaptation as well as at the final budget. The inherited critic initially estimates the old return, so do not interpret the first few updates as decisive. Preserve and report that transient rather than silently resetting only one critic.

**Evaluation and decision:** common protocol below, with the same practical promotion criterion. Falling training return after removing bonuses is expected and is not failure. Higher natural score and tile reach are success. If removal hurts or produces no useful signal, retain shaping for now; this does not prove it is optimal. If structural bonuses were negligible, or little reward clipping occurred, deprioritize those explanations. Do not change to a logarithmic score reward, add milestones, remove clipping and disable curriculum simultaneously.

**Risks:** losing useful exploration guidance, critic adaptation to new targets, and misreading incomparable reward scales. Potential-based shaping can preserve optimal policies under its assumptions [6], but simply replacing bonuses by a potential while retaining arbitrary clipping does not automatically retain that guarantee.

## Experiment 3 — a strong, trainable branch from released weights

**Hypothesis:** the local stack can preserve and adapt an already strong policy. This tests a practical transfer route, not the ability of scratch PPO to discover that policy.

**Initialization:** import the mapped released ONNX weights into the trainable implementation, with fresh Adam state. This is a new pretrained run, not a resume. Preserve the frozen imported baseline. Use natural starts only and consistent masking. Keep the existing reward for this first viability pilot so reward redesign does not add another uncertainty.

**Before updates:** compare imported PyTorch and ONNX legal-greedy play on 100 common natural seeds. Small floating-point discrepancies can occasionally change close decisions; investigate material divergence rather than demanding every long game be bitwise identical. Also evaluate the imported network under masked stochastic sampling. Greedy strength does not guarantee that stochastic PPO collection remains strong.

**Critic calibration:** collect 64 rollouts, 524,288 transitions. Freeze the encoder, LSTM and actor; update only the critic head at LR 0.0001 using local return targets. Ensure policy outputs remain unchanged. The original critic's return scale and training recipe are unverified. If target residuals remain badly biased, extend calibration or investigate before unfreezing; a fixed warm-up budget is not proof of calibration.

**Fine-tuning:** unfreeze the network for 256 rollouts, 2,097,152 transitions, at LR 0.00001. Retain the current PPO settings, including entropy coefficient, except for the explicitly listed initialization, LR, masking and natural-only reset distribution. Keep the frozen baseline, evaluate every 524,288 fine-tuning transitions, and stop a clearly destructive pilot early. This package is a viability test, not a controlled comparison to the scratch curriculum runs.

**Runtime:** the 2.62M collected transitions imply 8–9 minutes at the old throughput. Treat roughly 10–30 minutes of training/calibration as a provisional planning range, subject to measured performance. Strong-policy evaluation can dominate because games last much longer; measure it on 100 games before allocating the final comparison. Do not estimate the full job from parameter count or extrapolate weak-agent evaluation speed.

**Evaluation and decision:** compare the frozen imported baseline and fixed-budget fine-tuned policy on common natural validation games. Preserving roughly 90% or more of baseline mean with comparable large-tile rates is a useful engineering gate, not a statistical equivalence result. Improvement requires confidence intervals and later repeated runs. A collapse can reflect stochastic collection, reward mismatch, poor critic calibration, update size or a recurrent bug; it does not identify any one of these by itself.

**Risks:** unverified critic semantics, policy drift under shared critic gradients, catastrophic forgetting and collecting weak trajectories from a strong greedy policy. If fine-tuning is viable, it immediately provides a strong platform for further work. If the goal remains entirely teacher-free learning, keep experiments 1–2 as the scientific track and this branch as a separately labeled practical result. Imitation becomes useful when changing the architecture or model size.

## Common evaluation protocol

- Use 1,000 common natural-game validation seeds, such as the planned `810000–810999`, after confirming their current role in the repository. They are validation once used for these decisions. Leave `900000–909999` unopened.
- The primary endpoint is the fixed-budget final checkpoint. Report validation-selected best checkpoints separately; do not compare a selected best in one arm with a final checkpoint in another.
- Report score mean and median, valid moves per game, tile-384/768/1536/3072/6144 reach rates and truncations. Do not select on maximum score, critic loss or explained variance.
- Bootstrap game/seed-level score differences and report binomial uncertainty for reach rates. Pairing by seed is permissible but does not imply identical random trajectories. Those intervals measure game variation for fixed checkpoints, not uncertainty across training initializations or repeated model selection.
- Use periodic evaluations for monitoring, not hundreds of independent significance tests. Replicate promising procedures with independent training seeds before claiming a method advantage.
- Keep engine revision and observation privileges explicit. Port the heuristic to engine v2 before making a current heuristic-versus-neural claim.

## Reproduction and literature boundaries

The reported C/Python audit and inference equivalence checks are useful evidence, but I have not independently rerun them. Compatibility with a pinned community engine does not establish complete equivalence to commercial Threes. Reset changes, bonus scheduling, termination, bag information and the unknown original training configuration remain comparison boundaries. In particular, a numerical forward match does not validate gradients, optimizer behavior, recurrent minibatches or targets.

| Primary source | What is usable | What it does not establish |
| --- | --- | --- |
| [Released Threes project](https://github.com/pseudonam-gc/threes-web) | Recurrent ONNX weights already working locally; author describes PPO/PufferLib and billions of training steps | A complete successful training recipe or proof that the released checkpoint is the later 21.5% model |
| [Multi-Stage TD for 2048-like games](https://arxiv.org/pdf/1606.07374) | Threes-specific n-tuple learning plus search; reported 6,144 reach increases from 0.45% to 7.83% | A comparable neural-policy-only PPO result or identical simulator rules |
| [2048former model card](https://huggingface.co/jkeisling/2048former-50m-v0.1) | Released transformer for 2048, trained by supervised imitation of expectimax; author reports 605,491 mean and 66.1% reaching 32,768 | Evidence that a transformer fixes scratch PPO for Threes; these are a different game and learning setup |

The targeted search did not verify a strong Threes-specific transformer. That is not an absence or novelty claim. Leave transformer work for an architecture comparison with controlled action handling and training, or for a separately labeled teacher-trained student.

## What would change this recommendation

- A failed pre-update probability reconstruction: prioritize a trainer fix.
- Frequent artificial training timeouts: prioritize correct target semantics and timeout handling.
- Strong evidence that critic gradients dominate or conflict with actor updates under clipping: prioritize shared-gradient handling before the structural-reward experiment.
- Current results, newer than this snapshot, show sustained fresh-validation improvement: reconsider the proposed pause in long continuations.
- Correct masks and LR controls offer no benefit: shift attention to targets, reward and data distribution, not automatically a larger network.
- A teacher-initialized policy cannot survive small updates after calibration: examine stochastic trajectories and learning-path correctness before adding more runtime.

## Technical references

1. [Huang & Ontañón, invalid action masking](https://arxiv.org/abs/2006.14171).
2. [SB3-Contrib MaskablePPO documentation](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html): recurrent policies are not supported by the standard class.
3. [SB3-Contrib RecurrentPPO documentation](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html): shared-LSTM gradient convention and recurrent interfaces.
4. [The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/): reconstruction and ratio-one debugging check.
5. [Schulman et al., Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438).
6. [Ng, Harada & Russell, policy invariance under reward transformations](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf).
