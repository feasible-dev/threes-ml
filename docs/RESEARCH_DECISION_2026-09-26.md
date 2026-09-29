---
status: decision
updated: 2026-09-27T10:02:25-04:00
scope: Research decision after independent review
---

# Research decision after independent review

Supersession: [September 27 interim LR decision](RESEARCH_DECISION_2026-09-27.md) replaces section 2's equal-100M-budget second continuation and 1,000-game comparison. The original rationale below is retained as a dated decision record.

Decision date: September 26, 2026. Evidence cutoff: 66,715,648 attempted-action transitions in the curriculum/LR-decay run. This is a decision record, not a claim that the proposed experiments have been implemented.

Inputs: the independent [training review](THREES_TRAINING_REVIEW.md), the [earlier frozen evidence](../results/reviews/2026-09-26_202545_research_handoff), the [updated validation snapshot](../results/reviews/2026-09-26_training_decision/validation.csv), and the current launcher. The reviewer did not inspect the repository directly; its recommendations require engineering verification.

## 1. The new records support slow progress

Run: `2026-09-26_162446_recurrent_v2_curriculum_decay_s72`.

| Periodic checkpoint | Mean score | Median score | Games reaching 384 |
| --- | ---: | ---: | ---: |
| Previous record, 37,437,440 transitions | 3,368.34 | 2,914.5 | 16/100 |
| New record, 63,340,544 transitions | 3,480.81 | 2,899.5 | 20/100 |
| Latest evaluation at this cutoff, 66,387,968 transitions | 2,895.39 | 2,580.0 | 13/100 |

The record mean improved by 3.3%. The record checkpoint's median did not improve. None of these evaluations reached 3,072 or 6,144. They use the same 100 natural-start validation seeds, 700000–700099.

The broader trend is more useful than the single highest checkpoint:

| Transition window | Evaluations | Average validation mean | Average fraction reaching 384 |
| --- | ---: | ---: | ---: |
| 35–40M | 10 | 2,625.55 | 9.6% |
| 40–45M | 9 | 2,576.40 | 9.3% |
| 45–50M | 10 | 2,724.21 | 10.2% |
| 50–55M | 10 | 2,759.92 | 10.3% |
| 55–60M | 10 | 2,756.39 | 9.1% |
| 60–65M | 10 | 2,861.26 | 12.0% |

The 60–65M window is about 9% above 35–40M. This supports continued slow progress, not a demonstrated permanent ceiling or a decisive breakthrough. These are correlated checkpoints evaluated on reused seeds; window averages are descriptive, not independent replications. Selecting the best of many checkpoints also makes its estimate optimistic.

## 2. Finish the allocated comparison

Keep the running batch's existing bounded budget:

1. Finish curriculum with LR decay at approximately **100,007,936 total transitions**.
2. Retain the queued constant-LR curriculum continuation to the same total budget.
3. Let the existing launcher compare the selected models and released reference on 1,000 natural games per agent, seeds 810000–810999.

This differs from the review's recommendation to defer the queued 90M continuation. The reason is to complete the already allocated equal-budget LR comparison while the first condition still shows slow improvement. Constant LR has only been measured to 10M; comparing its present result with a 100M decay result would confound training budget and LR. The cost of preserving this comparison is approximately another 4–5 hours of training for the second condition, plus evaluation, at the observed throughput. This is a bounded allocation decision, not a prediction that constant LR will win.

Both conditions use training seed 72. They compare these two trajectories; they do not establish reproducibility across training seeds. The batch selects `best.zip` using periodic validation. Its 1,000-game comparison is a development comparison, not the reserved final test. A fixed-budget endpoint evaluation would be an additional useful analysis, not something the current launcher already performs.

Do not rerun `run_long_reproduction.ps1` to apply this decision: it adds its requested budget again. The existing batch already expresses this plan. No further long unmasked extension is allocated after this pair. A demonstrated correctness defect would take precedence over finishing the comparison.

## 3. Audit the trainer before the next experiment

Keep the current GELU encoder plus shared 1,024-unit LSTM. The released weights demonstrate strong representational capacity in our evaluation interface; they do not validate our learning procedure.

The next engineering work is:

- Reconstruct stored rollout action log probabilities through the recurrent training minibatch path **before changing weights**. Ratios should be approximately one and KL approximately zero on real, unpadded samples. Cover recurrent states, episode resets, rollout boundaries and padding. Any discrepancy must be explained or fixed before interpreting the ablation.
- Measure actor and weighted-critic gradients entering the shared encoder/LSTM, their alignment, and the effect of global gradient clipping. Scalar value loss alone cannot diagnose gradient interference.
- Add valid moves versus attempted actions, invalid probability mass/streaks, natural versus curriculum outcomes, timeout counts, reward components/clipping, and actual update/KL diagnostics.
- Inspect artificial-timeout targets if those endings occur materially. Keep any prerequisite correctness correction consistent across experimental arms.

These checks are **planned, not completed**. Develop and test them separately from the sources used by the running continuation. See the [PPO implementation discussion](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) and [MaskablePPO documentation](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html). Standard MaskablePPO does not support recurrent policies; replacing a class name is insufficient.

## 4. Next new training budget: masking × LR

After the audit, use one complete, preselected parent checkpoint from the completed long run. Select the endpoint/checkpoint by a fixed budget rule, not by a lucky validation peak. The previously suggested 60M intervention boundary has already passed.

| Arm | Legal-action mask during training | Effective LR | Additional transitions |
| --- | --- | ---: | ---: |
| A | No | 0.00015 | 5M |
| B | Yes | 0.00015 | 5M |
| C | No | 0.00005 | 5M |
| D | Yes | 0.00005 | 5M |

Keep curriculum, reward, PPO epochs, discounting, batch size and architecture fixed. Preserve parent optimizer state consistently in all arms; collect fresh rollouts under each arm's distribution. Set the *effective* LR explicitly so the inherited schedule cannot halve it again unintentionally. Record parent checkpoint, hashes, seed, overrides and state-restoration/reset policy in each new run directory. These forks need explicit support; ordinary unchanged continuation is not this experiment.

Mask sampling, old and recomputed action probabilities, and entropy consistently. Store the pre-action mask. Exclude padding from losses and reductions, use a finite dummy distribution for padded samples, test one-legal-action states, and reset terminal states before sampling.

Compare fixed-budget endpoints with the parent on common natural games; report selected-best checkpoints separately. Report mean, median, tile reach rates, uncertainty from paired per-game results, valid moves, wall time and attempted transitions. Four forks of one parent are not four independent training replications.

A useful initial promotion target is approximately 20% higher mean score with supportive paired uncertainty and no clear median regression. This is a chosen practical threshold, not a theorem or guaranteed attainable improvement. Replicate a promising treatment with additional training seeds before assigning a much larger budget. Twenty million pilot transitions would take roughly an hour at the present rate, before engineering and evaluation overhead; measure masked throughput rather than assuming it stays equal.

## 5. Subsequent directions

- **Reward ablation:** if masking is sound, compare the selected masked condition with positional shaping disabled, changing only that reward component. Allow for value-function adaptation when interpreting early results.
- **Pretrained branch:** adopt the reviewer's suggestion to import the released weights directly for an identical architecture. This is simpler than training the same architecture to imitate those weights. Verify output parity, preserve the untouched reference, calibrate the critic for our reward, then use cautious RL updates. Label this as pretrained fine-tuning, distinct from our random-initialization research. It has not been launched.
- **Transformers:** none are being trained in this project. Defer them until the training audit and first ablation results. Attention is not automatically long-horizon planning. Transformer work exists for 2048 (for example [2048former](https://huggingface.co/jkeisling/2048former-50m-v0.1)); this does not establish comparable Threes results, and we should not claim novelty from an incomplete literature search.

The next decision should follow measured ablations. New validation records are encouraging, but neither those records nor architectural similarity justify promising the published level of play from more time alone.
