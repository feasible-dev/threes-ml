---
status: snapshot
updated: 2026-09-26T22:01:00-04:00
scope: Read-only recurrent PPO and action-handling audit at the interrupted decay-run checkpoint
---

# Recurrent checkpoint audit

This is measured engineering evidence, not a new training run or a change to the accepted research decision. The parent is `2026-09-26_162446_recurrent_v2_curriculum_decay_s72`, with a complete checkpoint at **74,235,904 attempted transitions**. Its continuation stopped before the planned 100,007,936-transition endpoint. The core environment, policy, trainer and evaluation source hashes match the last continuation snapshot. The audit used the checkpoint's weights, recurrent state, active boards and configuration in a private in-memory environment. It collected one fresh 8,192-transition rollout and made **no optimizer update**. Results are in [diagnostics.json](diagnostics.json).

## Findings

| Check | Measured result | Interpretation |
| --- | --- | --- |
| Recurrent PPO reconstruction | 8,192 real samples, 3,520 padded positions, 51 episode resets; mean ratio 1.00000018; pre-update approximate KL 3.49e-10 | No material probability-reconstruction defect in this sampled rollout. Padding log probabilities stayed finite. |
| Numerical difference | Mean absolute action log-probability difference 2.16e-6; maximum 0.00237; 26 of 8,192 differences exceeded 1e-4 | Small outliers remain. This check does not prove exact equality on all states or at all rollout boundaries. |
| Invalid actions in the fresh rollout | Mean invalid probability mass 41.99%; sampled invalid attempts 42.02% | Invalid actions consume a substantial part of the training interaction budget at this checkpoint. |
| First epoch, four minibatches, before update | Median weighted-critic-to-actor gradient norm ratio 9.71 in encoder, 6.85 in shared LSTM; median implied global clipping multiplier 0.0306 | Critic gradients and global clipping are material optimization suspects. Four correlated minibatches from one rollout do not establish that the critic causes poor scores. |
| Paired greedy natural games, seeds 700100-700131 | Unmasked mean 356.06, median 60, 31/32 timeouts; legal-filtered mean 2,727, median 1,957.5, 0 timeouts | The current model's greedy evaluation depends strongly on legal filtering. This **does not** estimate the effect of training with action masks. |
| Training reward on the 32 legal-filtered natural games | Mean undiscounted return 31.40; mean discounted return 28.82; score correlations 0.942 and 0.931 respectively; mean preclip merge reward 31.16 and shaping reward 1.43; 7 positive rewards clipped | The current policy's realized reward broadly tracks score on this small sample. Small aggregate shaping does not rule out local incentive effects during learning. |
| Training episode log at interruption | 45 recorded timeouts in 638,081 completed episodes (26 natural, 19 curriculum) | Artificial timeouts are rare in this history. This log extends past the complete checkpoint and is not checkpoint-aligned. |

The paired greedy check uses development seeds and the latest saved checkpoint, not the `best.zip` selected by validation. It compares inference action rules for the same weights and seeds. The fresh rollout samples the unmasked training policy. The gradient measurements use the current PPO losses, including the configured 0.5 value coefficient and 0.01 entropy coefficient for the combined norm; the actor and critic norms isolate their respective losses. No parameter was changed.

## Remaining audit and experiment work

The sampled rollout contained no one-legal-action state; test that case explicitly when implementing masked training. Reward components were measured only on the small natural-game inference check; training natural/curriculum components and reward clipping frequency remain to be instrumented. It does not measure a masked training update, so the planned masking by learning-rate forks still need implementation and an equal-parent pilot. A fixed-budget endpoint and the queued constant-LR comparison remain part of the [accepted decision](../../../docs/RESEARCH_DECISION_2026-09-26.md).

A read-only continuation preview found no source compatibility issue. It calculated **25,772,032 additional transitions** from the last complete checkpoint to the allocated 100,007,936 endpoint. The metrics file reached 75,128,832 before interruption, but those later weights are not a complete saved boundary. This audit did not resume training.

## Frozen history investigation during the resumed run

The resumed continuation started at 2026-09-26T21:48:15-04:00. This analysis used its immutable [pre-resume copy](../../runs/2026-09-26_162446_recurrent_v2_curriculum_decay_s72/metadata/continuations/2026-09-26_214814_973683_from_74235904/before/), not the live log. The [analysis output](history.json) records input hashes and exact window boundaries; [analyze_history.py](analyze_history.py) reproduces it. Its metrics extend to 75,128,832 logged transitions and 638,081 completed episodes before interruption. These observations include work after the last complete checkpoint and are distinct from the checkpoint-based audit above.

| Transition window | Recent invalid attempts | Natural mean score / return / valid moves | Curriculum mean score / return |
| --- | ---: | ---: | ---: |
| 0–10M | 29.4% | 652 / 10.77 / 66.7 | 6,455 / 3.97 |
| 20–30M | 34.6% | 1,846 / 21.51 / 120.0 | 22,235 / 6.50 |
| 60–70M | 30.7% | 2,617 / 27.19 / 144.2 | 35,725 / 7.94 |
| 70–75.13M | 29.7% | 2,609 / 27.34 / 143.9 | 39,338 / 7.94 |

The invalid rate improved from its 20–30M peak but remained high. In the final logged window, natural games averaged 62.0 invalid attempts on top of 143.9 valid moves. The single-rollout checkpoint audit's 42.0% invalid rate agrees with the adjacent logged update's 43.0%; the broader 70–75M average hides substantial update-to-update variation. Natural score, return and moves were nearly flat between the last two windows. Curriculum scores continued rising while curriculum return and moves were nearly flat; raw curriculum score includes its evolving artificial start and cannot be read as additional natural-game progress. These windows contain correlated updates and episodes from one training trajectory.

For a fixed-budget paired check, the validation files nearest 60M and 75M used the same 100 seeds, 700000–700099. Their means were 2,699.70 at 59,785,216 transitions and 2,304.93 at 75,022,336; the paired mean difference was -394.77, with 50 wins and 50 losses and a paired bootstrap 95% interval of **[-944.34, 134.97]**. This does not establish regression. The 75M evaluation was logged after the resumable checkpoint, so its weights were not preserved as an endpoint model. Reused validation seeds and a single training seed limit inference about generalization or reproducibility.

A focused CPU test now reconstructs stored rollout probabilities through the actual recurrent minibatch path with a mid-rollout episode reset and padding. It passed with a maximum error below 2e-5. A separate engine fixture supplies a state with exactly one legal move for the future masked-policy test. Neither test changes the active trainer or demonstrates correct masked PPO updates.

## CPU-only invalid-streak probe

The [probe output](attempt_probe.json) follows the first 16 saved environments for 256 further time steps each: 4,096 **fresh stochastic attempts** from the 74,235,904-transition checkpoint. It runs entirely on CPU with an in-memory simulator and policy. The action calculation matched the simulator's invalid flag on every attempt. [probe_attempts.py](probe_attempts.py) is the reproducible source; this is a small selected subset of active boards, not a population estimate.

| Sampled episode type | Attempts | Invalid attempts / probability mass | Longest completed invalid streak | Reward components before clipping |
| --- | ---: | ---: | ---: | ---: |
| Natural | 3,097 | 43.1% / 43.2% | 86 | Merge 362.69, shaping 17.33 |
| Curriculum | 999 | 24.9% / 24.3% | 50 | Merge 142.63, shaping 10.91 |

The most concerning natural-game segment lasted **at least 224 consecutive invalid attempts** and was still ongoing at the probe boundary. Its board allowed down and left, while the policy sampled invalid up 219 times and invalid right 5 times. The probability of up was 96.8% at the first observed invalid attempt and 96.0% at the last. The board stayed unchanged during those attempts, although LSTM memory continued to evolve. The segment began inside this probe, so it is not left-censored; its final length is unknown. The longest fully observed natural streak was 86 attempts. These observations show a concrete repeated-action trap at this checkpoint, not its prevalence across training.

Only two positive rewards clipped across the 4,096 attempts. No one-legal-action state appeared in this small probe, so the crafted one-legal fixture remains necessary for the masked path. Natural/curriculum reward totals here are preclip components from this probe; they do not replace instrumentation over the full training distribution.

```powershell
.\.venv-repro\Scripts\python.exe results/reviews/2026-09-26_recurrent_audit/probe_attempts.py results/runs/2026-09-26_162446_recurrent_v2_curriculum_decay_s72 --output results/reviews/2026-09-26_recurrent_audit/attempt_probe.json
```

Reproduce from the repository root with:

```powershell
.\.venv-repro\Scripts\python.exe src/audit_recurrent_checkpoint.py results/runs/2026-09-26_162446_recurrent_v2_curriculum_decay_s72 --output results/reviews/2026-09-26_recurrent_audit/diagnostics.json
```

The script reads the latest complete checkpoint by default. If the parent run later resumes and saves another checkpoint, this command audits that newer boundary and overwrites the JSON; preserve this snapshot before rerunning.
