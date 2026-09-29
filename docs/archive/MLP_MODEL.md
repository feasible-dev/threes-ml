---
status: historical
updated: 2026-09-26T21:29:45-04:00
scope: Historical 35-input neural agent
---

# Historical 35-input neural agent

> Historical stage only. Earlier recommendations and agent instructions are superseded by [current status](../CURRENT_STATUS.md).

This document describes completed early PPO runs, notably `2026-09-26_115920_ppo_200k_s42` and `2026-09-26_120133_ppo_200k_s43`. The current GELU/LSTM experiment is in [REPRODUCTION.md](../REPRODUCTION.md). Each run's exact settings are frozen under `results/runs/`.

## What the network sees

The input is a vector of **35 numbers**:

| Positions | Information | Encoding |
| --- | --- | --- |
| 1–16 | The 16 board cells | Tile rank divided by 15; 0 for an empty cell |
| 17–32 | Announced next tile | 16 indicators; several may be 1 for a bonus hint |
| 33–35 | Ordinary cards still deducible from the bag | Three counts for tiles 1, 2, and 3, divided by 4 |

The legal moves are calculated from the board and supplied to the action mask **separately** from these 35 numbers. The network sees neither the bag's hidden order, the next spawn position, nor the random generator's state.

## Architecture

The policy and value estimate use separate fully connected networks. Each receives the 35 inputs and has two hidden layers of 64 neurons with `tanh` activation:

```text
                   ┌─ 64 tanh ─ 64 tanh ─ 4 action scores
35 numbers ────────┤
                   └─ 64 tanh ─ 64 tanh ─ 1 estimated value
```

The four action scores correspond to up, down, left, and right. After impossible moves are masked, the training policy samples from an action probability distribution. Evaluation chooses the highest-probability action. The other output estimates discounted future return from the current observation. Together, the networks have **13,253 trainable parameters**. This agent has no convolution, recurrent memory, Transformer, or move search.

## What is learned

This **is reinforcement learning**: `MaskablePPO` plays in the simulator, receives a reward after each action, and updates the weights of both networks. It does not imitate our heuristics.

For a legal move, these runs used this reward:

```text
r = ln(1 + score_after − score_before)
```

A merge that does not change the score yields 0; a gain of 3 yields `ln(4)`. The logarithm compresses large gains. An illegal move would yield −1, but the mask normally prevents it. There is **no explicit bonus** for survival, free cells, a large tile, or winning, and no explicit terminal penalty. The **official final score** is used for evaluation, not directly as the training reward. Summing the logarithms of individual gains does not produce the final score; this may help explain why PPO could optimize behavior that differs from our performance goal.

## How PPO updates the weights

With the 200,000-transition run configuration:

- The network collects **128 transitions**, then makes an update.
- It traverses them **twice** in batches of **64**: four optimization steps per update.
- Future rewards are discounted with `γ = 0.99`; advantage estimation uses `λ = 0.95` (GAE).
- The learning rate is `0.0003`. PPO limits policy changes with a **0.2** clipping threshold and limits the gradient norm to **0.5**.
- The total loss combines the PPO policy objective, squared value error weighted by **0.5**, and an entropy term weighted by **0**. Entropy is measured, but adds no explicit exploration bonus to the loss.

More precisely, an *advantage* compares observed return with what the value branch predicted. If `ρ` is the ratio of new to old action probability, PPO approximately minimizes:

```text
loss = −mean[min(ρ·advantage, clip(ρ, 0.8, 1.2)·advantage)]
       + 0.5·mean[(predicted value − target return)²]
       − 0·entropy
```

Advantages are normalized before the update. Clipping limits how much one update can change action probabilities. This formula omits details of batching, action masks, and GAE.

In short, the policy branch learns to increase the probability of actions whose outcomes exceeded expectations while limiting each update. The value branch learns to predict future return; its error helps compute that comparison. Predicted value is not the board's official score.

These historical runs trained on **CPU**. Periodic evaluations played the same reserved seeds and displayed mean official score. The dashboard tracked losses, approximate KL, and explained variance. A lower loss did not necessarily mean a better game score, which is why we kept separate evaluation games.

## Where to look

- [Experiment catalog](../../results/EXPERIENCES.md): compare runs.
- Each run's `FICHE.md` summarizes parameters, results, and reasons to keep or trim the experiment.
- `metadata/config.json` stores chosen settings; `metadata/method.json` stores other effective details. `data/metrics.csv` and `report/report.html` show learning progress.
- The implementation is in [observation.py](../../src/observation.py), [rl_env.py](../../src/rl_env.py), and [train.py](../../src/train.py).
