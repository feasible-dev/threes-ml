---
status: reference
updated: 2026-09-26T21:29:45-04:00
scope: Reproducing the published recurrent Threes agent
---

# Reproducing the published recurrent Threes agent

## Status and evidence

We have a trainable reconstruction of the **released ONNX architecture**, a native batched simulator, GPU training, recurrent checkpoint continuation, and visible experiment scripts. New training starts from random weights. Published weights are imported only for numerical verification and separately labelled reference evaluation.

**Architecture equivalence is verified. Published playing strength has not been reproduced by our training.** The final training recipe is not fully available. More training time alone cannot be promised to recover the reported result.

The [current status](CURRENT_STATUS.md) owns experiment selection and progress snapshots. The completed [initial recurrent pilot](../results/comparisons/2026-09-26_153305_recurrent/comparison.md) used an earlier engine revision; its results are historical. The [fidelity audit](FIDELITY_AUDIT.md) records the corrections and limits of equivalence.

Primary sources:

- [Josiah Kiok's article](https://medium.com/@josiah-kiok/beating-threes-with-reinforcement-learning-ae074dd28a68)
- [Released agent and simulator](https://github.com/pseudonam-gc/threes-web), audited commit `9527295de72333c91592fff45e8ccbcabd6ba80d`
- Local ONNX SHA-256: `63bff54b23de09801fb2bc77ca8856ed781c2abaa2efe896c3da15e5d810eff2`
- [Official PyTorch installation](https://pytorch.org/get-started/locally/)

The repository reports 21.5% of 10,000 games reaching 6,144 for a later agent. Whether the distributed weights are exactly that agent remains an open provenance question. The article describes hundreds of millions of transitions; the repository describes billions overall.

| Component | Evidence / implementation | Status |
| --- | --- | --- |
| Network | All ONNX weights map to our PyTorch graph; logits, value, hidden and cell states checked over recurrent steps | Verified numerically on CPU and RTX 3060 |
| Observation | 16 board ranks, preview rank, three bag counts, remaining total | 21 inputs verified against graph and C source; README's 24-input description is stale |
| Moves, bag, scheduled bonuses, rewards | Independent Numba implementation based on source audit | 10,000 controlled-randomness C/Python transition checks passed; reset fixes remain deliberate differences |
| Optimizer | SB3 recurrent PPO with shared actor/critic LSTM gradients | Original uses PufferLib; implementation differs |
| PPO hyperparameters | Explicit local configurations | Final published values unavailable |
| Curriculum / reward scaler / snake coefficient | V2 tests scaffolding probability 0.5 versus 0; explicit scale 1 and snake coefficient 0 | Final original recipe unavailable |
| Performance | Requires our own training and evaluation | Not yet reproduced |

## Network: 13,995,453 trainable parameters

Ranks encode `0,1,2,3,6,12,...` as `0,1,2,3,4,5,...`.

```text
16 board ranks + preview rank
    -> shared embedding (18 categories x 8)
    + learned position embeddings (17 x 8)
    -> flatten: 136 features
3 bag counts + total -> Linear(4,32) -> GELU
concatenate: 168
    -> Linear(168,2048) -> GELU
    -> Linear(2048,1024) -> GELU
    -> Linear(1024,1024) -> GELU
    -> shared LSTM(1024), retaining hidden and cell state between actions
       -> actor: Linear(1024,1024) -> GELU -> Linear(1024,4)
       -> value: Linear(1024,1024) -> GELU -> Linear(1024,1)
```

Action order: up, down, left, right. Memory resets between games. Both actor and critic losses train the shared encoder and LSTM; SB3's default shared-LSTM critic detaches gradients, which our policy overrides.

Training samples all four actions, including invalid moves. Evaluation chooses the legal action with the largest logit, matching the web agent. Bag counts assume knowledge of bag composition and draw history; they cannot generally be inferred from a single board. The benchmark supplies the same information as the released interface.

The current terminal displays `invalid recent`, measured over the last PPO rollout. The original pair displayed a cumulative rate; both are saved in v2 metrics. Invalid moves receive `-0.05`. HTML also shows the curriculum action share and actual learning rate. Natural training-score summaries exclude artificial curriculum games.

## Rules and rewards

The reproduction engine is separate from the historical game variant. It schedules bonus opportunities with a random phase per 21 valid moves, rather than independent 1/21 draws. Nine ordinary tiles initialize the board from a 4/4/4 bag. The preview supplies the lowest rank of a possible bonus window; the actual bonus stays hidden.

Base reward sums `0.0625 * r` for equal-rank merges of rank `r >= 3`; `1+2` contributes `0.1875`. Invalid moves receive `-0.05`; game over overrides the reward with `-1`. Rewards are clipped to `[-1,1]`. The shaped variant adds top-row/corner and monotonicity terms. The completed shaped pilot had the duplicate-largest-tile corner error described above; new runs use the corrected calculation. The extra snake coefficient remains zero because its final published value is unknown.

Known differences:

- Our per-episode RNG does not reproduce C `rand()` trajectories for the same seed.
- Optional scaffolding fixes inconsistent empty-cell bookkeeping and preview initialization in the source. It was disabled in the first pair; the v2 batch compares probabilities 0 and 0.5. When enabled, some episodes start with 3–6 larger tiles based on that environment's previous maximum. Evaluation always uses natural starts.
- Terminal rank 16 means tile **24,576**. Upstream calls it `65536`, inconsistent with its own value table. Our historical GUI ends at 12,288.
- Revision 2 caches the audited time limit until a valid move and disables timeout bootstrapping by default, matching the C terminal signal. The first pair used SB3 timeout bootstrapping. Evaluation records timeouts and caps games at 20,000 actions. Investigate any truncations before final claims.
- Optional upstream endgame/sparse-reward modes are not reproduced; their use in the checkpoint is undocumented.
- Score is the tile-score sum, equivalent to ordinary C merge/spawn accounting. Do not pool historical variant scores with this benchmark.

## Configurations and experiment history

The [v2 reference](REPRODUCTION_V2.md) describes the initial three-condition 10M batch at seed 72. The [current status](CURRENT_STATUS.md) records the subsequent 100M continuations and accepted next experiments. These are explicit local choices, not recovered final author settings.

`.venv-repro` contains CUDA PyTorch; `.venv` serves the CPU viewer and historical pipeline. New runs freeze configuration, source hashes, episode seeds, metrics, HTML reports and complete update-boundary checkpoints. Revision-1 runs cannot resume under current engine code; their original sources remain in their run directories.

Core PPO settings: 128 environments x 64 steps = 8,192 transitions per rollout; minibatch 2,048; up to four epochs; initial LR 0.0003; gamma 0.999; GAE lambda 0.95; entropy coefficient 0.01; value coefficient 0.5; PPO clip 0.2; gradient norm limit 0.5; target KL 0.03. The decay condition halves LR once at absolute transition 5M. Inspect the run configuration and metrics for its actual schedule and effective LR.

## Reference evaluation and monitoring

```powershell
# Downloaded reference only; no training.
.\.venv-repro\Scripts\python.exe -u src/evaluate_reproduction.py --published --games 300 --seed 800000

# Numerical equivalence and simulator throughput evidence.
.\.venv-repro\Scripts\python.exe src/validate_reproduction.py

# Live charts from a second terminal.
.\.venv-repro\Scripts\python.exe src/dashboard.py results\runs\YOUR_RUN_DIRECTORY
```

Periodic validation: 700000–700099. Batch comparisons: 800000 onward. Engineering checks: 710000 onward. Training uses the disjoint upper half of uint64 seeds. Keep final seeds 900000–909999 untouched until selection ends. See [seed protocol](SEED_PROTOCOL.md).

PPO loss need not decrease monotonically: the policy changes its data distribution, value targets change, and entropy is deliberately encouraged. Judge playing strength by independent games; use loss, KL, and entropy to diagnose optimization.

## Code and artifacts

`src/reproduction/` separates `policy.py`, `environment.py`, `training.py`, `evaluation.py`, and `artifacts.py`. Command entry points are `train_reproduction.py`, `evaluate_reproduction.py`, and `validate_reproduction.py`. Current settings are in `configs/reproduction_v2_*.json`; earlier base/shaped files describe the initial pilot conditions.

Run reviews live in `metadata/review.json`. Performance, scientific usefulness, and retention are separate decisions. No files are deleted automatically.
