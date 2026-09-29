# 2026-09-25_182257_ppo_20k_initial — first MaskablePPO trial

Historical trial predating automatic run tracking. [Final model](models/model.zip) and [provenance](metadata/provenance.json). The full configuration, detailed metrics, and exact code for this run were not archived; the numbers below come from the contemporary log.

- Variant: bonus preview windows of up to three values; 12-card ordinary bag.
- Historical command: `.\.venv\Scripts\python.exe train.py --steps 20000 --seed 42 --eval-games 100 --save results\2026-09-25_ppo_20k.zip`; this path belongs to the old repository layout.
- Network: default `MlpPolicy` from `sb3-contrib` 2.9.0; `n_steps=128`, `batch_size=64`, `n_epochs=2`, CPU.
- Reward: `log1p(official score gain)`; invalid moves masked.
- Transitions actually collected: 20,096, because PPO completes its last batch of 128.
- Throughput after observation caching: 2,124 transitions/s on the local machine; training time 9.46 s. The earlier uncached measurement was 10.91 s for the same number of transitions.
- Deterministic evaluation on seeds 100,042–100,141: mean score 344.9, median 261.0.
- References on the same starting seeds: random 284.3; merge 544.8; positional 1,102.0; spawn 1,519.0. The policies then encounter different games.
- Model saved here as `models/model.zip`. It was produced before the observation-cache optimization, which preserved the same 344.9 result in another run with the same seed.

The trial demonstrates that the engine connects to a network with invalid-move masking. It is too short to support conclusions about eventual performance or hyperparameter choices. No long training or hyperparameter search was run.
