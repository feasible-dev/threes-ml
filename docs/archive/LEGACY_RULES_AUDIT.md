---
status: historical
updated: 2026-09-26T21:29:45-04:00
scope: Rules and throughput audit — September 25, 2026
---

# Rules and throughput audit — September 25, 2026

> Historical stage only. Earlier recommendations and agent instructions are superseded by [current status](../CURRENT_STATUS.md).

This is an audit of the historical local GUI simulator. The separate recurrent reproduction simulator and its known differences are described in [REPRODUCTION.md](../REPRODUCTION.md).

## Conclusion

The engine reproduces several observed Threes! rules, including the bag of four `1` tiles, four `2` tiles, and four `3` tiles. It cannot yet be described as an exact reproduction of a specific commercial version. The current variant displays the possible values of the next bonus tile, as described in the guide revised in 2024.

## Verified points

| Mechanism | Project status | Public evidence and limitation |
| --- | --- | --- |
| One-cell movement, `1+2` and equal-pair merges, invalid moves | Implemented in `threes.py` | [Official Xbox manual](https://dlassets-ssl.xboxlive.com/public/content/3201c2f3-2726-4c73-91b3-a405ef9753d5/GameManual/016b7eb4-0333-422e-8647-d5e4625a4649/en-CA/index.html). The manual does not describe every line-processing priority case. |
| Ordinary bag, four tiles of each value, nine starting tiles | Implemented | [Account by the tile-draw analysis author](https://toucharcade.com/community/threads/threes-by-sirvo-llc.218248/page-27) and [revised guide](https://steamcommunity.com/sharedfiles/filedetails/?id=3155431026). The bag creates regularity: draws are not independent. |
| Spawn on the opposite edge of a changed line, chosen uniformly | Implemented in `spawn_positions()` and `Game.step()` | [Revised guide by the analysis author](https://steamcommunity.com/sharedfiles/filedetails/?id=3155431026), “Card Placement” section. |
| Bonus probability `1/21` once a `48` tile exists; pool from `6` through `maximum/8` | Implemented | [Hung Guei thesis, §5.2.1.2](https://arxiv.org/pdf/2212.11087) and [original tile-draw analysis](https://toucharcade.com/community/threads/threes-by-sirvo-llc.218248/page-27). These sources do not establish identity with every current version. |
| Bonus preview and value selection in recent versions | Implemented following the [2024 revised guide](https://steamcommunity.com/sharedfiles/filedetails/?id=3155431026): choose an equally likely window of up to three consecutive values, then draw uniformly within it | A [2018 game analysis](https://threes.danielschenker.ch/) reports the reverse order (value first, preview second), which gives different marginal probabilities. No official source code found settles these descriptions definitively. |

The project also omits the bonus-tile start mentioned in the revised guide. It does not reproduce the proprietary random-number generator or the same sequences for the “same seed” as the official app. A precise target platform and version must be chosen before claiming complete fidelity.

## Implications for the agent

The bag balances `1` and `2` tiles in the long run: each complete cycle adds four of each. A `1+2` merge also removes one of each. `Game.visible_ordinary_counts` records the counts of cards already revealed, including the current ordinary preview, without exposing the hidden order of the bag. Existing heuristics do not yet read these counts; a learned agent could receive them explicitly as card-tracking information available to a player.

## Local simulator measurement

After removing a redundant legal-move calculation in `random_agent.py`, 1,000 random games (43,638 moves) took 1.853 s: about **23,546 moves/s** in this Windows/Python 3.11 environment, without learning. Profiling before the change placed most of the cost in `move()` and its repeated validations. Instrumented profiling slows execution, so its timings are not throughput measurements.

At this indicative rate, 10 million transitions would require about seven minutes of simulation alone, and 100 million about 71 minutes, on this process and machine. Policy computation, collection, and network updates would add to that. Keep Python for orchestration and an initial learning prototype; measure the full loop before choosing an engine acceleration. A C++ port becomes useful if simulation actually dominates. Any port must be checked against the reference engine using controlled states and draw sequences.

The machine running these commands has an NVIDIA GeForce RTX 3060 with 12,288 MiB according to `nvidia-smi`. PyTorch, NumPy, Gymnasium, and `sb3-contrib` were installed in `.venv` for a CPU prototype. The 20,096-transition MaskablePPO loop reached about 2,124 transitions/s after observation caching, versus more than 24,000 moves/s for the random game alone. This first end-to-end measurement therefore does not justify a C++ engine port yet. See the [first trial card](../../results/runs/2026-09-25_182257_ppo_20k_initial/FICHE.md).
