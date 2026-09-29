# Two 200,000-transition PPO training runs

On September 26, 2026, two independent runs used the same `configs/ppo_200k.json` configuration except for the training seed (42 or 43). Each collected 200,064 transitions because PPO completed its final batch of 128. Training-time evaluations used the same 100 seeds, from 100,042 through 100,141, about every 8,192 steps. Checkpoints, metrics, frozen configurations, and HTML reports are in [`2026-09-26_115920_ppo_200k_s42`](../../runs/2026-09-26_115920_ppo_200k_s42/FICHE.md) and [`2026-09-26_120133_ppo_200k_s43`](../../runs/2026-09-26_120133_ppo_200k_s43/FICHE.md).

| Run | Duration | Best training-time mean | Final mean | Final median |
| --- | ---: | ---: | ---: | ---: |
| Seed 42 | 125.6 s | 495.6 at 172,160 steps | 495.6 | 379.5 |
| Seed 43 | 127.1 s | 558.2 at 16,512 steps | 523.7 | 402.0 |

Seed 43's best intermediate evaluation did not persist. Scores did not improve monotonically. The `model.zip` files contain the **final** models, not the best intermediate checkpoints.

## Separate evaluation

The final models and four reference agents played on 300 held-out seeds, from 200,000 through 200,299. These seeds were used neither for training nor for periodic tracking. Individual scores are in `ppo_200k_models_holdout.csv` and `ppo_200k_baselines_holdout.csv` in this directory.

| Agent | Mean | Median | Maximum |
| --- | ---: | ---: | ---: |
| Random | 297.6 | 247.5 | 1,155 |
| PPO seed 42 | 411.4 | 330.0 | 1,875 |
| PPO seed 43 | 532.5 | 405.0 | 3,390 |
| Immediate merge | 561.5 | 430.5 | 2,805 |
| Position | 971.4 | 892.5 | 3,810 |
| Expected spawn | 1,378.5 | 1,132.5 | 7,404 |

The agents start with the same seeds, but their choices lead to different trajectories. The seed-43 network beats random play but remains below the position and expected-spawn heuristics. Two training seeds are insufficient to estimate this method's variance precisely. At the end of both runs, the value head's explained variance is near zero and update KL is very small. These diagnostics suggest examining the reward and update strength before simply increasing the transition budget.

## Reproduction

From the project root, reevaluate the archived historical models:

```powershell
.\.venv\Scripts\python.exe src/evaluate_models.py results\runs\2026-09-26_115920_ppo_200k_s42\models\model.zip results\runs\2026-09-26_120133_ppo_200k_s43\models\model.zip --games 300 --seed 200000 --csv results\comparisons\2026-09-26_ppo_200k\reevaluation_models_holdout.csv
.\.venv\Scripts\python.exe src/compare_agents.py --games 300 --seed 200000 --csv results\comparisons\2026-09-26_ppo_200k\reevaluation_baselines_holdout.csv
```

To launch new runs with these settings in a visible terminal:

```powershell
.\run_experiments.ps1 -Series ppo_200k_replication -Seeds 42,43 -Steps 200000 -Config configs\ppo_200k.json
```

New runs receive timestamped names and archive a copy of the **current code**. The exact code for the two historical runs in this report was not archived. Use `src/train.py --resume` with a run directory to continue its weights and optimizer state.
