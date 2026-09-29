---
status: reference
updated: 2026-09-26T21:29:45-04:00
scope: Engine revision 2 - initial experiment design and validation
---

# Engine revision 2: initial experiment design and validation

This is a reference for the initial 10M batch and available commands. Read [current status](CURRENT_STATUS.md) before allocating another experiment.

## What changed

The network still reproduces the released 13,995,453-parameter ONNX graph. New agents start from random weights. This batch tests a corrected simulator and explicit training interventions; it does not claim to recover the unpublished final training recipe.

Engine revision 2 corrects duplicate-maximum corner shaping, matches the C bonus draw call order and merge/spawn/draw ordering, caches the time limit until a valid move, and only advances the curriculum frontier after natural games. Reward scaling is explicit. Timeouts end the value target by default, matching the terminal flag exposed by the C training interface. Optional timeout bootstrapping remains configurable.

The [C differential check](../results/references/2026-09-26_engine_v2_c_differential.json) compares 10,000 transitions under shared random draws: board, deck, preview, metadata, RNG state, clipped reward, and termination. It includes natural short trajectories and injected high-rank states. The pinned upstream simulation functions are compiled locally, rendering is omitted, and automatic reset is intercepted to inspect terminal states. Reset initialization is tested independently because our curriculum deliberately fixes upstream empty-count and stale-preview/max/score problems. This is bounded test evidence, not exhaustive equivalence.

The curriculum starts selected games with 3–6 larger tiles near each environment's best **natural-game** rank. It activates only after that frontier reaches rank 6 (tile 24). A probability of 0.5 chooses curriculum at eligible resets; the fraction of transitions can differ because game lengths differ. Artificial games do not move the frontier upward. Every validation game starts naturally with nine ordinary tiles.

## Launch from the project root

```powershell
.\run_reproduction_v2.ps1 -Preview
.\run_reproduction_v2.ps1
```

Default: three sequential runs, seed 72, 10 million transitions each, CUDA on the RTX 3060, 128 environments inside each learner. At the previous roughly 5,000 transitions/s, plan about **1 hour 40 minutes plus comparison/checkpoint overhead**. Curriculum can lengthen games and change throughput. PowerShell stays visible; this chat remains available. No training process is launched in the background by the assistant.

| Variant | Structural rewards | Curriculum probability | Learning rate |
| --- | --- | ---: | --- |
| `control` | Corrected C terms | 0 | 0.0003 constant |
| `curriculum` | Same | 0.5 when eligible | 0.0003 constant |
| `curriculum_decay` | Same | 0.5 when eligible | 0.0003, halved at 5 million transitions |
| `decay` (optional) | Same | 0 | Same halving schedule |

The default comparisons isolate curriculum and then learning-rate decay conditional on curriculum. Adding `decay` completes the two-by-two design. The 50% mixture and 5-million-step milestone are **local experimental choices**. The article describes reductions after plateaus, but its final schedule and curriculum ratio are not available. Snake coefficient stays 0, reward scale 1. PPO keeps 64-step rollouts, batch 2,048, four epochs, gamma 0.999, GAE 0.95, entropy 0.01, clipping 0.2, value coefficient 0.5, gradient norm 0.5, target KL 0.03.

For two training seeds, roughly double the time:

```powershell
.\run_reproduction_v2.ps1 -Seeds 72,73
```

For a shorter first batch, compare the first two conditions; a run shorter than 5 million transitions does not test the decay intervention:

```powershell
.\run_reproduction_v2.ps1 -Variants control,curriculum -Steps 5000000
```

Budgets round up to complete 8,192-transition rollouts. Learning-rate milestones use the total transition count and do not restart on resume.

## Read the results

Periodic validation uses 100 natural games at seeds 700000–700099 every approximately 500,000 transitions, plus the first and final updates. Best checkpoint selection uses validation mean. The batch compares selected checkpoints and the downloaded reference on 300 natural games at seeds 800000–800299. Final seeds 900000–909999 remain unused. Engine revision 2 changes seeded trajectories, so compare policies together under the new protocol; the old reference's 64/300 result is historical evidence.

The terminal shows recent invalid-action rate, actual learning rate, curriculum transition share, score, and tile-384/3072/6144 rates. HTML adds recent-action and learning-rate curves. Natural training scores exclude curriculum games; per-episode CSV retains the curriculum flag. Configuration, source snapshots, engine revision, checkpoint state, and comparison protocol are recorded per run.

The initial batch is complete. Its original research gates have been superseded by the [accepted decision](RESEARCH_DECISION_2026-09-26.md). A single training seed remains preliminary. Loss need not decrease monotonically because PPO changes its policy, sampled states and value targets.

## Resume and compatibility

Use the [continuation launcher](CONTINUING_RUNS.md) for a preview, active-process checks, terminal log and per-session history. Wait until the current batch finishes:

```powershell
.\resume_reproduction.ps1 -List
.\resume_reproduction.ps1 -Run results\runs\YOUR_V2_RUN_DIRECTORY -AdditionalSteps 10000000 -Preview
.\resume_reproduction.ps1 -Run results\runs\YOUR_V2_RUN_DIRECTORY -AdditionalSteps 10000000
```

The lower-level trainer entry point remains available:

```powershell
.\.venv-repro\Scripts\python.exe -u src/train_reproduction.py --resume results\runs\YOUR_V2_RUN_DIRECTORY --steps 10000000
```

`--steps` means additional transitions. Recurrent memory, optimizer, simulator, RNG, absolute learning-rate schedule, and episode log boundary resume together. CPU tests compare the next update exactly, including a schedule milestone. GPU arithmetic can differ across software/hardware versions.

Revision-1 checkpoints are rejected for training continuation under the revised engine. Their saved weights remain evaluable. Historical continuation requires the archived source and original dependencies. This prevents silently combining two rule versions in one learning curve.

## Rerun the engine audit

The local header is pinned to upstream commit `9527295de72333c91592fff45e8ccbcabd6ba80d` and its SHA-256. Zig is only an engineering compiler dependency; normal training does not require it.

```powershell
.\.venv-repro\Scripts\python.exe -m pip install ziglang==0.14.1
$env:PYTHONPATH = 'src'
.\.venv-repro\Scripts\python.exe -m reproduction.c_oracle --cases 10000 --output results\references\engine_v2_c_check.json
.\.venv-repro\Scripts\python.exe -m unittest discover -s tests -p test_reproduction.py
```

The full original trainer, final hyperparameters, initialization choices, curriculum schedule, and optional endgame/sparse mode settings remain unavailable. We still use SB3 recurrent PPO rather than the author's PufferLib implementation. See the [fidelity audit](FIDELITY_AUDIT.md).
