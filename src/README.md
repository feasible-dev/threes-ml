---
status: reference
updated: 2026-09-27T19:33:00-04:00
scope: Source map
---

# Source map

The [current status](../docs/CURRENT_STATUS.md) records what is being researched. This map separates the active recurrent pipeline from the retained historical game.

| Area | Responsibilities |
| --- | --- |
| `reproduction/environment.py` | Audited v2 simulator, reward, curriculum, batched state |
| `reproduction/policy.py` | ONNX-compatible encoder, shared LSTM and actor/value heads |
| `reproduction/training.py` | Recurrent PPO, learning schedule, checkpoint/runtime restoration |
| `reproduction/evaluation.py` | Natural-game evaluation and reference inference |
| `reproduction/artifacts.py` | Human-readable run summaries |
| `train_reproduction.py`, `evaluate_reproduction.py` | Current training/evaluation CLIs |
| `resume_reproduction.py` | Compatibility and worker checks, session logs, continuation history |
| `audit_recurrent_checkpoint.py` | Read-only checkpoint rollout, PPO reconstruction, gradient and action-rule diagnostics |
| `reproduction_forks/`, `run_reproduction_forks.py` | Isolated masked policy, recorded pre-action masks, fixed-LR fork runner, recovery and paired comparison |
| `extend_reproduction_fork.py` | Branch a completed fork from its fixed-budget endpoint into a new run without altering parent files |
| `reproduction_forks/lr_probe.py`, `run_masked_lr_probe.py` | Matched masked LR `0.00005` versus `0.00010` continuation, recovery and paired comparison |
| `extend_masked_lr_probe.py` | Long fixed-LR `0.00010` continuation of the completed probe arm, with custom-arm recovery |
| `inspector/`, `inspect_policy.py` | CPU policy viewer, controls, activation capture and plots |
| `experiment_catalog.py`, `dashboard.py`, `templates/` | Cross-run summaries and charts |
| `run_layout.py`, `run_provenance.py` | Artifact paths, source copies and hashes |
| `check_repository.py` | Maintained-document metadata and link validation |
| `threes.py`, `main.py`, `gui.py`, `external_agent.py` | Historical simulator and GUI/reference adapter |
| `train.py`, `rl_env.py`, `observation.py`, `rewards.py`, `experiment_config.py` | Historical MLP/MaskablePPO pipeline |
| `compare_agents.py`, `evaluate_models.py`, `evaluate_external.py`, `attach_comparison.py` | Historical evaluation and comparison attachment |

The historical engine is retained because the GUI, old experiments and tests still use it. Its metrics must not be pooled with the reproduction simulator's results. A folder or model name alone is not sufficient provenance; inspect the run's frozen configuration, method and source hashes.

The continuation compatibility gate checks the four core reproduction files (environment, policy, training, evaluation). Cosmetic edits to them can also invalidate continuation hashes. Make learning-method changes in an isolated experiment rather than altering an active run's provenance. Reporting-only maintenance can be verified independently.
