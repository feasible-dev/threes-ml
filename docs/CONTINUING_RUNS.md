---
status: reference
updated: 2026-09-26T21:29:45-04:00
scope: Continuing an existing recurrent training run
---

# Continuing an existing recurrent training run

The continuation launcher uses the existing engine-v2 trainer. It does not change the model, reward, curriculum, seed, or learning-rate schedule. New setup work does not alter already running processes or their files. Wait for the whole current batch, including its comparison, to finish before starting a continuation on the same GPU.

## Existing long batch

The [current status](CURRENT_STATUS.md) identifies the allocated curriculum/LR comparison. `run_long_reproduction.ps1` adds 90M transitions to each selected run by default and then compares selected models on 1,000 natural games from seed 810000. Starting from 10,002,432 transitions, the original plan reaches 100,007,936 after rollout rounding. Starting from a later checkpoint produces a later endpoint.

Every invocation adds a fresh budget. Preview with `.\run_long_reproduction.ps1 -Preview`; do not relaunch it to attach to an existing batch. After interruption, inspect saved counts and continue only the unfinished work with its remaining budget. `-Runs PATH` selects one run and still performs the final comparison.

The decay condition stays at LR 0.00015 after its 5M milestone; constant LR stays at 0.0003. Periodic validation is approximately every 500k transitions and complete checkpoints every 2M. The original pair of 90M extensions was estimated at 10?12 hours and about 31 GB of additional checkpoint storage. These are planning estimates, not resource reservations. No checkpoint cleanup occurs automatically.

The launcher evaluates `best.zip`, selected on periodic validation, rather than necessarily the fixed-budget endpoint. Its development comparison is separate from the reserved final test. Chat and the CPU inspector remain available during training.

## Individual continuation commands

Run these commands from the project root in your PowerShell terminal:

```powershell
# Read-only: list engine-v2 runs, saved transition counts and validation scores.
.\resume_reproduction.ps1 -List

# Read-only: inspect the first completed control run.
.\resume_reproduction.ps1 -Run results\runs\2026-09-26_162446_recurrent_v2_control_s72 -AdditionalSteps 10000000 -Preview

# Example only, when this extra budget has been selected: train 10 million MORE transitions.
.\resume_reproduction.ps1 -Run results\runs\2026-09-26_162446_recurrent_v2_control_s72 -AdditionalSteps 10000000 -Note 'Test whether the control improves beyond its first 10 million transitions.'
```

Replace the run path to continue another v2 condition. `-AdditionalSteps` always means **additional** transitions, rounded up to complete PPO rollouts. For the control above, the saved count is 10,002,432; adding 10 million plans a new boundary at 20,004,864. The default additional budget is 10 million.

The launcher checks for running recurrent training/evaluation processes and other continuation launchers. It refuses to start when it sees them, even if you chose a different completed run. Process inspection must succeed; if Windows denies it, no training starts. The check is not a scheduler or a lock shared by the older batch launcher: do not launch between two jobs of a still-running batch.

On Windows, the virtual-environment Python redirector may remain alive as the parent of the executing interpreter. The launcher excludes that parent only after matching the parent/child relationship, venv executable and identical script arguments. Other Python parents and concurrent trainers remain blockers. This fixes the initial false "already running" rejection without stopping any processes.

The current terminal receives progress and errors, and the same output is saved in a session log. The assistant does not need access to your specific VS Code terminal tab. Start the command yourself in the terminal you want to watch; the chat remains available.

## Exactly what resumes

The **latest complete checkpoint**, not `best.zip`, restores:

- Network weights and optimizer state.
- LSTM hidden/cell memory and current observations.
- Active boards, bags, next tiles, curriculum history, counters and returns.
- Python, NumPy, Torch CPU/CUDA and simulator random state.
- Absolute-step learning-rate milestones; an already-applied reduction stays applied.

The configuration remains frozen in `metadata/config.json`. The existing trainer adds a source snapshot under `metadata/resumes/` and records it in provenance. The launcher checks core training-source hashes against the most recent snapshot and the recorded Torch, SB3-Contrib and Numba versions. A mismatch blocks automatic continuation; changing the method should be a separately designed experiment.

`best.zip` remains the model selected for evaluation. Continuing the latest checkpoint avoids silently discarding later learning just because an earlier evaluation was better. Periodic validation and the HTML report continue in the same run directory.

## Files for each continuation

```text
RUN/
  CONTINUATIONS.md                         index of continuation sessions
  metadata/continuations/TIMESTAMP_from_N/
    SUMMARY.md                            readable session summary
    session.json                          requested budget, checkpoint, command,
                                          timestamps, outcome and last saved count
    terminal.log                          visible terminal output, also saved
    resume_reproduction.py                exact orchestration code used
    resume_reproduction.ps1
    before/
      data/                               metrics and completed-episode log before resume
      models/                             previous best weights and selection metadata
      metadata/                           previous config, provenance, status and review
      evaluation/                         previous validation files and comparison link
```

The pre-resume copy preserves evidence if the trainer rolls back rows written after its last complete checkpoint. It does not duplicate every checkpoint. The old independent comparison and relevance review are archived, then their active attachments are removed so they are not presented as assessments of the extended budget. The original comparison directory remains intact. Review and compare the newly selected checkpoint again after continuation; the launcher does not automatically run a fresh independent comparison.

The standalone reports remain in `report/report.html`, and live monitoring is available in another terminal:

```powershell
.\.venv-repro\Scripts\python.exe src/dashboard.py results\runs\2026-09-26_162446_recurrent_v2_control_s72
```

Open the localhost URL printed by the dashboard.

## Interruption and recovery

`Ctrl+C` stops the current invocation. Inspect the newest **complete saved boundary** with `-List` or `-Preview`, then request only the remaining additional budget needed for the intended endpoint. Repeating the original command requests its full additional budget again. Unsaved work since that boundary is lost; this is not an instantaneous pause of every in-flight GPU operation. The current v2 configurations save approximately every two million transitions, at the first update, and at normal completion. A directory without `complete.json` is ignored.

After an abrupt process kill or power loss, a session record can still say `running`; inspect the saved checkpoint count and terminal log. The next launch inspects live processes, rather than trusting that old status alone. Engine-v1 runs are not eligible under current rules. GPU execution may not be bitwise identical across hardware/software changes; the existing CPU test checks exact next-update parity across a save/load and learning-rate milestone.

## Interpreting this batch

`control` uses corrected structural rewards and natural starts. `curriculum` starts half of eligible training games with 3–6 larger tiles near each environment's natural-game frontier. `curriculum_decay` adds a learning-rate reduction at five million transitions. All three validate on the same natural-start games. A larger training score from artificial starts is not evidence of stronger normal play; judge the natural validation curve and independent comparison.
