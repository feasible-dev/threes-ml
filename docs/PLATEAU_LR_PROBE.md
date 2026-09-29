---
status: reference
updated: 2026-09-28T18:08:32-04:00
scope: Visible launcher for the post-run assessment and matched LR-halving pilot
---

# Plateau LR probe launcher

The [current decision](RESEARCH_DECISION_2026-09-28_PLATEAU.md) selects the complete 330,039,296-transition masked LR `0.00010` checkpoint as the pilot parent. This launcher runs the planned assessment on 300 natural development games, seeds 850000–850299, then creates two separate 10,002,432-transition child runs. The control keeps LR `0.00010`; the treatment halves it to `0.00005`. Both retain masking, curriculum, rewards, optimizer state, live simulator and recurrent state. Finally, it compares the two fixed endpoints and their parent on 300 paired natural development games, seeds 860000–860299.

Run from the project root in a visible PowerShell terminal:

```powershell
.\run_plateau_lr_probe.ps1 -Preview
.\run_plateau_lr_probe.ps1
```

The launcher prints its group name before evaluation. Baseline results appear in `results/comparisons/GROUP_baseline/`; training runs appear in `results/runs/GROUP_masked_lr_00010/` and `results/runs/GROUP_masked_lr_00005/`; the paired pilot report appears in `results/fork_batches/GROUP/comparison.md`. Each stage keeps per-game scores and provenance. The baseline report lists the 230M parent, the fixed 330M endpoint and the separately labeled selected-best model. The training comparison uses fixed-budget endpoints, not selected-best checkpoints.

If an arm is interrupted, preview and resume that run, then rerun the group command. Completed stages are reused after provenance checks:

```powershell
.\run_plateau_lr_probe.ps1 -ResumeRun results\runs\GROUP_masked_lr_00010 -Preview
.\run_plateau_lr_probe.ps1 -ResumeRun results\runs\GROUP_masked_lr_00010
.\run_plateau_lr_probe.ps1 -Group GROUP
```

Substitute the interrupted arm name if it is `masked_lr_00005`. If both arms completed but evaluation stopped, run:

```powershell
.\run_plateau_lr_probe.ps1 -CompareGroup GROUP
```

The launcher refuses a concurrent recurrent trainer/evaluator, an incomplete parent, a source/version mismatch, or a group with different provenance. It does not modify the completed parent runs. Final-test seeds 900000–909999 remain unused.
