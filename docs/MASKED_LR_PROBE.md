---
status: reference
updated: 2026-09-27T19:33:00-04:00
scope: Short matched masked learning-rate probe from the completed 225M endpoint
---

# Masked learning-rate probe

The matched 5M-transition test from the completed 225,026,048-transition masked LR `0.00005` checkpoint has [completed and been compared](../results/fork_batches/2026-09-27_185554_mask_lr_probe/comparison.md). The two branches differ only in fixed effective LR: control `0.00005`, treatment `0.00010`. Each restored the same parent weights, optimizer, simulator, LSTM memory and RNG before collecting new rollouts. The completed parent and earlier forks remain unchanged. The [current decision](RESEARCH_DECISION_2026-09-27_LR10_LONG.md) selects a later continuation of the LR `0.00010` arm.

These commands describe the completed probe protocol. Re-running the second command with a new group creates another pair of runs. The first command is read only; the second runs the two arms sequentially and then evaluates the parent and both endpoints on paired natural development seeds 840000–840299:

```powershell
.\run_masked_lr_probe.ps1 -ParentFork results\runs\2026-09-27_125618_mask_lr_extension_masked_lr_00005 -AdditionalSteps 5000000 -Preview
.\run_masked_lr_probe.ps1 -ParentFork results\runs\2026-09-27_125618_mask_lr_extension_masked_lr_00005 -AdditionalSteps 5000000
```

The launcher prints its group under `results/fork_batches/` and each child under `results/runs/`. A requested 5M rounds to 5,005,312 attempted transitions per arm, ending at 230,031,360. The group stores a frozen plan and, after completion, `comparison.md`, `comparison.json` and per-game CSV files. The comparison is development evidence from one parent training trajectory, not a final test or independent-seed replication. The source hash gate and worker lock protect the completed parent and any other active training.

If an arm is interrupted, preview and resume **that child** with the probe launcher so the custom `masked_lr_00010` arm is recognized:

```powershell
.\run_masked_lr_probe.ps1 -ResumeRun results\runs\GROUP_masked_lr_00010 -Preview
.\run_masked_lr_probe.ps1 -ResumeRun results\runs\GROUP_masked_lr_00010
```

After the interrupted arm completes, rerun the training command with `-Group GROUP` to skip completed arms and finish any missing arm, then generate the comparison. If both arms completed but only comparison was interrupted, run `.\run_masked_lr_probe.ps1 -CompareGroup GROUP`. Do not use the generic fork-resume command for the custom LR arm; it does not register that arm name.
