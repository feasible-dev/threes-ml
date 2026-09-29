---
status: reference
updated: 2026-09-27T19:33:00-04:00
scope: Long masked LR 0.00010 continuation from the completed short probe
---

# Long masked LR 0.00010 continuation

The [current decision](RESEARCH_DECISION_2026-09-27_LR10_LONG.md) selects the completed `masked_lr_00010` probe at **230,031,360 transitions** as the parent. It is already a 5M-transition descendant of the completed 225M long run, so extending it preserves that work. The new child keeps training masking on and LR fixed at `0.00010`; it restores the endpoint weights, optimizer, simulator, recurrent memory and RNG and collects fresh rollouts. Completed parent runs remain unchanged.

Run from the project root in a visible PowerShell terminal. The first command is a read-only preview; the second adds **100,007,936 attempted transitions** after rollout rounding, targeting **330,039,296**:

```powershell
.\extend_masked_lr_probe.ps1 -ParentFork results\runs\2026-09-27_185554_mask_lr_probe_masked_lr_00010 -AdditionalSteps 100000000 -Preview
.\extend_masked_lr_probe.ps1 -ParentFork results\runs\2026-09-27_185554_mask_lr_probe_masked_lr_00010 -AdditionalSteps 100000000
```

The launcher prints the new group and child paths. It refuses an incomplete parent, source/version mismatch or concurrent continuation. If interrupted, preview and resume the new child with this launcher so the custom LR arm is recognized:

```powershell
.\extend_masked_lr_probe.ps1 -ResumeRun results\runs\GROUP_masked_lr_00010 -Preview
.\extend_masked_lr_probe.ps1 -ResumeRun results\runs\GROUP_masked_lr_00010
```

The [short paired probe](../results/fork_batches/2026-09-27_185554_mask_lr_probe/comparison.md) favored LR `0.00005` over `0.00010` at equal 5M-transition budget. This user-selected long continuation tests a different horizon, without a matched long LR `0.00005` control. Periodic 100-game validations use the existing development seeds 700000–700099; reserve fresh development seeds 850000–850299 for endpoint evaluation and final-test seeds 900000–909999 until model selection is frozen.
