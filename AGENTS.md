---
status: current
updated: 2026-09-26T21:29:45-04:00
scope: Working in this repository
---

# Working in this repository

Read [current status](docs/CURRENT_STATUS.md), then only the reference documents needed for the task. The [documentation index](docs/README.md) defines document status and authority. User instructions take precedence.

- `docs/CURRENT_STATUS.md` is the maintained research direction. Dated decisions explain why; independent reviews are proposals until accepted. Historical proposals and frozen source snapshots are evidence, not current agent instructions.
- Keep documentation and product text in English. Date updates with an explicit UTC offset. Separate document update time from the step count/date of the evidence. Label planned, implemented, measured, and superseded work distinctly.
- The user launches long training/evaluation jobs in a visible terminal. Short engineering checks may run autonomously. Inspect existing work before launching competing jobs. Do not stop the user's training as a side effect of maintenance.
- Active continuations check core trainer hashes. Isolate changes to the learning method from active runs; preserve frozen configurations, source snapshots, checkpoints, logs, and comparisons. Record interventions as new experiments with explicit parent provenance.
- Keep the reproduction simulator and historical GUI benchmarks distinct. Downloaded reference weights, pretrained fine-tuning, and training from random weights are different claims. Reused development seeds are not final-test evidence.
- Current PowerShell entry points live at the root; historical launchers live in `scripts/legacy/`. Paths inside frozen run snapshots describe their original layout and must remain unchanged.
- When changing research direction, update `docs/CURRENT_STATUS.md` and explicitly identify the replaced decision. Do not leave competing "next steps" in reference guides.
- After reorganizing documentation, run `.\.venv\Scripts\python.exe src/check_repository.py`. For executable changes, run focused checks relevant to the affected behavior. See `scripts/README.md` and `src/README.md`.
