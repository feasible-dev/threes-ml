---
status: snapshot
updated: 2026-09-26T21:29:45-04:00
scope: Repository organization and documentation maintenance
---

# Repository cleanup ? September 26, 2026

Start with [current status](../../../docs/CURRENT_STATUS.md) and the [documentation index](../../../docs/README.md).

## Changes

- Added one accepted-direction page and agent entry instructions; reference guides now defer budget decisions to it.
- Added status, timestamp with UTC offset, and scope to 27 maintained Markdown files. Distinguished edit time from scientific evidence cutoff and proposals from implemented work.
- Archived eight historical documents with English filenames, repaired maintained links, and removed repeated early-work-log agenda text.
- Moved four old launchers to `scripts/legacy/` and fixed project-root discovery. Root entry points now serve the current recurrent workflow and viewer.
- Corrected interruption guidance: reusing an additional-budget command requests that budget again.
- Added generated-at/transition-cutoff notices to future run cards and the catalog. Clarified initial versus effective LR and development comparison versus final test.
- Added `--catalog-only`, refreshed the catalog, and kept per-run writes out of that refresh path. The active trainer independently continues to refresh its own card using its already loaded code.
- Updated source snapshots to include current and relocated PowerShell entry points. Added a read-only document checker.

## Verification

Four focused unit tests passed; nine PowerShell files parsed; all four historical launchers completed preview mode without training. All 27 maintained Markdown files passed metadata/local-link checks. Result Markdown outside frozen metadata was also checked for missing local targets, with none found. URL availability and section anchors were not checked.

Core environment/policy/training/evaluation hashes, JSON configuration templates and active batch/continuation launcher hashes match their pre-cleanup values. Both selected continuation previews report no compatibility issues. No new training/evaluation run was launched, no worker was stopped, and no checkpoint or frozen scientific evidence was deleted.

The active run's metrics and card naturally change during this work; their filesystem timestamps are not evidence of a maintenance edit. Existing Python workers may emit the older report format until restarted normally.

[Move map and verification](changes.json) ? [Pre-cleanup file hashes](before_hashes.json)
