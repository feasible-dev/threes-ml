---
status: current
updated: 2026-09-28T19:22:39-04:00
scope: Documentation index
---

# Documentation index

Start with [CURRENT_STATUS.md](CURRENT_STATUS.md). It is the single maintained summary of the accepted research direction; other guides explain implementation or operations.

## Reading order

| Need | Document |
| --- | --- |
| What we are doing next | [Current status](CURRENT_STATUS.md) |
| Why that direction was selected | [September 28 long low-LR decision](RESEARCH_DECISION_2026-09-28_LOW_LR_LONG.md); [plateau decision](RESEARCH_DECISION_2026-09-28_PLATEAU.md); [September 27 long LR 0.00010 decision](RESEARCH_DECISION_2026-09-27_LR10_LONG.md); [short LR-probe decision](RESEARCH_DECISION_2026-09-27_LR_PROBE.md); [long-extension decision](RESEARCH_DECISION_2026-09-27_LONG_EXTENSION.md); [fork-pilot decision](RESEARCH_DECISION_2026-09-27_FORK_PILOT.md); [earlier interim decision](RESEARCH_DECISION_2026-09-27.md); [September 26 prior decision](RESEARCH_DECISION_2026-09-26.md) |
| Suggested alternatives and objections | [Independent review](THREES_TRAINING_REVIEW.md) |
| Actual network, observations and rewards | [Implemented method](REPRODUCTION.md) |
| Original v2 experiment design and engine tests | [V2 reference](REPRODUCTION_V2.md) |
| Masking by learning-rate fork commands and protocol | [Fork guide](REPRODUCTION_FORKS.md) |
| Matched LR `0.00005` versus `0.00010` probe | [LR-probe guide](MASKED_LR_PROBE.md) |
| Continue the completed LR `0.00010` arm for 100M | [Long-LR guide](MASKED_LR10_LONG.md) |
| Assess the plateau and run the matched LR-halving pilot | [Plateau LR guide](PLATEAU_LR_PROBE.md) |
| What matches the released model | [Fidelity audit](FIDELITY_AUDIT.md) |
| Resume, budgets, checkpoints, interruption | [Continuation guide](CONTINUING_RUNS.md) |
| Seed allocation and test discipline | [Seed protocol](SEED_PROTOCOL.md) |
| Watch, steer and inspect activations | [Policy inspector](POLICY_INSPECTOR.md) |
| Interpretability findings and proposed work | [Interpretability](INTERPRETABILITY.md) |
| Commands and code ownership | [Script index](../scripts/README.md), [source map](../src/README.md) |
| Earlier methods and reasoning | [Historical archive](archive/README.md) |

## Dates, status and conflicting statements

Maintained Markdown files carry a small metadata header:

- `status`: `current`, `reference`, `decision`, `proposal`, `historical`, or `snapshot`.
- `updated`: an ISO 8601 timestamp with UTC offset. It dates the document edit, not a new scientific measurement.
- `scope`: what the document covers.

For evidence, also state the run, engine, transition cutoff, evaluation seeds and game count where relevant. Exact source/configuration hashes live in each run's provenance. A filesystem modification date is not a reliable evidence date.

Resolve conflicts by subject:

1. User instructions govern the task. [Current status](CURRENT_STATUS.md) records the accepted research direction and points to its decision.
2. A review proposes changes; it does not supersede an accepted decision simply because its file was edited later. Record acceptance/rejection explicitly.
3. For what actually ran, use frozen run configuration, method, source hashes, continuation sessions and per-game results. For what runs now, inspect current source. Prose cannot override either.
4. Reference guides explain commands and mechanisms; they do not allocate the next training budget.
5. Historical documents, old catalogs, reviews and source snapshots describe their stated cutoff. Preserve them as evidence, not current instructions.

Generated run cards and the catalog are snapshots. A newly generated card may combine a later training count with an earlier separately identified comparison. It is not a live process monitor. Read metrics/checkpoint markers for progress, and the comparison's own protocol for score provenance.

Older generated evidence may have only a timestamped directory/run name and recorded transition count, rather than this metadata header. Do not invent a generation date for it. An already running Python process retains its loaded report generator until it exits, so its next card may still use the older format. Frozen scientific records are not rewritten just to standardize formatting.

## Maintenance

Update the current-status page when a decision changes. Keep dated reasoning in decision records and add a supersession notice when needed. Put historical material in `archive/`; avoid a second work log repeating the current plan. Use links to the authoritative page instead of copying its agenda into every guide.

Run `.\.venv\Scripts\python.exe src/check_repository.py` after moves or link changes. It checks maintained-document metadata and local links; it does not fetch the web or rewrite frozen evidence.
