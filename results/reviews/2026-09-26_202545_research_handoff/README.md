# Frozen evidence for the research handoff

Captured September 26, 2026 at 20:25:45 America/New_York. See the [independent review](../../../docs/THREES_TRAINING_REVIEW.md) and the subsequent [research decision](../../../docs/RESEARCH_DECISION_2026-09-26.md). The evidence below remains frozen at the original capture time.

- `snapshot.json`: capture time, selected metric rows and training-source SHA-256 hashes.
- `10m_models.csv`, `10m_protocol.json`, `10m_comparison.md`: completed 300-game comparison, before the long continuations.
- `control/`, `curriculum/`, `curriculum_decay/`: frozen configs, method versions, recorded status, complete metrics, validation rows and per-game evaluations for the best and latest periodic checkpoints at capture time.

The active run continues to change its live files. These copies preserve what the reviewer saw. The per-game best/latest files use the periodic 100-seed validation set and must not be confused with the 300-game comparison. `status.json` is the recorded run status, not a process-health monitor. No model weights are included.
