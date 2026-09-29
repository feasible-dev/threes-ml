# Recurrent model comparison

300 games per policy; seeds 800000 through 800299.

All policies use the same native reproduction simulator, natural initial boards, legal greedy actions, and reset recurrent memory between games.

| Model | Mean | Median | >=3072 | >=6144 | 6144 95% Wilson CI | Truncated |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 2026-09-26_153305_recurrent_base_s71 | 459.4 | 360.0 | 0.0% | 0.0% | 0.0%–1.3% | 0 |
| 2026-09-26_153305_recurrent_shaped_s71 | 520.0 | 390.0 | 0.0% | 0.0% | 0.0%–1.3% | 0 |
| published_pretrained_reference | 284310.9 | 248308.5 | 75.0% | 21.3% | 17.1%–26.3% | 0 |

These are validation results once inspected for model selection. They are not directly interchangeable with scores from our historical game variant.

The published reference, when included, uses downloaded weights. Our training runs start randomly; importing reference weights is restricted to verification/evaluation.

Research relevance and retention judgments are in each run's `metadata/review.json`. No files are deleted automatically.

## Research interpretation (September 26, 2026)

The trained models reached tile 192 in only 5/300 games each and never reached 384. The downloaded reference reached at least 384 in 300/300 games and 6,144 in 64/300 games in this simulator. Its 21.3% large-tile rate is close to the 21.5% reported by the project, but this 300-game sample does not establish the full published evaluation protocol or identical weights.

Structural reward increased mean score by 60.59 on these paired initial seeds. A 20,000-resample paired bootstrap of per-game score differences gives an approximate 95% interval of -9.5 to +132.2, so this single-seed comparison does not establish an improvement. Both runs used 2,007,040 transitions and showed no large-tile progress. The base 100-game periodic mean peaked at 548.61 around 1.53 million transitions and ended at 410.16; the shaped mean fluctuated between roughly 357 and 529 after its early rise. These curves do not justify projecting the published result from longer runtime alone.

The cumulative training invalid-action rates ended at 25.10% (base) and 23.98% (shaped). Training samples all four directions; evaluation selects a legal move. This policy/decision mismatch and the unpublished original training settings are concrete hypotheses for future controlled tests. The short budget, only one training seed, and absent curriculum also limit conclusions about what a larger corrected program could learn.

**Fidelity correction:** a later [source audit](../../../docs/FIDELITY_AUDIT.md) found that the completed shaped run omitted the published +0.01 corner reward when two equally large maximum tiles occupied the upper-left positions. Its source snapshot preserves the actual training code. The empirical scores above stand, but the shaped condition is not an exact copy of the published structural reward. Current code fixes the case for new runs; do not mix old and corrected shaping within a continuation. The base reward was unaffected.
