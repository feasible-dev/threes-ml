---
status: historical
updated: 2026-09-26T21:29:45-04:00
scope: Published ONNX agent - compatibility audit
---

# Published ONNX agent: compatibility audit

> Historical stage only. Earlier recommendations and agent instructions are superseded by [current status](../CURRENT_STATUS.md).

This audit concerns transfer of the downloaded model into the historical GUI simulator. Evaluation in the newer reproduction simulator is documented in [REPRODUCTION.md](../REPRODUCTION.md).

This audit compares our engine with [pseudonam-gc/threes-web](https://github.com/pseudonam-gc/threes-web) at commit `9527295de72333c91592fff45e8ccbcabd6ba80d`. The local model's provenance and checksum are in [`external_models/kiok/README.md`](../../external_models/kiok/README.md).

| Item | Upstream model and game | Our adapter and engine | Result |
| --- | --- | --- | --- |
| Observation | 16 row-major tile ranks, one next-tile rank or first bonus-preview rank, three remaining ordinary-bag counts, and their total | Same 21 `int64` values, derived from the visible board, preview, and revealed ordinary draws | Matched |
| Actions | 0 up, 1 down, 2 left, 3 right | Same order; highest-logit legal action | Matched |
| Memory | Two 1 x 1024 `float32` LSTM states | Passed to the next decision and reset at the start of every game | Matched |
| Ordinary tiles | Four each of 1, 2, and 3 per bag, with refill | Same bag composition and refill | Matched |
| Movement and spawn | Tiles move one cell per action; a new tile appears on the opposite edge of a changed line | Same movement and spawn rule | Matched at rule level |
| Bonus preview | A public consecutive window of one to three possible bonus values | Same public window, encoded by its first tile rank | Matched at observation level |
| Bonus timing | One random phase is selected for each 21-move block | An independent 1-in-21 chance at each preview draw | Different; affects game trajectories |
| Episode end | The browser game continues until no legal move; training has additional tick and high-tile truncation conditions | Our game ends at tile 12,288 or when no move remains | Different; affects high-tile outcomes |

The bag counts are reconstructible from tiles already shown to the player, including the initial board and the announced next ordinary tile. The agent receives neither the hidden bag order nor the actual bonus tile before it appears. The LSTM can remember previous public observations.

The upstream README mentions 24 observation dimensions, but its browser adapter, C observation writer, and the ONNX input signature use 21. The adapter follows those executable interfaces. We have not established whether the distributed ONNX weights are the checkpoint behind the upstream 21.5% tile-6,144 claim.

Consequently, [`src/evaluate_external.py`](../../src/evaluate_external.py) measures how this published policy plays **in our game**, not a reproduction of the author's reported evaluation. It evaluates the four local heuristics on the same initial seeds. Random draws after the first move can diverge across policies, so these are common initial seeds rather than identical future tile sequences. Reserve a separate seed range for a final comparison after any tuning or imitation work.
