---
status: reference
updated: 2026-09-26T21:29:45-04:00
scope: Watching and steering a recurrent policy
---

# Watching and steering a recurrent policy

## Start the viewer

```powershell
# Published pretrained reference, initially paused so you can examine its advice.
.\watch_policy.ps1

# Start autoplay; the slider remains available while it plays.
.\watch_policy.ps1 -Autoplay -Speed 10

# Inspect live internal activations of the published agent.
.\watch_policy.ps1 -Features

# Inspect one of our trained recurrent models instead.
.\watch_policy.ps1 -Model results\runs\2026-09-26_162446_recurrent_v2_control_s72\models\best.zip
```

The viewer uses the existing `.venv` CPU environment, Pygame, ONNX Runtime and the audited engine-v2 simulator. It does not use CUDA, start training, or modify run artifacts. It consumes some CPU resources, so it is not entirely free while other work runs. Inference uses one CPU thread for ONNX or PyTorch. Choose a completed checkpoint when viewing our own models. The weights load once; a later change to the file does not change the current viewer policy.

The downloaded strong agent is explicitly labelled **Published pretrained reference**. Our locally trained models are separately labelled by their run. This viewer uses the same game engine as recurrent evaluations; `main.py --external-ai` remains the older historical-game GUI.

## Controls

| Control | Result |
| --- | --- |
| Arrow keys / click a listed direction | Play your own legal move; autoplay pauses |
| Space / Play | Toggle autoplay using the highest-logit legal action |
| N / AI step | Play exactly one recommended move and remain paused |
| Speed slider | Set 0.25–60 moves per second; actual speed depends on CPU/rendering |
| U / Undo | Restore the prior board, deck, RNG and LSTM memory; up to 256 moves |
| R / New game | Increment the inspection seed and reset recurrent memory |
| E / Export state | Save the observation, decision and recurrent state for later analysis |
| F / Flip top-bottom | Reflect the board vertically, together with arrows and keyboard controls |
| V / Inside the network | Switch between move advice and live layer activations |
| Escape | Close the viewer |

Manual moves are observed by the agent on the next state. Redrawing the window, waiting, or pausing does not feed the same position repeatedly into the LSTM. Undo restores the hidden state and random stream, allowing a controlled alternative move from the same prior state. Different actions can consume randomness differently afterward; this is not a claim of matched future tile placement across divergent games.

## What the numbers mean

The taller window gives the board narrow, vertical tiles. The next card is centered just above the board; bonus possibilities appear as a centered group of cards. Score, largest tile and move count are compact indicators in the right column. Advice is arranged as a directional pad with clickable arrows, whose button backgrounds range from neutral to dark green. Illegal directions are omitted. You can play every move yourself while keeping the AI's advice visible; any manual move pauses autoplay.

The optional vertical reflection is a display/control transformation. It does not transform the network input or advance its memory. Screen-up maps to canonical-down when reflected; left/right remain unchanged. This lets a bottom-corner strategy appear at the top without changing its decisions. Event logs record view changes; move directions and NPZ exports remain in canonical simulator coordinates.

The actor returns four **logits** `z[a]`, one for each direction. They rank action preferences. The displayed number from 0 to 1 is the softmax restricted to legal directions:

```text
p(a | state, legal) = exp(z[a]) / sum(exp(z[b]) for b in legal_actions)
```

Illegal directions are omitted from the panel and cannot be selected. Arrow color varies from nearly neutral at 0 to dark green at 1; the recommended arrow also has an outline. Colors use absolute probabilities, not a rescaling that forces the highest value to 1. Very small probabilities show `<0.01`. Autoplay chooses the largest legal logit deterministically; it does not sample the distribution.

These probabilities do not estimate winning or a move being objectively optimal. Logits are not action values `Q(s,a)` or predicted final scores. The separate critic value `V(s)` is a scalar estimate of discounted training reward for the position/history. It depends on reward scaling and the training method, and is not the official tile score.

The bonus preview shows the same possible-value window as evaluation observations; it does not disclose which bonus will actually spawn. The policy receives board ranks, preview and bag counts as in the released interface.

## Live move-quality chart

The chart plots the last 120 accepted moves of the current game. Blue points are human choices; green points are AI steps or autoplay. Its horizontal axis follows move order and scrolls after 120 moves. Undo removes the undone point; New game clears the chart. Invalid key presses add no points.

Each point is evaluated using the policy's cached decision **before** that move:

```text
relative_preference = p(chosen legal action) / max(p(legal actions))
                    = exp(chosen logit - maximum legal logit)
```

One means the move matches a top policy preference, including ties. Lower values indicate stronger disagreement. This is agreement with the loaded agent, **not proven optimality, expected score loss, or a chance of winning**. A confident policy can still be wrong. Autoplay therefore scores one by construction. The arrow buttons retain their original normalized probabilities; the chart uses a ratio to the best legal action, so these numbers have different meanings.

Each move's `relative_preference` and source are saved in `events.jsonl`, alongside its parent node. The full log retains undo branches; the visible chart shows only the current branch. No extra inference or lookahead is performed for the chart.

## Internal activations

Start with `-Features`, or press **V** in a newly launched viewer. Choose a layer, then click a heatmap cell to track that unit through the last 80 analyzed positions. Keyboard moves, AI step, autoplay, speed, undo and export continue to work. V restores the directional advice and move-quality chart. Layer buttons and heatmap clicks never play a move or advance memory.

| Layer | Shape for the published model | Meaning |
| --- | --- | --- |
| Tile + position | 17 x 8 | Learned value embedding plus position embedding for 16 squares and the preview |
| Encoder 1 / 2 / 3 | 2048 / 1024 / 1024 | Outputs after each encoder GELU |
| LSTM h / c | 1024 each | Hidden output and cell memory **after** processing the current observation |
| Actor / Critic | 1024 each | Post-GELU hidden layers before four logits / scalar value |

The tile slots use canonical row/column labels, even when the board is flipped. Selecting a board slot highlights the corresponding displayed tile. These input slots are explicitly built into the architecture. The deeper heatmaps arrange units in index order; their layout is **not a spatial map of the board**. A bright unit is a large activation, not a discovered feature or a measure of importance.

Blue is negative, orange is positive, and dark is zero. The hidden-state heatmap uses a fixed range [-1, 1]; other heatmaps show the current tensor's symmetric maximum-absolute range. The chosen-unit trace has its own displayed range, computed over its visible history. These scales are labelled because changes in color alone need not mean changes of equal numerical size across layers or states.

The original model file is unchanged. For the released ONNX model, existing intermediate tensors are exposed as additional outputs of an in-memory graph. For our PyTorch checkpoints, temporary forward hooks record the same layers. Both paths use one CPU inference per analyzed position; rendering reads cached arrays. Undo restores the cached activations and trims the active trace, and New game resets it. Custom ONNX graphs with different tensor names expose only recognized layers and the public LSTM states.

**E** now also saves `activation_slots`, `activation_encoder_1/2/3`, `activation_hidden`, `activation_cell`, `activation_actor`, and `activation_critic` when available. These arrays are captured during the decision represented by the export. The JSONL event log retains decisions, not all large activation arrays; use NPZ exports for selected positions. There is no claim yet that a deep unit detects the maximum tile or a merge opportunity.

## Inspection files

Every launch creates a timestamped directory in `results/inspections/`:

```text
TIMESTAMP_model_sSEED/
  README.md
  metadata.json              model path/hash, engine revision/hash, CPU, seed
  events.jsonl               states, legal masks, logits, probabilities,
                             chosen moves, manual/automatic source, undo branches
  state_NODE_TIME.npz        optional exports made with E
```

NPZ files contain `observation`, `hidden_before`, `cell_before`, `hidden_after`, `cell_after`, `logits`, `probabilities`, `legal`, `value`, and simulator arrays. They can reproduce a network forward pass with the matching model and support later activation analysis. We do not yet claim to identify concepts or causal circuits. The model file itself is not duplicated; retain the checkpoint matching its hash.

Interactive sessions may include human overrides, pauses and undo branches, so they are not performance benchmarks. Default inspection seeds start at 730000. The reserved final-test range 900000–909999 is refused and skipped by New game.

See the [interpretability research plan](INTERPRETABILITY.md) for probes and causal tests we can build next.
