---
status: reference
updated: 2026-09-26T21:29:45-04:00
scope: Interpreting the Threes policy
---

# Interpreting the Threes policy

The live viewer is implemented. The probe and intervention studies below are proposals; their scheduling follows [current status](CURRENT_STATUS.md). Transformer experiments are deferred.

## Existing evidence from other games

There is relevant published work, but its findings are not measurements of our Threes LSTM.

- **Li et al., ICLR 2023 — Othello.** A sequence model trained to predict moves from move histories developed a decodable board representation. The authors intervened on internal representations and measured changes in predictions. This provides a useful pattern: decode a candidate concept, then test its causal influence. [Paper](https://arxiv.org/abs/2210.13382).
- **Nanda, Lee and Wattenberg, BlackboxNLP 2023 — Othello.** Probing in terms of the current player's versus the opponent's pieces exposed a linear representation; vector interventions could control behavior. The choice of concept labels matters. [Paper](https://aclanthology.org/2023.blackboxnlp-1.2/).
- **McGrath et al., PNAS 2022 — AlphaZero chess.** Probes tracked where and when human-defined chess concepts became recoverable during learning. This motivates comparisons across layers and training checkpoints. Decodability alone does not establish a causal mechanism. [Paper](https://arxiv.org/abs/2111.09259).

A targeted search did not identify an equivalent mechanistic study of this particular released Threes agent. That is a search result, not proof that no such work exists.

## What is already explicit in this network

The input contains 16 board ranks, a preview rank, and four bag-count features. Each of the first 17 entries receives an 8-dimensional tile embedding plus a learned positional embedding. Thus, the early representation has 17 explicit position slots by construction. Finding those slots would verify the implementation, not discover an emergent strategy.

After concatenation, dense GELU layers mix the slots; a 1,024-unit LSTM integrates current features and prior memory. Its hidden and cell states are separate objects worth studying. Useful concepts may be distributed across many units; there need not be one neuron or one contiguous group per square.

The deeper questions are: where are tile identity, the maximum tile, available space and merge opportunities recoverable; how do their representations change across training; and does the policy use them to choose moves?

## A concrete first study

Freeze one published model, one locally trained checkpoint and a random-weight network of the same architecture. Replay the **same observation histories** through all three so differing gameplay does not confound the comparison. Include early and late natural-game positions and multiple seeds. Keep interactive hand-picked cases separate from the representative dataset. No final-test seeds are needed.

Record activations after embeddings, each encoder GELU layer, the LSTM hidden/cell states and the actor hidden layer. The viewer now captures these layers, plus the critic hidden layer, in the same decision forward pass and exports them with the input and recurrent state. A representative multi-game dataset collector and probe training remain future work.

Train small regularized linear decoders, keeping the policy frozen:

| Target concept | Labels / metric |
| --- | --- |
| Tile in each of 16 cells | Rank classification; per-cell and macro accuracy/F1, not just empty-cell accuracy |
| Maximum tile rank | Rank classification, stratified across game stages |
| Location of maximum tiles | Multi-label 16-cell mask, allowing ties |
| Number of empty cells | Regression MAE or 17-class classification |
| Legal directions | Four binary labels, balanced accuracy |
| Immediate merge opportunities | Defined per-direction counts/ranks from the audited engine |
| Maximum in a corner; monotone row structure | Explicit Boolean/continuous definitions, evaluated separately |

Split by **whole games**, not nearby frames, to prevent nearly identical positions leaking between training and test. Compare majority-label, shuffled-label and random-network baselines; report sample counts by tile rank and game stage. Matching raw-board features are a useful ceiling/baseline because several targets are deterministic functions of the board. High probe accuracy is evidence that information is accessible, not that the trained policy needs it.

For a linear probe, concept scores have the form `W h + b`, where `h` is a layer's activation. Testing whether this predicts a concept on held-out games is a manageable first experiment. Compare both individual units and multi-unit directions; do not assume localization in advance.

## Then test whether the concept changes decisions

After establishing reliable held-out decoding, intervene on candidate activation directions, or patch activations between carefully matched positions. Compare changes in the relevant **logit differences**, legal conditional probabilities and, eventually, natural-game performance. Keep the observation and legal-action mask fixed when testing an internal intervention, so rule changes do not masquerade as representation effects.

Use random directions of matched magnitude, unrelated concepts, multiple intervention strengths and several positions as controls. For the LSTM, preserve the actual prior history and distinguish changes to hidden state from changes to cell state. An intervention can push the network outside its normal activation distribution; a large behavioral effect alone is not proof of a clean concept mechanism.

Input perturbation maps can also be useful, but arbitrary edits to tile ranks/bags can create implausible states. Gradients through continuous embeddings are sensitivity measurements, not a direct demonstration of what a discrete tile "means" to the network.

## What is implemented now

The [policy inspector](POLICY_INSPECTOR.md) displays legal-action logits and conditional probabilities, supports manual overrides/undo/autoplay, and exports input/recurrent-state snapshots. **V**, or `watch_policy.ps1 -Features`, opens live layer heatmaps. Clicking a unit shows its recent activation trace. Selecting an input slot highlights its board square. Export includes the actual cached layer arrays, with model hash and simulator metadata in the session directory.

The two views answer **what the policy prefers here** and **what numerical representations it computed here**. They do not yet assign concepts to deep units. Linear probes, activation interventions, and claims about discovered concepts are not yet implemented or established. The 32-column hidden-state display is an arbitrary packing of 1,024 units, not a learned map of board locations.

The next useful research artifact would be a layer-by-concept table for the frozen published policy and our control checkpoint, with held-out-game probe results and random-network baselines. Only after that should we label a feature as causally involved in a particular decision.

## LSTM, attention and future planning

The strong downloaded model and our reproduction policies use a one-layer 1,024-unit LSTM, with shared recurrent features for actor and critic. They contain no transformer or attention block. The encoder already mixes all 16 squares through dense layers; the LSTM carries information across observations. The model makes a policy decision without explicit tree search. Training returns and value learning supply a future-reward objective; memory of past observations and explicit evaluation of future moves are different mechanisms.

Three questions should guide an architectural experiment:

| Question | Candidate mechanism |
| --- | --- |
| Which squares need to be related in this position? | Dense/convolutional encoding or spatial self-attention |
| What from earlier positions should remain available? | LSTM state or a temporal transformer context |
| Which move has better future consequences? | Learned returns/value functions and, optionally, explicit lookahead |

Self-attention can mix information across position tokens ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)). An architecture with attention does not automatically plan future moves or outperform this LSTM. [Parisotto et al., 2019](https://arxiv.org/abs/1910.06764) found optimization difficulties with standard transformers in RL and proposed GTrXL; it outperformed LSTM baselines on their memory benchmarks. That is evidence to investigate the family, not a Threes result.

**Deferred proposal, not implemented or selected as the next experiment:** replace only the spatial dense encoder with a small position-aware transformer over the 16 board tokens plus preview and bag context, retaining the LSTM and PPO objective. This would isolate spatial interaction modeling from temporal memory. Match environment rules, information, evaluation seeds and transition budget; report parameter count, throughput and wall time. Use at least two training seeds and natural-start evaluation, with large-tile rates alongside score. A temporal transformer or a search-augmented policy would be a separate experiment.

Attention weights would offer another visualization, but should not be presented as a causal explanation of a decision: [Jain and Wallace, 2019](https://arxiv.org/abs/1902.10186) demonstrated cases in NLP where substantially different attention distributions produced equivalent predictions. For this project, validate proposed board concepts with held-out probes and interventions regardless of architecture.
