# Interactive inspection

[Model and rules](metadata.json) · [Events](events.jsonl)

This session can contain manual overrides and undo branches; it is not a benchmark. Each move records the observation, legal mask, logits, conditional probabilities, chosen action and next observation. Press V to inspect internal activations. Press E to save an NPZ snapshot with observations, simulator state, recurrent memory before/after inference and available activation_* arrays from the cached decision. These exports support later activation analysis; they are not causal explanations.
