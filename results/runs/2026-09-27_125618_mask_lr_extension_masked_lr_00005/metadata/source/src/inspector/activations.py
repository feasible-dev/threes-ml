"""Read intermediate computations without changing weights or recurrent state."""

import numpy as np


LAYER_LABELS = {
    "slots": "Tile + position", "encoder_1": "Encoder 1", "encoder_2": "Encoder 2",
    "encoder_3": "Encoder 3", "hidden": "LSTM h", "cell": "LSTM c",
    "actor": "Actor", "critic": "Critic",
}

# Tensor names belong to the released ONNX graph. Unknown graphs still expose
# their public h/c outputs; never guess that an arbitrary tensor is a layer.
ONNX_LAYERS = {
    "slots": ("/Add_output_0", ["batch", 17, 8]),
    "encoder_1": ("/encoder/encoder.2/Mul_1_output_0", ["batch", 2048]),
    "encoder_2": ("/encoder/encoder.4/Mul_1_output_0", ["batch", 1024]),
    "encoder_3": ("/encoder/encoder.6/Mul_1_output_0", ["batch", 1024]),
    "actor": ("/decoder/decoder.1/Mul_1_output_0", ["batch", 1024]),
    "critic": ("/value/value.1/Mul_1_output_0", ["batch", 1024]),
}


def instrument_onnx(model_bytes):
    """Expose existing tensors in an in-memory copy; the model file is untouched."""
    import onnx
    model = onnx.load_model_from_string(model_bytes)
    existing = {output.name for output in model.graph.output}
    available = {name for node in model.graph.node for name in node.output}
    outputs = {}
    for layer, (name, shape) in ONNX_LAYERS.items():
        if name not in available:
            continue
        outputs[layer] = name
        if name not in existing:
            model.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape))
    return model.SerializeToString(), outputs


def torch_inference(policy, observation, hidden, cell):
    """Capture post-GELU activations during the same CPU forward pass."""
    import torch
    captured = {}
    encoder = policy.features_extractor
    layers = {"encoder_1": encoder.encoder[2], "encoder_2": encoder.encoder[4],
              "encoder_3": encoder.encoder[6], "actor": policy.mlp_extractor.policy_net[1],
              "critic": policy.mlp_extractor.value_net[1]}

    def capture(name):
        def hook(_module, _inputs, output):
            captured[name] = output[0].detach().cpu().numpy().copy()
        return hook

    handles = [module.register_forward_hook(capture(name)) for name, module in layers.items()]
    # The embedding output plus the learned positional tensor is the actual
    # 17x8 representation consumed by the encoder, not a board saliency map.
    def slots_hook(_module, _inputs, output):
        captured["slots"] = (output[0] + encoder.position_embed).detach().cpu().numpy().copy()
    handles.append(encoder.value_embed.register_forward_hook(slots_hook))
    try:
        with torch.no_grad():
            outputs = [x.cpu().numpy() for x in policy.raw_forward(
                torch.from_numpy(observation), torch.from_numpy(hidden), torch.from_numpy(cell))]
    finally:
        for handle in handles:
            handle.remove()
    return outputs, captured


def activation_series(session, layer, unit, limit=80):
    """The active undo branch, one value per analyzed position (including now)."""
    decisions = [entry[3] for entry in session.history]
    if session.decision is not None:
        decisions.append(session.decision)
    values = []
    for decision in decisions[-limit:]:
        vector = decision.activations.get(layer)
        if vector is not None and unit < vector.size:
            values.append(float(vector.flat[unit]))
    return np.asarray(values)
