"""Inference adapter for the published Kiok Threes! ONNX policy.

The 21-value observation follows the upstream browser adapter. Bag counts are
reconstructed from revealed ordinary tiles, so the agent receives no hidden
draw order or actual value of a bonus whose preview shows several options.
"""

from pathlib import Path

import numpy as np

from observation import ACTIONS, observe, tile_rank
from threes import Direction, Game


DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "external_models" / "kiok" / "onnx_model.onnx"
HIDDEN_SIZE = 1024


def encode_public_observation(game: Game) -> tuple[np.ndarray, tuple[bool, bool, bool, bool]]:
    state = observe(game)
    hint_index = (game.next_hint if isinstance(game.next_hint, int)
                  else tile_rank(game.next_hint[0]))
    bag = state.bag_remaining
    values = (*state.board_ranks, hint_index, *bag, sum(bag))
    return np.asarray(values, dtype=np.int64).reshape(1, 21), state.legal_actions


class KiokOnnxAgent:
    def __init__(self, model_path: Path = DEFAULT_MODEL) -> None:
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("Install the ML dependencies, including onnxruntime") from error
        if not model_path.is_file():
            raise FileNotFoundError(f"External ONNX model not found: {model_path}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        expected = {"observations", "lstm_h", "lstm_c"}
        if {item.name for item in self.session.get_inputs()} != expected:
            raise ValueError("Unexpected ONNX model inputs")
        if self.session.get_inputs()[0].shape[-1] != 21:
            raise ValueError("Expected a 21-value observation")
        self.reset()

    def reset(self) -> None:
        self.hidden = np.zeros((1, HIDDEN_SIZE), dtype=np.float32)
        self.cell = np.zeros((1, HIDDEN_SIZE), dtype=np.float32)

    def choose_action(self, game: Game) -> Direction:
        observation, legal = encode_public_observation(game)
        if not any(legal):
            raise ValueError("No legal move remains")
        logits, _, next_hidden, next_cell = self.session.run(
            ["logits", "value", "lstm_h_out", "lstm_c_out"],
            {"observations": observation, "lstm_h": self.hidden, "lstm_c": self.cell},
        )
        self.hidden, self.cell = next_hidden, next_cell
        scores = np.where(legal, logits[0], -np.inf)
        return ACTIONS[int(np.argmax(scores))]
