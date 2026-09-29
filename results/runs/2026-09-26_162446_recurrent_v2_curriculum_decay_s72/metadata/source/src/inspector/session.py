"""CPU inference and one-decision-per-position control for the policy inspector."""

from collections import deque
from dataclasses import dataclass
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from reproduction.environment import ENGINE_REVISION, NativeBatch, SCORES, legal_masks


ACTIONS = ("Up", "Down", "Left", "Right")
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "external_models/kiok/onnx_model.onnx"


def tile_value(rank):
    rank = int(rank)
    return rank if rank < 3 else 3 * 2 ** (rank - 3)


def legal_probabilities(logits, legal):
    """Conditional policy distribution, not action values or win probabilities."""
    logits = np.asarray(logits, dtype=np.float64).reshape(4)
    legal = np.asarray(legal, dtype=bool).reshape(4)
    if not np.isfinite(logits).all():
        raise ValueError("Policy returned non-finite logits")
    probabilities = np.zeros(4)
    if legal.any():
        weights = np.exp(logits[legal] - logits[legal].max())
        probabilities[legal] = weights / weights.sum()
    return probabilities


class CpuPolicy:
    def __init__(self, path):
        self.path = Path(path).resolve()
        # Load one immutable byte sequence, even if the selected file is later
        # replaced by training. The recorded hash identifies these exact weights.
        model_bytes = self.path.read_bytes()
        self.sha256 = hashlib.sha256(model_bytes).hexdigest()
        if self.path.suffix.lower() == ".onnx":
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            self.model = ort.InferenceSession(model_bytes, sess_options=options,
                                              providers=["CPUExecutionProvider"])
            self.hidden_size = 1024
            self.kind = "onnx"
            self.label = "Published pretrained reference" if self.path == DEFAULT_MODEL.resolve() else self.path.name
        elif self.path.suffix.lower() == ".zip":
            import torch
            from reproduction.training import TrackedRecurrentPPO
            torch.set_num_threads(1)
            self.model = TrackedRecurrentPPO.load(io.BytesIO(model_bytes), device="cpu").policy
            self.model.set_training_mode(False)
            self.hidden_size = self.model.lstm_actor.hidden_size
            self.kind = "pytorch"
            self.label = self.path.parent.parent.name + " / " + self.path.name
        else:
            raise ValueError("Choose a published .onnx or recurrent PPO .zip model")

    def infer(self, observation, hidden, cell):
        if self.kind == "onnx":
            outputs = self.model.run(["logits", "value", "lstm_h_out", "lstm_c_out"],
                                     {"observations": observation.astype(np.int64),
                                      "lstm_h": hidden, "lstm_c": cell})
        else:
            import torch
            with torch.no_grad():
                outputs = [x.numpy() for x in self.model.raw_forward(
                    torch.from_numpy(observation), torch.from_numpy(hidden), torch.from_numpy(cell))]
        return outputs[0][0].copy(), float(outputs[1].reshape(-1)[0]), outputs[2].copy(), outputs[3].copy()


@dataclass
class Decision:
    logits: np.ndarray
    probabilities: np.ndarray
    legal: np.ndarray
    value: float
    hidden_after: np.ndarray
    cell_after: np.ndarray

    @property
    def recommendation(self):
        return int(np.argmax(self.probabilities))


class InspectionSession:
    """Inference is cached until a legal move. Painting/pause never advances memory."""
    def __init__(self, policy, seed=730000, output=None):
        self.validate_seed(seed)
        self.policy = policy
        self.env = NativeBatch(1)
        self.output = Path(output) if output else None
        self.log = None
        self.node = 0
        self.sequence = 0
        self.history = deque(maxlen=256)
        if self.output:
            self.output.mkdir(parents=True, exist_ok=False)
            metadata = {"started_at": datetime.now().astimezone().isoformat(),
                        "model": str(policy.path), "model_sha256": policy.sha256,
                        "engine_revision": ENGINE_REVISION, "device": "cpu", "initial_seed": seed,
                        "environment_sha256": hashlib.sha256((ROOT / "src/reproduction/environment.py").read_bytes()).hexdigest(),
                        "action_order": ACTIONS, "purpose": "interactive inspection; not a performance benchmark",
                        "display": "raw actor logits and softmax conditional on legal actions; value head is a global reward estimate"}
            (self.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            self.log = (self.output / "events.jsonl").open("w", encoding="utf-8")
            (self.output / "README.md").write_text(
                "# Interactive inspection\n\n[Model and rules](metadata.json) · [Events](events.jsonl)\n\n"
                "This session can contain manual overrides and undo branches; it is not a benchmark. "
                "Each move records the observation, legal mask, logits, conditional probabilities, chosen action and next observation. "
                "Press E in the viewer to save an NPZ snapshot with observations, simulator state and recurrent memory before/after inference. "
                "These exports support later activation analysis; they are not causal explanations.\n", encoding="utf-8")
        self.new_game(seed)

    def record(self, kind, **data):
        if self.log:
            self.log.write(json.dumps({"event": kind, "node": self.node, "seed": self.seed, **data}) + "\n")
            self.log.flush()

    def new_game(self, seed):
        self.validate_seed(seed)
        self.seed = int(seed)
        self.env.meta[:] = 0
        self.env.reset([0], [seed])
        self.hidden = np.zeros((1, self.policy.hidden_size), np.float32)
        self.cell = self.hidden.copy()
        self.done = False
        self.end_reason = ""
        self.history.clear()
        self.sequence += 1
        self.node = self.sequence
        self.analyze()
        self.record("new_game", observation=self.env.observe()[0].tolist())

    @staticmethod
    def validate_seed(seed):
        if not 0 <= seed < 2**63:
            raise ValueError("Inspection seeds must lie below 2**63")
        if 900000 <= seed <= 909999:
            raise ValueError("Seeds 900000..909999 are reserved for the final test")

    @property
    def score(self):
        return int(SCORES[self.env.board[0]].sum())

    @property
    def maximum(self):
        return tile_value(self.env.board[0].max())

    @property
    def preview(self):
        first = int(self.env.meta[0, 1])
        if first <= 3:
            return str(first)
        last = min(first + 2, int(self.env.meta[0, 6]) - 3)
        return " / ".join(str(tile_value(r)) for r in range(first, last + 1))

    def analyze(self):
        legal = legal_masks(self.env.board)[0]
        if self.done or not legal.any():
            self.decision = None
            return
        logits, value, hidden, cell = self.policy.infer(self.env.observe(), self.hidden, self.cell)
        self.decision = Decision(logits, legal_probabilities(logits, legal), legal, value, hidden, cell)

    def move(self, action, source="manual"):
        decision = self.decision
        if decision is None or not 0 <= action < 4 or not decision.legal[action]:
            return False
        self.history.append(({name: getattr(self.env, name).copy() for name in ("board", "bag", "meta", "rng", "seeds")},
                             self.hidden.copy(), self.cell.copy(), decision, self.node))
        observation = self.env.observe()[0].tolist()
        parent = self.node
        self.hidden, self.cell = decision.hidden_after.copy(), decision.cell_after.copy()
        _, _, done, stats = self.env.step([action])
        self.done = bool(done[0])
        self.end_reason = ("Time limit reached" if stats[0, 5] else
                           "Tile limit reached" if self.env.meta[0, 6] >= 16 else "No legal moves") if self.done else ""
        self.sequence += 1
        self.node = self.sequence
        self.record("move", parent_node=parent, observation=observation, action=action, direction=ACTIONS[action], source=source,
                    recommended=decision.recommendation, logits=decision.logits.tolist(), legal=decision.legal.tolist(),
                    probabilities=decision.probabilities.tolist(), value=decision.value,
                    next_observation=self.env.observe()[0].tolist(), score=self.score, done=self.done)
        self.analyze()
        return True

    def undo(self):
        if not self.history:
            return False
        previous = self.node
        state, self.hidden, self.cell, self.decision, self.node = self.history.pop()
        for name, array in state.items():
            getattr(self.env, name)[:] = array
        self.done, self.end_reason = False, ""
        self.record("undo", from_node=previous)
        return True

    def export(self):
        if self.output is None or self.decision is None:
            return None
        stamp = datetime.now().strftime("%H%M%S_%f")
        path = self.output / f"state_{self.node:06d}_{stamp}.npz"
        decision = self.decision
        np.savez_compressed(path, observation=self.env.observe(), hidden_before=self.hidden, cell_before=self.cell,
                            hidden_after=decision.hidden_after, cell_after=decision.cell_after, logits=decision.logits,
                            probabilities=decision.probabilities, legal=decision.legal, value=decision.value,
                            **{name: getattr(self.env, name) for name in ("board", "bag", "meta", "rng", "seeds")})
        self.record("export", file=path.name)
        return path

    def close(self):
        if self.log:
            self.record("close", score=self.score, done=self.done)
            self.log.close()
