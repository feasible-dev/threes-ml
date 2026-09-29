"""Activation instrumentation must preserve decisions and recurrent history."""

import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pygame

from inspector.activations import activation_series, torch_inference
from inspector.session import CpuPolicy, DEFAULT_MODEL, InspectionSession
from inspector.window import InspectorWindow, SIZE


class ActivationTests(unittest.TestCase):
    @unittest.skipUnless(DEFAULT_MODEL.exists(), "Published reference is not installed")
    def test_released_onnx_instrumentation_matches_original_and_exports_cached_arrays(self):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        original = ort.InferenceSession(str(DEFAULT_MODEL), sess_options=options,
                                       providers=["CPUExecutionProvider"])
        policy = CpuPolicy(DEFAULT_MODEL)
        pygame.font.init()
        self.addCleanup(pygame.font.quit)
        with tempfile.TemporaryDirectory() as directory:
            session = InspectionSession(policy, output=Path(directory) / "session")
            try:
                for index in range(8):
                    outputs = original.run(["logits", "value", "lstm_h_out", "lstm_c_out"],
                                           {"observations": session.env.observe().astype(np.int64),
                                            "lstm_h": session.hidden, "lstm_c": session.cell})
                    d = session.decision
                    for expected, actual in zip(outputs, (d.logits[None], np.array([[d.value]]),
                                                         d.hidden_after, d.cell_after)):
                        np.testing.assert_allclose(actual, expected, atol=3e-4, rtol=3e-4)
                    self.assertEqual(d.activations["slots"].shape, (17, 8))
                    self.assertEqual(d.activations["encoder_1"].shape, (2048,))
                    for name in ("encoder_2", "encoder_3", "hidden", "cell", "actor", "critic"):
                        self.assertEqual(d.activations[name].shape, (1024,))
                    self.assertEqual(d.recommendation, np.argmax(np.where(d.legal, outputs[0][0], -np.inf)))
                    session.move(d.recommendation)
                trace = activation_series(session, "hidden", 7)
                session.undo()
                np.testing.assert_array_equal(activation_series(session, "hidden", 7), trace[:-1])
                decision = session.decision
                window = InspectorWindow(session, features=True)
                window.feature_panel.layer = "slots"
                rect, _, width, height = window.feature_panel.geometry(decision)
                # Select slot 5, channel 2. Clicking the heatmap is not a move.
                point = (rect.x + 2 * width + 1, rect.y + 5 * height + 1)
                window.event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=point), 0)
                for _ in range(2):
                    window.draw(pygame.Surface(SIZE))
                self.assertIs(session.decision, decision)
                self.assertEqual(window.feature_panel.board_slot(), 5)
                self.assertEqual(window.feature_panel.units["slots"], 42)
                with np.load(session.export()) as state:
                    for name, array in decision.activations.items():
                        np.testing.assert_array_equal(state[f"activation_{name}"], array)
                session.new_game(730001)
                self.assertEqual(len(activation_series(session, "hidden", 7)), 1)
            finally:
                session.close()
        self.assertEqual(policy.sha256, hashlib.sha256(DEFAULT_MODEL.read_bytes()).hexdigest())

    def test_pytorch_hooks_capture_real_layers_without_changing_forward_or_leaking_hooks(self):
        import gymnasium as gym
        import torch
        from reproduction.policy import PublishedPolicy
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        policy = PublishedPolicy(gym.spaces.Box(0, 17, (21,), dtype=np.float32),
                                 gym.spaces.Discrete(4), lambda _: 0., hidden_size=16).eval()
        obs = np.zeros((1, 21), dtype=np.int64)
        obs[:, :17] = np.arange(17)
        h, c = np.zeros((1, 16), np.float32), np.ones((1, 16), np.float32)
        with torch.no_grad():
            original = policy.raw_forward(torch.from_numpy(obs), torch.from_numpy(h), torch.from_numpy(c))
            expected_slots = (policy.features_extractor.value_embed(torch.from_numpy(obs[:, :17]))
                              + policy.features_extractor.position_embed)[0].numpy()
        outputs, captured = torch_inference(policy, obs, h, c)
        for expected, actual in zip(original, outputs):
            np.testing.assert_array_equal(actual, expected.detach().numpy())
        np.testing.assert_array_equal(captured["slots"], expected_slots)
        self.assertEqual(captured["encoder_1"].shape, (32,))
        self.assertEqual(captured["actor"].shape, (16,))
        self.assertTrue(all(not module._forward_hooks for module in policy.modules()))


if __name__ == "__main__":
    unittest.main()
