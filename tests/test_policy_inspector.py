import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pygame

from inspector.session import InspectionSession, legal_probabilities
from inspector.window import InspectorWindow, SIZE, SLIDER, speed_from_fraction


class FakePolicy:
    hidden_size = 4
    label = "Test policy"
    path = Path("fake.onnx")
    sha256 = "test"

    def __init__(self):
        self.calls = []

    def infer(self, observation, hidden, cell):
        self.calls.append((observation.copy(), hidden.copy(), cell.copy()))
        return np.array([4., 3., 2., 1.]), 12., hidden + 1, cell + 2


class PolicyInspectorTests(unittest.TestCase):
    def setUp(self):
        pygame.font.init()
        self.addCleanup(pygame.font.quit)
        self.policy = FakePolicy()
        self.session = InspectionSession(self.policy)
        self.addCleanup(self.session.close)

    def test_masked_distribution_is_stable_and_not_influenced_by_illegal_logit(self):
        p = legal_probabilities([10000, 2, 3, -10000], [False, True, True, False])
        self.assertAlmostEqual(p.sum(), 1)
        self.assertEqual(p[0], 0)
        self.assertEqual(p[3], 0)
        self.assertAlmostEqual(p[2], 1 / (1 + np.exp(-1)))
        np.testing.assert_array_equal(legal_probabilities([1, 2, 3, 4], [False] * 4), np.zeros(4))

    def test_draw_pause_and_invalid_moves_do_not_advance_memory(self):
        self.session.env.board[0] = [1] + [0] * 15
        self.session.analyze()
        window = InspectorWindow(self.session)
        count = len(self.policy.calls)
        rng = self.session.env.rng.copy()
        for _ in range(3):
            window.draw(pygame.Surface(SIZE))
            window.tick(10000)
        self.assertFalse(self.session.move(0))
        self.assertEqual(len(self.policy.calls), count)
        np.testing.assert_array_equal(self.session.env.rng, rng)
        np.testing.assert_array_equal(self.session.hidden, np.zeros((1, 4)))
        self.assertEqual([action for action, _ in window.action_rows()], [1, 3])

    def test_manual_override_advances_once_and_undo_restores_branch(self):
        original = self.session.env.observe().copy()
        decision = self.session.decision
        action = next(a for a in range(4) if decision.legal[a] and a != decision.recommendation)
        self.assertTrue(self.session.move(action))
        expected_board = self.session.env.board.copy()
        expected_rng = self.session.env.rng.copy()
        expected_probabilities = self.session.decision.probabilities.copy()
        self.assertEqual(len(self.policy.calls), 2)
        np.testing.assert_array_equal(self.policy.calls[-1][1], np.ones((1, 4)))
        self.assertTrue(self.session.undo())
        np.testing.assert_array_equal(self.session.env.observe(), original)
        self.assertIs(self.session.decision, decision)
        self.session.move(action)
        np.testing.assert_array_equal(self.session.env.board, expected_board)
        np.testing.assert_array_equal(self.session.env.rng, expected_rng)
        np.testing.assert_array_equal(self.session.decision.probabilities, expected_probabilities)

    def test_move_quality_uses_pre_move_policy_and_tracks_undo_and_new_game(self):
        self.session.env.board[0] = [1] + [0] * 15
        self.session.analyze()
        # Legal down/right have logits 3/1: the illegal up logit 4 must not
        # influence the ratio. Right is exp(1 - 3), not its raw probability.
        self.assertFalse(self.session.move(0))
        self.assertEqual(self.session.move_quality, [])
        self.session.move(3, "manual")
        self.assertAlmostEqual(self.session.move_quality[-1].relative_preference, np.exp(-2))
        self.assertEqual(self.session.move_quality[-1].source, "manual")
        self.session.move(self.session.decision.recommendation, "autoplay")
        self.assertEqual(self.session.move_quality[-1].relative_preference, 1)
        self.assertEqual(self.session.move_quality[-1].source, "autoplay")
        self.session.undo()
        self.assertEqual(len(self.session.move_quality), 1)
        self.session.undo()
        self.session.move(1, "AI step")
        self.assertEqual(len(self.session.move_quality), 1)
        self.assertEqual(self.session.move_quality[-1].relative_preference, 1)
        window = InspectorWindow(self.session)
        window.draw(pygame.Surface(SIZE))
        self.session.new_game(730001)
        self.assertEqual(self.session.move_quality, [])

    def test_slider_autoplay_step_and_manual_pause(self):
        window = InspectorWindow(self.session)
        window.event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(SLIDER.right, SLIDER.centery)), 0)
        self.assertAlmostEqual(window.speed, 60)
        window.event(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=(SLIDER.right, SLIDER.centery)), 0)
        window.command("play", 0)
        window.tick(17)
        self.assertEqual(int(self.session.env.meta[0, 2]), 1)
        self.assertTrue(window.autoplay)
        action = self.session.decision.recommendation
        key = (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT)[action]
        window.event(pygame.event.Event(pygame.KEYDOWN, key=key), 18)
        self.assertFalse(window.autoplay)
        self.assertEqual(int(self.session.env.meta[0, 2]), 2)
        window.command("step", 19)
        self.assertFalse(window.autoplay)
        self.assertEqual(int(self.session.env.meta[0, 2]), 3)
        self.assertEqual(speed_from_fraction(-1), .25)
        self.assertEqual(speed_from_fraction(2), 60)

    def test_reset_memory_and_bonus_preview_never_reveal_actual_bonus(self):
        self.session.move(self.session.decision.recommendation)
        self.session.new_game(730001)
        np.testing.assert_array_equal(self.policy.calls[-1][1], np.zeros((1, 4)))
        self.assertFalse(self.session.history)
        self.session.env.meta[0, 1] = 4
        self.session.env.meta[0, 6] = 10
        for actual in (4, 5, 6):
            self.session.env.meta[0, 0] = actual
            self.assertEqual(self.session.preview, "6 / 12 / 24")
        with self.assertRaisesRegex(ValueError, "reserved"):
            self.session.new_game(900000)

    def test_flipped_view_maps_keyboard_and_clicks_without_changing_policy_state(self):
        window = InspectorWindow(self.session)
        before = self.session.env.observe().copy()
        hidden = self.session.hidden.copy()
        calls = len(self.policy.calls)
        original_cell = window.cell_rect(0)
        window.command("flip", 0)
        self.assertEqual(window.cell_rect(12), original_cell)
        np.testing.assert_array_equal(self.session.env.observe(), before)
        np.testing.assert_array_equal(self.session.hidden, hidden)
        self.assertEqual(len(self.policy.calls), calls)
        # Screen-up must play canonical-down, with the same mapping for clicks.
        self.assertTrue(self.session.decision.legal[1])
        window.event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP), 1)
        expected = self.session.env.observe().copy()
        window.command("undo", 2)
        rect = dict(window.action_rows())[1]
        window.event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=rect.center), 3)
        np.testing.assert_array_equal(self.session.env.observe(), expected)

    def test_export_and_event_log_retain_memory_and_override(self):
        with tempfile.TemporaryDirectory() as folder:
            session = InspectionSession(FakePolicy(), output=Path(folder) / "inspection")
            try:
                path = session.export()
                with np.load(path) as state:
                    np.testing.assert_array_equal(state["hidden_before"], np.zeros((1, 4)))
                    np.testing.assert_array_equal(state["hidden_after"], np.ones((1, 4)))
                    self.assertEqual(state["observation"].shape, (1, 21))
                session.move(session.decision.recommendation, "manual")
                session.undo()
            finally:
                session.close()
            rows = [json.loads(line) for line in (Path(folder) / "inspection/events.jsonl").read_text().splitlines()]
            move = next(row for row in rows if row["event"] == "move")
            self.assertEqual(move["source"], "manual")
            self.assertEqual(move["relative_preference"], 1)
            self.assertAlmostEqual(sum(move["probabilities"]), 1)
            self.assertEqual(len(move["observation"]), 21)
            self.assertTrue(any(row["event"] == "undo" for row in rows))


if __name__ == "__main__":
    unittest.main()
