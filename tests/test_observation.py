import unittest

from observation import observe
from threes import Game, legal_moves


class ObservationTests(unittest.TestCase):
    def test_initial_observation_uses_public_counts_and_legal_moves(self):
        game = Game(seed=42)
        observed = observe(game)
        self.assertEqual(len(observed.board_ranks), 16)
        self.assertEqual(len(observed.hint_mask), 16)
        self.assertEqual(sum(observed.hint_mask), 1)
        self.assertEqual(sum(observed.bag_remaining), 2)
        self.assertTrue(any(observed.legal_actions))
        self.assertFalse(hasattr(observed, "bag"))

    def test_bonus_observation_exposes_every_displayed_choice(self):
        board = ((192, 0, 0, 0),) + ((0, 0, 0, 0),) * 3
        game = Game(board, seed=4)
        game.next_hint = (6, 12, 24)
        observed = observe(game)
        self.assertEqual([rank for rank, present in enumerate(observed.hint_mask) if present], [4, 5, 6])

    def test_public_bag_count_resets_after_twelve_reveals(self):
        game = Game(seed=42)
        for _ in range(2):
            self.assertTrue(game.step(legal_moves(game.board)[0]))
        self.assertEqual(sum(game.visible_ordinary_counts), 12)
        self.assertEqual(observe(game).bag_remaining, (4, 4, 4))


if __name__ == "__main__":
    unittest.main()
