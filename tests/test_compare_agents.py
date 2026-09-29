import csv
import random
import tempfile
import unittest
from pathlib import Path

from compare_agents import (
    evaluate,
    expected_spawn_move,
    greedy_move,
    position_move,
    position_value,
    write_csv,
)
from random_agent import play_random_game
from threes import legal_moves


class CompareAgentsTests(unittest.TestCase):
    def test_greedy_prefers_immediate_merges(self):
        board = (
            (6, 6, 0, 0),
            (0, 0, 0, 0),
            (3, 3, 0, 0),
            (0, 0, 0, 0),
        )
        self.assertEqual(greedy_move(board, 1, ("left", "up"), random.Random(1)), "left")

    def test_position_value_rewards_free_space(self):
        full = ((3, 6, 3, 6),) * 4
        with_space = ((0, 6, 3, 6),) + full[1:]
        self.assertGreater(position_value(with_space), position_value(full))
        choice = position_move(with_space, 1, ("left", "up"), random.Random(1))
        self.assertIn(choice, ("left", "up"))

    def test_spawn_policy_uses_public_hint(self):
        board = ((1, 2, 0, 0), (3, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))
        available = legal_moves(board)
        direction = expected_spawn_move(board, 3, available, random.Random(2))
        self.assertIn(direction, available)
        self.assertIn(
            expected_spawn_move(board, (6, 12, 24), available, random.Random(2)),
            available,
        )

    def test_comparison_reuses_random_reference_and_is_reproducible(self):
        first, _ = evaluate(3, 42)
        second, _ = evaluate(3, 42)
        self.assertEqual(first, second)
        self.assertEqual(first["aléatoire"][0][1:], play_random_game(42))
        self.assertEqual(len(first["fusion"]), 3)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.csv"
            write_csv(path, first)
            with path.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 12)
            self.assertEqual(rows[0]["seed"], "42")
            self.assertEqual(rows[0]["score"], str(play_random_game(42)[0]))


if __name__ == "__main__":
    unittest.main()
