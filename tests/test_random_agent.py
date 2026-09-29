import argparse
import unittest

from random_agent import play_random_game, positive_int


class RandomAgentTests(unittest.TestCase):
    def test_game_is_reproducible_and_terminates(self):
        first = play_random_game(42)
        self.assertEqual(first, play_random_game(42))
        score, moves, largest_tile = first
        self.assertGreater(score, 0)
        self.assertGreater(moves, 0)
        self.assertGreaterEqual(largest_tile, 3)

    def test_game_count_must_be_positive(self):
        self.assertEqual(positive_int("10"), 10)
        with self.assertRaises(argparse.ArgumentTypeError):
            positive_int("0")


if __name__ == "__main__":
    unittest.main()
