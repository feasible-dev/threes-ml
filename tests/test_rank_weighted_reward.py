import unittest

from rewards import rank_weighted_merges
from threes import move


class RankWeightedMergeTests(unittest.TestCase):
    def test_multiple_merge_levels_in_one_move(self):
        before = (
            (1, 2, 0, 0),
            (3, 3, 0, 0),
            (6, 6, 0, 0),
            (12, 0, 0, 0),
        )
        self.assertEqual(rank_weighted_merges(before, move(before, "left")), 1 + 2 + 3)

    def test_slide_without_merge_has_zero_reward(self):
        before = ((0, 3, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))
        self.assertEqual(rank_weighted_merges(before, move(before, "left")), 0)


if __name__ == "__main__":
    unittest.main()
