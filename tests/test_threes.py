import unittest

from threes import legal_moves, move


EMPTY_ROW = (0, 0, 0, 0)


def board_with_row(row: tuple[int, int, int, int]):
    return (row, EMPTY_ROW, EMPTY_ROW, EMPTY_ROW)


class MovementTests(unittest.TestCase):
    def test_one_and_two_merge_in_both_orders(self):
        for row in ((1, 2, 0, 0), (2, 1, 0, 0)):
            with self.subTest(row=row):
                self.assertEqual(move(board_with_row(row), "left")[0], (3, 0, 0, 0))

    def test_equal_tiles_merge_once_at_the_front(self):
        self.assertEqual(move(board_with_row((3, 3, 3, 3)), "left")[0], (6, 3, 3, 0))

    def test_gap_takes_priority_over_merge(self):
        self.assertEqual(move(board_with_row((0, 3, 3, 0)), "left")[0], (3, 3, 0, 0))

    def test_tile_moves_only_one_cell(self):
        self.assertEqual(move(board_with_row((0, 0, 0, 3)), "left")[0], (0, 0, 3, 0))

    def test_incompatible_tiles_block_line(self):
        self.assertEqual(move(board_with_row((1, 1, 1, 3)), "left")[0], (1, 1, 1, 3))

    def test_all_directions_and_unchanged_input(self):
        original = (
            (0, 0, 0, 0),
            (0, 3, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        )
        self.assertEqual(move(original, "left")[1], (3, 0, 0, 0))
        self.assertEqual(move(original, "right")[1], (0, 0, 3, 0))
        self.assertEqual(move(original, "up")[0], (0, 3, 0, 0))
        self.assertEqual(move(original, "down")[2], (0, 3, 0, 0))
        self.assertEqual(original[1], (0, 3, 0, 0))

    def test_merges_toward_right_and_down(self):
        right_board = board_with_row((0, 0, 1, 2))
        self.assertEqual(move(right_board, "right")[0], (0, 0, 0, 3))

        down_board = (
            EMPTY_ROW,
            EMPTY_ROW,
            (0, 3, 0, 0),
            (0, 3, 0, 0),
        )
        self.assertEqual(move(down_board, "down")[3], (0, 6, 0, 0))

    def test_legal_moves_excludes_unchanged_directions(self):
        board = board_with_row((1, 1, 1, 3))
        self.assertEqual(legal_moves(board), ("down",))


if __name__ == "__main__":
    unittest.main()
