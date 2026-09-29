import unittest
from collections import Counter

from threes import Game, legal_moves, move, score


EMPTY_ROW = (0, 0, 0, 0)


class GameTests(unittest.TestCase):
    def test_initial_board_uses_nine_tiles_from_the_bag(self):
        game = Game(seed=42)
        cells = [value for row in game.board for value in row]
        self.assertEqual(cells.count(0), 7)
        self.assertEqual(len(game.board), 4)
        self.assertTrue(all(len(row) == 4 for row in game.board))
        self.assertEqual(
            Counter(cells + [game.next_hint] + [game._draw_tile() for _ in range(2)]),
            {0: 7, 1: 4, 2: 4, 3: 4},
        )

    def test_visible_counts_include_initial_board_and_announced_tile(self):
        game = Game(seed=42)
        board_counts = Counter(value for row in game.board for value in row)
        expected = tuple(board_counts[value] + (game.next_hint == value) for value in (1, 2, 3))
        self.assertEqual(game.visible_ordinary_counts, expected)
        self.assertEqual(sum(expected), 10)

        before = game.visible_ordinary_counts
        direction = legal_moves(game.board)[0]
        self.assertTrue(game.step(direction))
        after = game.visible_ordinary_counts
        self.assertEqual(sum(after), sum(before) + isinstance(game.next_hint, int))

    def test_initial_board_repeats_with_same_seed(self):
        first = Game(seed=17)
        second = Game(seed=17)
        self.assertEqual((first.board, first.next_hint), (second.board, second.next_hint))

    def test_new_tile_appears_on_opposite_edge_of_changed_line(self):
        cases = (
            (
                "left",
                ((1, 2, 0, 0), (1, 1, 1, 3), (3, 6, 12, 24), (1, 1, 1, 3)),
                (0, 3),
            ),
            ("right", (EMPTY_ROW, EMPTY_ROW, (0, 3, 0, 0), EMPTY_ROW), (2, 0)),
            ("up", (EMPTY_ROW, EMPTY_ROW, (0, 3, 0, 0), EMPTY_ROW), (3, 1)),
            ("down", (EMPTY_ROW, (0, 3, 0, 0), EMPTY_ROW, EMPTY_ROW), (0, 1)),
        )
        for direction, board, spawn_position in cases:
            with self.subTest(direction=direction):
                game = Game(board, seed=7)
                incoming_tile = game.next_hint
                moved = move(board, direction)
                self.assertTrue(game.step(direction))
                for row in range(4):
                    for column in range(4):
                        expected = incoming_tile if (row, column) == spawn_position else moved[row][column]
                        self.assertEqual(game.board[row][column], expected)

    def test_invalid_move_does_not_consume_randomness_or_tile(self):
        board = ((1, 1, 1, 3), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW)
        game = Game(board, seed=19)
        reference = Game(board, seed=19)

        self.assertFalse(game.step("left"))
        self.assertEqual(game.board, board)
        self.assertEqual(game.next_hint, reference.next_hint)
        self.assertTrue(game.step("down"))
        self.assertTrue(reference.step("down"))
        self.assertEqual((game.board, game.next_hint), (reference.board, reference.next_hint))

    def test_spawn_uses_only_changed_lines(self):
        board = (
            (1, 2, 0, 0),
            (1, 1, 1, 3),
            (0, 3, 0, 0),
            (3, 6, 12, 24),
        )
        for seed in range(10):
            with self.subTest(seed=seed):
                game = Game(board, seed=seed)
                incoming_tile = game.next_hint
                self.assertTrue(game.step("left"))
                self.assertEqual(game.board[1], board[1])
                self.assertEqual(game.board[3], board[3])
                self.assertEqual(
                    (game.board[0][3], game.board[2][3]).count(incoming_tile), 1
                )

    def test_bag_contains_four_of_each_ordinary_tile(self):
        game = Game((EMPTY_ROW,) * 4, seed=3)
        drawn = [game.next_hint]
        drawn.extend(game._draw_tile() for _ in range(11))
        self.assertEqual(Counter(drawn), {1: 4, 2: 4, 3: 4})

    def test_same_seed_reproduces_sequence(self):
        board = (
            (1, 2, 0, 0),
            (3, 3, 3, 3),
            (0, 1, 2, 0),
            (6, 0, 0, 6),
        )
        first = Game(board, seed=42)
        second = Game(board, seed=42)
        for direction in ("left", "up", "right", "down"):
            self.assertEqual(first.step(direction), second.step(direction))
            self.assertEqual((first.board, first.next_hint), (second.board, second.next_hint))

    def test_bonus_hint_shows_consecutive_choices_and_preserves_bag(self):
        board = ((384, 0, 0, 0), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW)
        game = Game(board, seed=4)
        remaining = game._bag.copy()
        original_randrange = game._random.randrange
        game._random.randrange = lambda limit: 0
        try:
            game.next_hint = game._draw_hint()
        finally:
            game._random.randrange = original_randrange

        self.assertEqual(game.next_hint, (6, 12, 24))
        self.assertEqual(game._bag, remaining)
        self.assertFalse(hasattr(game, "next_tile"))
        self.assertTrue(game.step("right"))
        self.assertIn(game.board[0][0], (6, 12, 24))

    def test_bonus_hint_can_show_the_next_window(self):
        board = ((384, 0, 0, 0), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW)
        game = Game(board, seed=4)
        draws = iter((0, 1))
        game._random.randrange = lambda limit: next(draws)
        self.assertEqual(game._draw_hint(), (12, 24, 48))

    def test_short_bonus_pool_shows_every_possible_value(self):
        for maximum, expected in ((48, (6,)), (96, (6, 12)), (192, (6, 12, 24))):
            with self.subTest(maximum=maximum):
                board = ((maximum, 0, 0, 0), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW)
                game = Game(board, seed=4)
                game._random.randrange = lambda limit: 0
                self.assertEqual(game._draw_hint(), expected)

    def test_bonus_is_unavailable_below_48(self):
        board = ((24, 0, 0, 0), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW)
        game = Game(board, seed=4)
        game._random.randrange = lambda limit: 0
        self.assertIn(game._draw_hint(), (1, 2, 3))

    def test_score_and_game_over(self):
        board = (
            (0, 1, 2, 3),
            (6, 12, 24, 48),
            EMPTY_ROW,
            EMPTY_ROW,
        )
        self.assertEqual(score(board), 363)
        self.assertEqual(Game(board, seed=1).current_score, 363)

        blocked = (
            (1, 3, 1, 3),
            (3, 1, 3, 1),
            (1, 3, 1, 3),
            (3, 1, 3, 1),
        )
        game = Game(blocked, seed=1)
        self.assertTrue(game.is_over)
        self.assertFalse(game.step("left"))

    def test_invalid_board_value_is_rejected(self):
        board = ((5, 0, 0, 0), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW)
        with self.assertRaises(ValueError):
            Game(board, seed=1)

    def test_12288_ends_game_before_new_tile_appears(self):
        board = ((6144, 6144, 0, 0), EMPTY_ROW, EMPTY_ROW, EMPTY_ROW)
        game = Game(board, seed=1)
        self.assertTrue(game.step("left"))
        self.assertEqual(game.board[0], (12288, 0, 0, 0))
        self.assertTrue(game.is_over)
        self.assertEqual(game.current_score, 1594323)
        self.assertFalse(game.step("right"))

    def test_seeded_games_reach_a_terminal_board(self):
        for seed in range(5):
            with self.subTest(seed=seed):
                game = Game(seed=seed)
                for _ in range(1000):
                    if game.is_over:
                        break
                    self.assertTrue(game.step(legal_moves(game.board)[0]))
                self.assertTrue(game.is_over)
                self.assertGreater(game.current_score, 0)

    def test_ordinary_bag_bounds_blue_red_imbalance(self):
        for seed in range(20):
            game = Game(seed=seed)
            while True:
                cells = [value for row in game.board for value in row]
                self.assertLessEqual(abs(cells.count(1) - cells.count(2)), 4)
                if game.is_over:
                    break
                self.assertTrue(game.step(legal_moves(game.board)[0]))


if __name__ == "__main__":
    unittest.main()
