import unittest
from unittest.mock import patch

import pygame

from gui import (
    BOARD_LEFT,
    BOARD_TOP,
    TILE_COLORS,
    TILE_GAP,
    TILE_HEIGHT,
    TILE_WIDTH,
    WHITE_TILE,
    WINDOW_SIZE,
    draw_game,
)
from threes import Game


class GuiTests(unittest.TestCase):
    def test_one_and_two_have_distinct_tile_colors(self):
        self.assertGreater(TILE_HEIGHT, TILE_WIDTH)
        board = (
            (1, 2, 3, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        )
        pygame.font.init()
        try:
            surface = pygame.Surface(WINDOW_SIZE)
            fonts = {
                size: pygame.font.Font(None, size)
                for size in (20, 22, 24, 26, 31, 38, 39, 46, 50, 62)
            }
            draw_game(surface, Game(board, seed=42), fonts, "", 42)
            self.assertEqual(surface.get_at((BOARD_LEFT + 8, BOARD_TOP + 8))[:3], TILE_COLORS[1])
            self.assertEqual(
                surface.get_at((BOARD_LEFT + TILE_WIDTH + TILE_GAP + 8, BOARD_TOP + 8))[:3],
                TILE_COLORS[2],
            )
            self.assertEqual(
                surface.get_at((BOARD_LEFT + 2 * (TILE_WIDTH + TILE_GAP) + 8, BOARD_TOP + 8))[:3],
                WHITE_TILE,
            )
        finally:
            pygame.font.quit()

    def test_bonus_preview_displays_all_three_values(self):
        board = ((192, 0, 0, 0),) + ((0, 0, 0, 0),) * 3
        game = Game(board, seed=4)
        game.next_hint = (6, 12, 24)
        pygame.font.init()
        try:
            surface = pygame.Surface(WINDOW_SIZE)
            fonts = {
                size: pygame.font.Font(None, size)
                for size in (20, 22, 24, 26, 31, 38, 39, 46, 50, 62)
            }
            with patch("gui._draw_centered") as draw_text:
                draw_game(surface, game, fonts, "", 4)
            self.assertTrue(any(call.args[1] == "6 / 12 / 24" for call in draw_text.call_args_list))
        finally:
            pygame.font.quit()


if __name__ == "__main__":
    unittest.main()
