import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from main import human_mode_enabled, main, play, print_state
from threes import Game


class HumanModeTests(unittest.TestCase):
    def test_terminal_shows_bonus_choices(self):
        board = ((192, 0, 0, 0),) + ((0, 0, 0, 0),) * 3
        game = Game(board, seed=4)
        game.next_hint = (6, 12, 24)
        output = io.StringIO()
        with redirect_stdout(output):
            print_state(game)
        self.assertIn("Prochaine tuile : 6 / 12 / 24", output.getvalue())

    def test_no_argument_launches_gui(self):
        with (
            patch("sys.argv", ["main.py"]),
            patch("main.human_mode_enabled", return_value=False),
            patch("gui.run_gui") as run_gui,
        ):
            main()

        run_gui.assert_called_once_with(42)

    def test_demo_remains_available_explicitly(self):
        with (
            patch("sys.argv", ["main.py", "--demo", "--seed", "7"]),
            patch("main.demo") as demo,
        ):
            main()

        demo.assert_called_once_with(7)

    def test_config_switch_controls_human_mode(self):
        with patch("main.CONFIG_PATH") as config_path:
            config_path.read_text.return_value = '{"human_mode": true}'
            self.assertTrue(human_mode_enabled())
            config_path.read_text.return_value = '{"human_mode": false}'
            self.assertFalse(human_mode_enabled())

    def test_blocked_direction_is_marked_and_explained(self):
        board = (
            (6, 3, 3, 3),
            (2, 2, 1, 0),
            (0, 3, 0, 0),
            (0, 1, 0, 0),
        )
        game = Game(board, seed=42)
        next_hint = game.next_hint
        output = io.StringIO()
        with (
            patch("main.Game", return_value=game),
            patch("builtins.input", side_effect=("w", "x")),
            redirect_stdout(output),
        ):
            play(seed=42)

        self.assertIn("[Z/W]*", output.getvalue())
        self.assertIn("Coup impossible (W)", output.getvalue())
        self.assertEqual((game.board, game.next_hint), (board, next_hint))


if __name__ == "__main__":
    unittest.main()
