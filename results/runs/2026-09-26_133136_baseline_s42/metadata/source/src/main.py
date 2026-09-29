import argparse
import json
import os
import sys
from pathlib import Path

from threes import Board, Direction, Game, legal_moves


KEYS: dict[str, Direction] = {
    "z": "up",
    "w": "up",
    "s": "down",
    "q": "left",
    "a": "left",
    "d": "right",
}
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
TILE_BACKGROUNDS = {
    1: "\x1b[48;2;173;223;245m\x1b[30m",
    2: "\x1b[48;2;255;185;204m\x1b[30m",
}
RESET_COLOR = "\x1b[0m"


def human_mode_enabled() -> bool:
    try:
        settings = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False
    if not isinstance(settings, dict):
        raise ValueError("config.json doit contenir un objet JSON")
    human_mode = settings.get("human_mode", False)
    if not isinstance(human_mode, bool):
        raise ValueError("config.json : human_mode doit etre true ou false")
    return human_mode


def format_moves(available: tuple[Direction, ...]) -> str:
    def label(keys: str, direction: Direction) -> str:
        return f"[{keys}]{'*' if direction not in available else ' '}"

    return "\n".join(
        (
            f"       {label('Z/W', 'up')}",
            f"{label('Q/A', 'left')}  [ ]  {label('D', 'right')}",
            f"        {label('S', 'down')}",
            "* = coup impossible",
        )
    )


def print_board(board: Board) -> None:
    width = max(3, *(len(str(value)) for row in board for value in row))
    use_color = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    for row in board:
        cells = []
        for value in row:
            cell = f"{value if value else '.':^{width}}"
            if use_color and value in TILE_BACKGROUNDS:
                cell = f"{TILE_BACKGROUNDS[value]}{cell}{RESET_COLOR}"
            cells.append(cell)
        print(" ".join(cells))


def print_state(game: Game) -> None:
    print()
    print_board(game.board)
    next_hint = "-" if game.is_over else game.next_hint
    if isinstance(next_hint, tuple):
        next_hint = " / ".join(str(value) for value in next_hint)
    print(f"Score : {game.current_score} | Prochaine tuile : {next_hint}")


def demo(seed: int) -> None:
    game = Game(seed=seed)
    print(f"Threes ML - demonstration (graine {seed})")
    print_state(game)
    direction = legal_moves(game.board)[0]
    game.step(direction)
    print(f"\nApres un coup {direction} :")
    print_state(game)


def play(seed: int) -> None:
    game = Game(seed=seed)
    print(f"Threes ML - partie (graine {seed})")
    print("ZQSD ou WASD pour bouger, X pour quitter.")
    while not game.is_over:
        available = legal_moves(game.board)
        print_state(game)
        print(format_moves(available))
        while True:
            try:
                key = input("Coup : ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                print(f"Score final : {game.current_score}")
                return
            if key == "x":
                print(f"Score final : {game.current_score}")
                return
            direction = KEYS.get(key)
            if direction is None:
                print("Touche inconnue. Utilise ZQSD, WASD ou X.")
            elif direction not in available:
                print(f"Coup impossible ({key.upper()}) : aucune tuile ne bouge dans cette direction.")
            else:
                game.step(direction)
                break
    else:
        print_state(game)
        print("Partie terminee.")
    print(f"Score final : {game.current_score}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Jouer a Threes ML")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--play", action="store_true", help="jouer dans le terminal")
    mode.add_argument("--gui", action="store_true", help="ouvrir une fenetre graphique (par defaut)")
    mode.add_argument("--demo", action="store_true", help="afficher une demonstration dans le terminal")
    parser.add_argument("--seed", type=int, default=42, help="graine aleatoire")
    args = parser.parse_args()
    if args.demo:
        demo(args.seed)
    elif args.play or (not args.gui and human_mode_enabled()):
        play(args.seed)
    else:
        from gui import run_gui

        run_gui(args.seed)


if __name__ == "__main__":
    main()
