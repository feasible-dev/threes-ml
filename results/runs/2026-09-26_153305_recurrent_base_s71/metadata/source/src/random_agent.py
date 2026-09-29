"""Référence simple : parties complètes avec des coups légaux tirés au hasard."""

import argparse
import random
from statistics import mean, median
from time import perf_counter
from typing import Callable

from threes import Board, Direction, Game, Hint, legal_moves


Policy = Callable[[Board, Hint, tuple[Direction, ...], random.Random], Direction]


def random_move(
    board: Board, hint: Hint, available: tuple[Direction, ...], policy_random: random.Random
) -> Direction:
    return policy_random.choice(available)


def play_game(seed: int, choose_move: Policy) -> tuple[int, int, int]:
    """Renvoie (score, nombre de coups, plus grande tuile) pour une partie."""
    game = Game(seed=seed)
    policy_random = random.Random(seed)
    moves = 0
    while not any(12288 in row for row in game.board):
        available = legal_moves(game.board)
        if not available:
            break
        direction = choose_move(game.board, game.next_hint, available, policy_random)
        if not game.step(direction):
            raise ValueError(f"L'agent a choisi un coup impossible : {direction}")
        moves += 1
    return game.current_score, moves, max(value for row in game.board for value in row)


def play_random_game(seed: int) -> tuple[int, int, int]:
    return play_game(seed, random_move)


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("le nombre doit être positif")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description="Mesurer un agent Threes à coups légaux aléatoires")
    parser.add_argument("--games", type=positive_int, default=100, help="nombre de parties (défaut : 100)")
    parser.add_argument("--seed", type=int, default=42, help="première graine (défaut : 42)")
    args = parser.parse_args()

    start = perf_counter()
    results = [play_random_game(args.seed + index) for index in range(args.games)]
    elapsed = perf_counter() - start
    scores = [score for score, _, _ in results]
    moves = sum(turns for _, turns, _ in results)

    print(f"Agent aléatoire | {args.games} parties | graines {args.seed} à {args.seed + args.games - 1}")
    print(f"Score moyen : {mean(scores):.1f} | médian : {median(scores):.1f} | maximum : {max(scores)}")
    print(f"Coups : {moves} | temps : {elapsed:.3f} s")
    print(f"Débit : {args.games / elapsed:.1f} parties/s | {moves / elapsed:.1f} coups/s")


if __name__ == "__main__":
    main()
