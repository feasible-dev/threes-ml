"""Comparer quatre politiques simples sur des parties complètes sans affichage."""

import argparse
import csv
import math
import random
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from random_agent import play_game, positive_int, random_move
from threes import Board, Direction, Hint, move, score, spawn_positions


def greedy_move(
    board: Board, hint: Hint, available: tuple[Direction, ...], policy_random: random.Random
) -> Direction:
    """Maximise le gain de score immédiat, puis les cases libres avant apparition."""
    candidates: list[Direction] = []
    best: tuple[int, int] | None = None
    current_score = score(board)
    for direction in available:
        moved = move(board, direction)
        priority = (
            score(moved) - current_score,
            sum(value == 0 for row in moved for value in row),
        )
        if best is None or priority > best:
            best = priority
            candidates = [direction]
        elif priority == best:
            candidates.append(direction)
    return policy_random.choice(candidates)


def position_value(board: Board) -> int:
    """Valorise l'espace et des tuiles proches en rang, sans information cachée."""
    empty = sum(value == 0 for row in board for value in row)
    ranks = tuple(
        tuple(0 if value == 0 else 1 if value < 3 else (value // 3).bit_length() + 1 for value in row)
        for row in board
    )
    roughness = 0
    for row in range(4):
        for column in range(4):
            here = ranks[row][column]
            if not here:
                continue
            if row < 3 and ranks[row + 1][column]:
                roughness += abs(here - ranks[row + 1][column])
            if column < 3 and ranks[row][column + 1]:
                roughness += abs(here - ranks[row][column + 1])
    maximum = max(value for row in board for value in row)
    corner_maximum = maximum in (board[0][0], board[0][3], board[3][0], board[3][3])
    return 25 * empty - 2 * roughness + 8 * corner_maximum


def position_move(
    board: Board, hint: Hint, available: tuple[Direction, ...], policy_random: random.Random
) -> Direction:
    """Évalue le plateau après déplacement, avant l'apparition aléatoire."""
    candidates: list[Direction] = []
    best = -math.inf
    current_score = score(board)
    for direction in available:
        moved = move(board, direction)
        gain = score(moved) - current_score
        priority = position_value(moved) + 4 * math.log2(gain + 1)
        if priority > best:
            best = priority
            candidates = [direction]
        elif priority == best:
            candidates.append(direction)
    return policy_random.choice(candidates)


def expected_spawn_move(
    board: Board, hint: Hint, available: tuple[Direction, ...], policy_random: random.Random
) -> Direction:
    """Moyenne les valeurs visibles et les cases possibles de la prochaine tuile."""
    possible_tiles = hint if isinstance(hint, tuple) else (hint,)
    candidates: list[Direction] = []
    best = -math.inf
    current_score = score(board)
    for direction in available:
        moved = move(board, direction)
        if any(12288 in row for row in moved):
            return direction
        values = []
        for row, column in spawn_positions(board, moved, direction):
            for tile in possible_tiles:
                updated = [list(board_row) for board_row in moved]
                updated[row][column] = tile
                spawned = tuple(tuple(board_row) for board_row in updated)
                values.append(position_value(spawned))
        gain = score(moved) - current_score
        priority = mean(values) + 4 * math.log2(gain + 1)
        if priority > best:
            best = priority
            candidates = [direction]
        elif priority == best:
            candidates.append(direction)
    return policy_random.choice(candidates)


def evaluate(games: int, seed: int) -> tuple[dict[str, list[tuple[int, int, int, int]]], dict[str, float]]:
    results: dict[str, list[tuple[int, int, int, int]]] = {}
    elapsed: dict[str, float] = {}
    for name, policy in (
        ("aléatoire", random_move),
        ("fusion", greedy_move),
        ("position", position_move),
        ("apparition", expected_spawn_move),
    ):
        start = perf_counter()
        results[name] = [
            (game_seed, *play_game(game_seed, policy))
            for game_seed in range(seed, seed + games)
        ]
        elapsed[name] = perf_counter() - start
    return results, elapsed


def write_csv(path: Path, results: dict[str, list[tuple[int, int, int, int]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("agent", "seed", "score", "moves", "max_tile"))
        for agent, games in results.items():
            writer.writerows((agent, *game) for game in games)


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparer quatre agents Threes sans affichage")
    parser.add_argument("--games", type=positive_int, default=100, help="parties par agent (défaut : 100)")
    parser.add_argument("--seed", type=int, default=42, help="première graine (défaut : 42)")
    parser.add_argument("--csv", type=Path, help="enregistrer une ligne par partie dans ce fichier")
    args = parser.parse_args()

    results, elapsed = evaluate(args.games, args.seed)
    display_names = {"aléatoire": "random", "fusion": "merge",
                     "position": "position", "apparition": "spawn"}
    print(f"{args.games} games per agent | seeds {args.seed} to {args.seed + args.games - 1}")
    print("Agent       Mean     Median   Maximum  Moves    Games/s    Moves/s")
    for name, games in results.items():
        scores = [game[1] for game in games]
        moves = sum(game[2] for game in games)
        print(
            f"{display_names[name]:<11} {mean(scores):>7.1f}  {median(scores):>7.1f}  "
            f"{max(scores):>7}  {moves:>5}  {len(games) / elapsed[name]:>10.1f}  "
            f"{moves / elapsed[name]:>7.1f}"
        )
    print("\nAgent       Max tile   >=96    >=192   >=384")
    for name, games in results.items():
        largest = [game[3] for game in games]
        rates = [100 * sum(value >= threshold for value in largest) / len(largest) for threshold in (96, 192, 384)]
        print(f"{display_names[name]:<11} {max(largest):>9}  {rates[0]:>5.1f}%  {rates[1]:>5.1f}%  {rates[2]:>5.1f}%")
    if args.csv:
        write_csv(args.csv, results)
        print(f"Details: {args.csv}")


if __name__ == "__main__":
    main()
