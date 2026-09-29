"""Evaluate the published ONNX policy and our heuristics on the same local game."""

import argparse
import csv
import hashlib
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import onnxruntime as ort

from compare_agents import evaluate as evaluate_heuristics, write_csv as write_heuristics_csv
from external_agent import DEFAULT_MODEL, KiokOnnxAgent
from random_agent import positive_int
from run_layout import PROJECT_ROOT
from threes import Game


def summarize(games: list[tuple[int, int, int, int]]) -> tuple[float, float, int, list[float]]:
    scores = [score for _, score, _, _ in games]
    largest = [tile for _, _, _, tile in games]
    rates = [sum(tile >= threshold for tile in largest) / len(largest)
             for threshold in (192, 384, 3072, 6144)]
    return mean(scores), median(scores), max(scores), rates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--games", type=positive_int, default=10)
    parser.add_argument("--seed", type=int, default=400000)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir or (
        PROJECT_ROOT / "results" / "comparisons" /
        f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_kiok_transfer"
    )
    if output_dir.exists():
        parser.error(f"Output directory already exists: {output_dir}")
    model_hash = hashlib.sha256(args.model.read_bytes()).hexdigest()
    agent = KiokOnnxAgent(args.model)
    print(f"Model: {args.model.resolve()}\nSHA-256: {model_hash}", flush=True)
    print(f"Playing {args.games} games on seeds {args.seed} to {args.seed + args.games - 1}", flush=True)

    model_games = []
    start = perf_counter()
    for index, seed in enumerate(range(args.seed, args.seed + args.games), 1):
        game = Game(seed=seed)
        agent.reset()
        moves = 0
        while not game.is_over and moves < 10000:
            game.step(agent.choose_action(game))
            moves += 1
        if not game.is_over:
            raise RuntimeError(f"Game {seed} did not finish after 10,000 moves")
        tile = max(max(row) for row in game.board)
        model_games.append((seed, game.current_score, moves, tile))
        if args.games <= 20 or index % 10 == 0 or index == args.games:
            print(f"  {index}/{args.games}: seed {seed}, score {game.current_score}, "
                  f"tile {tile}, {moves} moves", flush=True)
    elapsed = perf_counter() - start

    print("Evaluating the four heuristics on the same seeds...", flush=True)
    heuristic_games, _ = evaluate_heuristics(args.games, args.seed)
    output_dir.mkdir(parents=True)
    with (output_dir / "onnx_games.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("agent", "seed", "score", "moves", "max_tile"))
        writer.writerows(("kiok_onnx", *game) for game in model_games)
    write_heuristics_csv(output_dir / "heuristics.csv", heuristic_games)

    labels = {"aléatoire": "random", "fusion": "immediate merge",
              "position": "position", "apparition": "expected spawn"}
    rows = [("Kiok ONNX", summarize(model_games))]
    rows += [(labels[name], summarize(games)) for name, games in heuristic_games.items()]
    report = [
        "# Published ONNX agent on our Threes! variant", "",
        f"Model: [pseudonam-gc/threes-web](https://github.com/pseudonam-gc/threes-web), "
        f"commit `9527295de72333c91592fff45e8ccbcabd6ba80d`; SHA-256 `{model_hash}`.",
        f"Runtime: ONNX Runtime {ort.__version__} on CPU; {elapsed:.1f} s for model play.",
        f"Protocol: {args.games} complete games, initial seeds {args.seed} to {args.seed + args.games - 1}, "
        "deterministic highest-logit legal action, LSTM state reset between games.",
        "All agents use our game engine and the same initial seeds; their later draws can diverge.", "",
        "| Agent | Mean score | Median | Maximum | Tile >=192 | >=384 | >=3072 | >=6144 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, (average, middle, maximum, rates) in rows:
        report.append(f"| {name} | {average:,.1f} | {middle:,.1f} | {maximum:,} | "
                      + " | ".join(f"{rate:.1%}" for rate in rates) + " |")
    report += [
        "", "[ONNX per-game results](onnx_games.csv) | [Heuristic per-game results](heuristics.csv).", "",
        "The upstream training rules and bonus timing may differ from this engine. "
        "The upstream README does not identify whether this ONNX file is the later checkpoint "
        "used for the reported 10,000-game result. This table is a local transfer benchmark.", "",
    ]
    (output_dir / "comparison.md").write_text("\n".join(report), encoding="utf-8")
    for name, (average, middle, maximum, rates) in rows:
        print(f"{name}: mean {average:,.1f}, median {middle:,.1f}, max {maximum:,}, "
              f"tile >=3072 {rates[2]:.1%}, tile >=6144 {rates[3]:.1%}", flush=True)
    print(f"Report: {output_dir / 'comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
