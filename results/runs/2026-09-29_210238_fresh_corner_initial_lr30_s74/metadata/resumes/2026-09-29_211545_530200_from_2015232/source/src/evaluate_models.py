"""Évaluer des checkpoints PPO sur les mêmes graines réservées."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import torch

from random_agent import positive_int
from train import TrackedPPO, evaluate_games


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparer des checkpoints PPO sauvegardés")
    parser.add_argument("models", nargs="+", type=Path, help="fichiers model.zip à évaluer")
    parser.add_argument("--games", type=positive_int, default=100)
    parser.add_argument("--seed", type=int, default=200000)
    parser.add_argument("--csv", required=True, type=Path, help="scores individuels")
    parser.add_argument("--report", type=Path, help="Markdown comparison report")
    parser.add_argument("--reference-csv", type=Path,
                        help="heuristic results on the same seeds")
    args = parser.parse_args()

    torch.set_num_threads(1)
    rows = []
    summaries = []
    for path in args.models:
        model = TrackedPPO.load(path, device="cpu")
        games = evaluate_games(model, args.games, args.seed)
        scores = [score for score, _ in games]
        name = path.parent.parent.name if path.parent.name == "models" else path.parent.name
        rows.extend((name, args.seed + index, score, tile)
                    for index, (score, tile) in enumerate(games))
        rates = tuple(sum(tile >= threshold for _, tile in games) / len(games)
                      for threshold in (192, 384, 3072, 6144))
        summaries.append((name, mean(scores), median(scores), max(scores), *rates))
        print(f"{name}: mean {mean(scores):.2f}, median {median(scores):.1f}, "
              f"maximum {max(scores)}, tile >=192 {rates[0]:.1%}, "
              f">=384 {rates[1]:.1%}, >=3072 {rates[2]:.1%}, "
              f">=6144 {rates[3]:.1%} over {len(scores)} games", flush=True)

    if args.reference_csv:
        references = defaultdict(list)
        with args.reference_csv.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                references[row["agent"]].append(
                    (int(row["seed"]), int(row["score"]), int(row["max_tile"])))
        expected_seeds = set(range(args.seed, args.seed + args.games))
        for name, games in references.items():
            if len(games) != args.games or {seed for seed, _, _ in games} != expected_seeds:
                raise ValueError(f"Reference {name} does not match the held-out seeds")
            scores = [score for _, score, _ in games]
            english_name = {"aléatoire": "random", "fusion": "immediate merge",
                            "position": "position", "apparition": "expected spawn"}.get(name, name)
            rates = tuple(sum(tile >= threshold for _, _, tile in games) / len(games)
                          for threshold in (192, 384, 3072, 6144))
            summaries.append((f"Heuristic: {english_name}", mean(scores), median(scores),
                              max(scores), *rates))

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("agent", "seed", "score", "largest_tile"))
        writer.writerows(rows)
    print(f"Scores: {args.csv}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Held-out model comparison", "",
            f"Each agent played {args.games} complete games on initial seeds "
            f"{args.seed}–{args.seed + args.games - 1}. These seeds were reserved for "
            "evaluation, but individual training episode seeds were not logged or checked for overlap.",
            "Different policies can encounter different draws after making different moves.", "",
            "| Run | Mean score | Median | Maximum | Tile >=192 | >=384 | >=3072 | >=6144 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        lines.extend(
            f"| {name} | {average:,.1f} | {middle:,.1f} | {maximum:,} | "
            f"{reach_192:.1%} | {reach_384:.1%} | {reach_3072:.1%} | {reach_6144:.1%} |"
            for name, average, middle, maximum, reach_192, reach_384,
                reach_3072, reach_6144 in summaries
        )
        lines += ["", f"Model per-game data: [{args.csv.name}]({args.csv.name})."]
        if args.reference_csv:
            lines.append(f"Heuristic per-game data: "
                         f"[{args.reference_csv.name}]({args.reference_csv.name}).")
        lines += ["",
                  "Interpretation and retention decisions remain pending review.", ""]
        args.report.write_text("\n".join(lines), encoding="utf-8")
        print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
