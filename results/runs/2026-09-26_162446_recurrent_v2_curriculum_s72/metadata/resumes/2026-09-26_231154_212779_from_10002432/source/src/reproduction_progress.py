"""Read-only, outcome-focused progress display for recurrent runs."""

import argparse
import csv
import re
import time
from pathlib import Path


TILES = (192, 384, 768, 1536, 3072, 6144)
STEP_LINE = re.compile(r"^\s*([\d,]+) steps \|.*$")


def latest_row(run):
    path = Path(run) / "data/metrics.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if (row.get("steps") or "").isdigit()]
    return rows[-1] if rows else None


def reach_rates(run, steps):
    path = Path(run) / "evaluation" / f"step_{steps:012d}.csv"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return None
    tiles = [int(row["max_tile"]) for row in rows]
    return len(tiles), {tile: sum(value >= tile for value in tiles) / len(tiles)
                        for tile in TILES}


def latest_evaluation_row(run, steps):
    path = Path(run) / "data/metrics.csv"
    result = None
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for candidate in csv.DictReader(stream):
            if ((candidate.get("steps") or "").isdigit() and
                    int(candidate["steps"]) <= steps and candidate.get("eval_mean")):
                result = candidate
    return result


def format_row(run, row):
    steps = int(row["steps"])
    pieces = [f"{steps:>12,} steps"]
    if row.get("steps_per_second"):
        pieces.append(f"{float(row['steps_per_second']):,.0f} steps/s")
    if row.get("invalid_rate_recent"):
        pieces.append(f"invalid {float(row['invalid_rate_recent']):.1%}")
    evaluated = row if row.get("eval_mean") else latest_evaluation_row(run, steps)
    if evaluated:
        eval_steps = int(evaluated["steps"])
        label = "validation" if eval_steps == steps else f"validation@{eval_steps:,}"
        pieces.append(f"{label} {float(evaluated['eval_mean']):,.1f}")
        reached = reach_rates(run, eval_steps)
        if reached:
            games, rates = reached
            pieces.append(f"{games} games")
            pieces.extend(f">={tile} {rates[tile]:.1%}" for tile in TILES)
    return " | ".join(pieces)


def format_trainer_line(run, line):
    match = STEP_LINE.match(line)
    if not match:
        return line
    steps = int(match[1].replace(",", ""))
    row = latest_row(run)
    if row is None or int(row["steps"]) != steps:
        return line
    return format_row(run, row) + ("\n" if line.endswith("\n") else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--interval", type=float, default=15)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    last_step = None
    try:
        while True:
            row = latest_row(args.run)
            if row and row["steps"] != last_step:
                print(format_row(args.run, row), flush=True)
                last_step = row["steps"]
            if not args.follow:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
