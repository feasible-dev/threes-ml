"""Read-only, outcome-focused progress display for recurrent runs."""

import argparse
import csv
import json
import time
from pathlib import Path


TILES = (192, 384, 768, 1536)


def latest_row(run):
    path = Path(run) / "data/metrics.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if (row.get("steps") or "").isdigit()]
    return rows[-1] if rows else None


def latest_evaluation_row(run, through_steps=None):
    path = Path(run) / "data/metrics.csv"
    result = None
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if ((row.get("steps") or "").isdigit() and row.get("eval_mean") and
                    (through_steps is None or int(row["steps"]) <= through_steps)):
                result = row
    return result


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


def format_row(run, row):
    steps = int(row["steps"])
    pieces = [f"{steps:>12,} steps"]
    if row.get("eval_mean"):
        pieces.append(f"validation avg score {float(row['eval_mean']):,.1f}")
        reached = reach_rates(run, steps)
        if reached:
            _, rates = reached
            pieces.extend(f">={tile} {rates[tile]:.1%}" for tile in TILES)
    if row.get("steps_per_second"):
        pieces.append(f"{float(row['steps_per_second']):,.0f} steps/s")
    return " | ".join(pieces)


def terminal_status_line(run):
    status_path = Path(run) / "metadata/status.json"
    status = json.loads(status_path.read_text(encoding="utf-8-sig")).get("status")
    if status not in ("complete", "interrupted", "failed"):
        return None
    if status == "complete":
        return "Run complete; no more updates are expected."
    markers = sorted((Path(run) / "models/checkpoints").glob("step_*/complete.json"))
    if markers:
        steps = int(json.loads(markers[-1].read_text(encoding="utf-8-sig"))["steps"])
        return f"Run {status}; last complete checkpoint: {steps:,} steps. No more updates are expected."
    return f"Run {status}; no complete checkpoint is available. No more updates are expected."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--interval", type=float, default=15)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    last_evaluation_step = None
    try:
        while True:
            evaluation = latest_evaluation_row(args.run)
            if evaluation and evaluation["steps"] != last_evaluation_step:
                print(format_row(args.run, evaluation), flush=True)
                last_evaluation_step = evaluation["steps"]
            if not args.follow:
                break
            finished = terminal_status_line(args.run)
            if finished:
                print(finished, flush=True)
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
