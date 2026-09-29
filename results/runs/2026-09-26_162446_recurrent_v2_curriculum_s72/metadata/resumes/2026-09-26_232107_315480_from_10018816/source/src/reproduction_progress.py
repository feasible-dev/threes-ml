"""Read-only, outcome-focused progress display for recurrent runs."""

import argparse
import csv
import json
import re
import time
from pathlib import Path


TILES = (192, 384, 768, 1536)
STEP_LINE = re.compile(r"^\s*([\d,]+) steps \|.*$")


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
    if row.get("steps_per_second"):
        pieces.append(f"{float(row['steps_per_second']):,.0f} steps/s")
    if row.get("invalid_rate_recent"):
        pieces.append(f"invalid {float(row['invalid_rate_recent']):.1%}")
    evaluated = row if row.get("eval_mean") else latest_evaluation_row(run, steps)
    if evaluated:
        label = f"validation avg score {float(evaluated['eval_mean']):,.1f}"
        if int(evaluated["steps"]) != steps:
            label += f" (at {int(evaluated['steps']):,})"
        pieces.append(label)
    if row.get("eval_mean"):
        reached = reach_rates(run, steps)
        if reached:
            games, rates = reached
            pieces.append(f"{games} games")
            pieces.extend(f">={tile} {rates[tile]:.1%}" for tile in TILES)
    return " | ".join(pieces)


def format_trainer_line(run, line, *, resuming=False):
    if resuming and line.startswith("Training starts from random weights for a new run."):
        return "Continuing from the saved checkpoint. The simulator may compile on first use.\n"
    match = STEP_LINE.match(line)
    if not match:
        return line
    steps = int(match[1].replace(",", ""))
    row = latest_row(run)
    if row is None or int(row["steps"]) != steps:
        return line
    return format_row(run, row) + ("\n" if line.endswith("\n") else "")


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
    last_step = None
    try:
        while True:
            row = latest_row(args.run)
            if row and row["steps"] != last_step:
                if last_step is None:
                    evaluation = latest_evaluation_row(args.run)
                    if evaluation and evaluation["steps"] != row["steps"]:
                        print("Latest validation: " + format_row(args.run, evaluation), flush=True)
                print(format_row(args.run, row), flush=True)
                last_step = row["steps"]
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
