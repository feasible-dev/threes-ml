"""Summarize frozen v2 metrics and episode logs without touching a live run."""

import argparse
import bisect
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def file_info(path):
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def summarize_episodes(rows):
    if not len(rows):
        return {"episodes": 0}
    score, moves, actions, reward, _, timeout = rows.T
    return {"episodes": len(rows), "mean_score": float(score.mean()),
            "median_score": float(np.median(score)),
            "mean_return": float(reward.mean()),
            "mean_valid_moves": float(moves.mean()),
            "mean_attempts": float(actions.mean()),
            "mean_invalid_attempts": float((actions - moves).mean()),
            "invalid_attempt_fraction": float((actions - moves).sum() / actions.sum()),
            "score_return_pearson": float(np.corrcoef(score, reward)[0, 1]),
            "timeouts": int(timeout.sum())}


def paired_validation(evaluation_dir):
    candidates = [(int(path.stem.split("_")[1]), path)
                  for path in evaluation_dir.glob("step_*.csv")]
    selected = [min(candidates, key=lambda item: abs(item[0] - target))
                for target in (60_000_000, 75_000_000)]
    scores = []
    for _, path in selected:
        with path.open(encoding="utf-8", newline="") as stream:
            scores.append({int(row["seed"]): int(row["score"]) for row in csv.DictReader(stream)})
    seeds = sorted(scores[0])
    assert seeds == sorted(scores[1]) and len(seeds) == 100
    earlier = np.array([scores[0][seed] for seed in seeds])
    later = np.array([scores[1][seed] for seed in seeds])
    difference = later - earlier
    rng = np.random.default_rng(710024)
    resampled = difference[rng.integers(0, len(seeds), (20_000, len(seeds)))].mean(axis=1)
    return {"selection_rule": "evaluation files closest to fixed 60M and 75M transition targets",
            "earlier_file": file_info(selected[0][1]), "later_file": file_info(selected[1][1]),
            "earlier_steps": selected[0][0], "later_steps": selected[1][0],
            "seeds": [seeds[0], seeds[-1]], "games": len(seeds),
            "earlier_mean": float(earlier.mean()), "later_mean": float(later.mean()),
            "earlier_median": float(np.median(earlier)), "later_median": float(np.median(later)),
            "paired_mean_difference": float(difference.mean()),
            "paired_median_difference": float(np.median(difference)),
            "wins": int((difference > 0).sum()), "losses": int((difference < 0).sum()),
            "paired_bootstrap_95_percentile_interval": np.quantile(resampled, [.025, .975]).tolist(),
            "bootstrap_draws": 20_000, "bootstrap_seed": 710024}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, help="A frozen before/data directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics_path, episode_path = args.snapshot / "metrics.csv", args.snapshot / "episodes.csv"
    with metrics_path.open(encoding="utf-8", newline="") as stream:
        metrics = list(csv.DictReader(stream))
    with episode_path.open(encoding="utf-8", newline="") as stream:
        episodes = np.asarray([[float(row[key]) for key in
                                ("score", "moves", "actions", "return", "curriculum", "truncated")]
                               for row in csv.DictReader(stream)])
    steps = [int(row["steps"]) for row in metrics]
    completed = [int(row["train_episodes"]) for row in metrics]
    assert steps == sorted(steps) and completed == sorted(completed)
    assert completed[-1] <= len(episodes)
    boundaries = [0] + list(range(10_000_000, steps[-1], 10_000_000)) + [steps[-1]]
    result = {"analyzed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
              "metrics": file_info(metrics_path), "episodes": file_info(episode_path),
              "last_logged_steps": steps[-1], "last_logged_completed_episodes": completed[-1],
              "episode_rows": len(episodes), "windows": [],
              "fixed_budget_validation_pair": paired_validation(args.snapshot.parent / "evaluation")}
    for low, high in zip(boundaries[:-1], boundaries[1:]):
        first = 0 if low == 0 else bisect.bisect_right(steps, low) - 1
        last = bisect.bisect_right(steps, high) - 1
        episode_first = 0 if low == 0 else completed[first]
        episode_last = completed[last]
        included_metrics = metrics[first + (low != 0):last + 1]
        window = {"requested_start_steps": low, "requested_end_steps": high,
                  "actual_start_steps": 0 if low == 0 else steps[first],
                  "actual_end_steps": steps[last],
                  "first_episode_row": episode_first, "last_episode_row": episode_last,
                  "updates": len(included_metrics)}
        for key in ("invalid_rate_recent", "approx_kl", "entropy_loss", "value_loss"):
            window["mean_" + key] = float(np.mean([float(row[key]) for row in included_metrics]))
        evaluations = [float(row["eval_mean"]) for row in included_metrics if row["eval_mean"]]
        window["validation_evaluations"] = len(evaluations)
        window["mean_validation_score"] = float(np.mean(evaluations)) if evaluations else None
        window["best_validation_score"] = max(evaluations) if evaluations else None
        selected = episodes[episode_first:episode_last]
        window["natural"] = summarize_episodes(selected[selected[:, 4] == 0])
        window["curriculum"] = summarize_episodes(selected[selected[:, 4] == 1])
        result["windows"].append(window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Analyzed {len(episodes):,} frozen episodes through {steps[-1]:,} logged transitions.")


if __name__ == "__main__":
    main()
