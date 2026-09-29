"""Paired natural-game comparison of fixed-budget fork endpoints and best models."""

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from reproduction.evaluation import evaluate
from reproduction.training import TrackedRecurrentPPO
from run_layout import RUNS_ROOT


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def score_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return {int(row["seed"]): {"score": int(row["score"]),
                                   "max_tile": int(row["max_tile"]),
                                   "moves": int(row["moves"]),
                                   "attempts": int(row["actions"]),
                                   "truncated": row["truncated"].lower() == "true"}
                for row in csv.DictReader(stream)}


def summarize(rows):
    values = list(rows.values())
    scores = np.array([row["score"] for row in values])
    tiles = np.array([row["max_tile"] for row in values])
    return {"games": len(values), "mean_score": float(scores.mean()),
            "median_score": float(np.median(scores)),
            "mean_valid_moves": float(np.mean([row["moves"] for row in values])),
            "mean_attempts": float(np.mean([row["attempts"] for row in values])),
            "reach_384": float(np.mean(tiles >= 384)),
            "reach_3072": float(np.mean(tiles >= 3072)),
            "reach_6144": float(np.mean(tiles >= 6144)),
            "truncated": sum(row["truncated"] for row in values)}


def paired_result(left, right, *, seed=820999):
    seeds = sorted(left)
    if seeds != sorted(right):
        raise ValueError("Paired comparison requires exactly matching game seeds")
    difference = np.array([right[s]["score"] - left[s]["score"] for s in seeds])
    rng = np.random.default_rng(seed)
    bootstrap = difference[rng.integers(0, len(seeds), (20_000, len(seeds)))].mean(axis=1)
    return {"games": len(seeds), "mean_score_difference": float(difference.mean()),
            "median_paired_difference": float(np.median(difference)),
            "wins": int((difference > 0).sum()), "losses": int((difference < 0).sum()),
            "ties": int((difference == 0).sum()),
            "paired_bootstrap_95_percentile_interval": np.quantile(bootstrap, [.025, .975]).tolist(),
            "bootstrap_draws": 20_000, "bootstrap_seed": seed}


def evaluate_model(model_path, output, games, seed, terminal_rank, device):
    model = TrackedRecurrentPPO.load(model_path, device=device)
    try:
        evaluate(model.policy, games, seed, batch_size=min(64, games),
                 terminal_rank=terminal_rank, output=output)
    finally:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return score_rows(output)


def compare_group(group_dir, *, device=None):
    group_dir = Path(group_dir)
    plan = read_json(group_dir / "plan.json")
    parent_run = Path(plan["parent_run"])
    parent_config = read_json(parent_run / "metadata/config.json")
    device = device or parent_config["training"]["device"]
    games = plan["development_comparison"]["games"]
    seed = plan["development_comparison"]["seed_start"]
    terminal_rank = parent_config["environment"]["terminal_rank"]
    agents = {"parent_endpoint": Path(plan["parent_checkpoint"]) / "model.zip"}
    from .runner import ARMS
    for arm in ARMS:
        run = RUNS_ROOT / f"{plan['group']}_{arm}"
        if read_json(run / "metadata/status.json")["status"] != "complete":
            raise ValueError(f"Fork arm is not complete: {run}")
        endpoint = run / "models/checkpoints" / f"step_{plan['endpoint_steps']:012d}"
        if not (endpoint / "complete.json").exists():
            raise ValueError(f"Missing fixed-budget endpoint: {endpoint}")
        record = read_json(run / "metadata/fork.json")
        if (record["parent_model_sha256"] != plan["parent_model_sha256"] or
                record["parent_runtime_sha256"] != plan["parent_runtime_sha256"]):
            raise ValueError(f"Parent provenance differs for {arm}")
        agents[arm + "_endpoint"] = endpoint / "model.zip"
        agents[arm + "_selected_best"] = run / "models/best.zip"
    data_dir = group_dir / "evaluation"
    data_dir.mkdir(exist_ok=True)
    rows = {}
    summaries = {}
    for name, model_path in agents.items():
        print(f"Evaluating {name} on {games} natural games", flush=True)
        rows[name] = evaluate_model(model_path, data_dir / f"{name}.csv",
                                    games, seed, terminal_rank, device)
        summaries[name] = {"model": str(model_path), **summarize(rows[name])}
    comparisons = {}
    baseline = rows["parent_endpoint"]
    for arm in ARMS:
        comparisons[arm + "_endpoint_vs_parent"] = paired_result(
            baseline, rows[arm + "_endpoint"])
    for rate in ("00015", "00005"):
        comparisons[f"mask_effect_lr_{rate}"] = paired_result(
            rows[f"unmasked_lr_{rate}_endpoint"], rows[f"masked_lr_{rate}_endpoint"])
    for masked in ("unmasked", "masked"):
        comparisons[f"lr_effect_{masked}"] = paired_result(
            rows[f"{masked}_lr_00015_endpoint"], rows[f"{masked}_lr_00005_endpoint"])
    output = {"measured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
              "group": plan["group"], "parent_run": plan["parent_run"],
              "parent_steps": plan["parent_steps"], "endpoint_steps": plan["endpoint_steps"],
              "engine_revision": 2, "games": games, "seed_start": seed,
              "seed_end": seed + games - 1,
              "protocol": "Fixed-budget endpoints on paired natural seeds; selected-best reported separately",
              "agents": summaries, "paired_comparisons": comparisons}
    (group_dir / "comparison.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    lines = ["# Masking × learning-rate pilot comparison", "",
             f"Measured: {output['measured_at']}. Parent: `{parent_run.name}` at {plan['parent_steps']:,} transitions. ",
             f"Each fork adds {plan['rounded_additional_steps']:,} attempted transitions to "
             f"{plan['endpoint_steps']:,}. Natural-game development seeds: {seed}–{seed + games - 1}. ",
             "These are paired development games and one parent training trajectory, not a final test or independent training replication.", "",
             "| Agent | Mean score | Median | Reach 384 | Reach 3072 | Mean valid moves |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, summary in summaries.items():
        lines.append(f"| {name} | {summary['mean_score']:.1f} | {summary['median_score']:.1f} | "
                     f"{summary['reach_384']:.1%} | {summary['reach_3072']:.1%} | "
                     f"{summary['mean_valid_moves']:.1f} |")
    lines += ["", "| Paired comparison (right minus left) | Mean score difference | Bootstrap 95% interval | Wins / losses |",
              "| --- | ---: | ---: | ---: |"]
    for name, result in comparisons.items():
        low, high = result["paired_bootstrap_95_percentile_interval"]
        lines.append(f"| {name} | {result['mean_score_difference']:.1f} | "
                     f"[{low:.1f}, {high:.1f}] | {result['wins']} / {result['losses']} |")
    lines += ["", "Per-game CSV files and machine-readable results are in "
              "[evaluation/](evaluation/) and [comparison.json](comparison.json).", ""]
    (group_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Comparison: {group_dir / 'comparison.md'}", flush=True)
    return output
