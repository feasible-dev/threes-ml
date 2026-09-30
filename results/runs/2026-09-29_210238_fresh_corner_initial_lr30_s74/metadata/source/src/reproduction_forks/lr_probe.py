"""Compare two short masked LR continuations from one completed fork endpoint."""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from . import runner
from .compare import evaluate_model, paired_result, summarize
from extend_reproduction_fork import extension_plan


CONTROL = "masked_lr_00005"
TREATMENT = "masked_lr_00010"
ARMS = (CONTROL, TREATMENT)
SEED_START = 840000
GAMES = 300


def register_probe_arm():
    # The frozen fork runner's source is hash-checked by active continuations.
    # Register this isolated treatment in memory without editing that source.
    runner.ARMS[TREATMENT] = (True, 0.00010)


def probe_plan(parent_fork, additional_steps):
    register_probe_arm()
    specification = extension_plan(parent_fork, additional_steps)
    if list(specification["arms"]) != [CONTROL]:
        raise ValueError("The LR probe requires a completed masked LR 0.00005 parent")
    return {**specification,
            "probe": "masked_lr_00005_vs_00010",
            "arms": {arm: {"mask_training": mask, "effective_lr": rate}
                     for arm in ARMS for mask, rate in [runner.ARMS[arm]]},
            "development_comparison": {"games": GAMES, "seed_start": SEED_START}}


def compare_probe(group):
    register_probe_arm()
    group_dir = (runner.GROUP_ROOT / group).resolve()
    if not group_dir.is_relative_to(runner.GROUP_ROOT.resolve()):
        raise ValueError("Comparison group must be inside results/fork_batches")
    plan = runner.read_json(group_dir / "plan.json")
    if plan.get("probe") != "masked_lr_00005_vs_00010" or tuple(plan["arms"]) != ARMS:
        raise ValueError("This is not a two-arm masked LR probe")
    checkpoint = Path(plan["parent_checkpoint"])
    if (runner.sha256(checkpoint / "model.zip") != plan["parent_model_sha256"] or
            runner.sha256(checkpoint / "runtime.pt") != plan["parent_runtime_sha256"]):
        raise ValueError("The parent checkpoint changed since the probe was created")
    model_paths = {"parent_endpoint": checkpoint / "model.zip"}
    for arm in ARMS:
        run = runner.RUNS_ROOT / f"{group}_{arm}"
        record = runner.read_json(run / "metadata/fork.json")
        if (runner.read_json(run / "metadata/status.json")["status"] != "complete" or
                record["arm"] != arm or record["mask_training"] is not True or
                record["parent_model_sha256"] != plan["parent_model_sha256"] or
                record["parent_runtime_sha256"] != plan["parent_runtime_sha256"] or
                record["effective_lr"] != runner.ARMS[arm][1]):
            raise ValueError(f"Incomplete or mismatched LR probe arm: {run}")
        endpoint = run / "models/checkpoints" / f"step_{plan['endpoint_steps']:012d}"
        if not (endpoint / "complete.json").is_file():
            raise ValueError(f"Missing fixed-budget endpoint: {endpoint}")
        model_paths[arm] = endpoint / "model.zip"
    data_dir = group_dir / "evaluation"
    data_dir.mkdir(exist_ok=True)
    parent_config = runner.read_json(Path(plan["parent_run"]) / "metadata/config.json")
    games = plan["development_comparison"]["games"]
    seed = plan["development_comparison"]["seed_start"]
    device = parent_config["training"]["device"]
    terminal_rank = parent_config["environment"]["terminal_rank"]
    rows = {}
    summaries = {}
    for name, model_path in model_paths.items():
        print(f"Evaluating {name} on {games} natural games", flush=True)
        rows[name] = evaluate_model(model_path, data_dir / f"{name}.csv",
                                    games, seed, terminal_rank, device)
        summaries[name] = {"model": str(model_path), **summarize(rows[name])}
    comparisons = {
        "control_vs_parent": paired_result(rows["parent_endpoint"], rows[CONTROL], seed=840999),
        "treatment_vs_parent": paired_result(rows["parent_endpoint"], rows[TREATMENT], seed=841000),
        "lr_00010_minus_00005": paired_result(rows[CONTROL], rows[TREATMENT], seed=841001)}
    result = {"measured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
              "group": group, "parent_run": plan["parent_run"],
              "parent_steps": plan["parent_steps"], "endpoint_steps": plan["endpoint_steps"],
              "games": games, "seed_start": seed, "seed_end": seed + games - 1,
              "protocol": "Same masked parent and attempted-transition budget; fixed endpoints; paired natural development games",
              "agents": summaries, "paired_comparisons": comparisons}
    runner.write_json(group_dir / "comparison.json", result)
    lines = ["# Masked learning-rate probe", "",
             f"Measured: {result['measured_at']}. Parent: `{Path(plan['parent_run']).name}` at "
             f"{plan['parent_steps']:,} transitions. Each arm adds "
             f"{plan['rounded_additional_steps']:,} attempted transitions to "
             f"{plan['endpoint_steps']:,}. Natural development seeds: {seed}–{seed + games - 1}.",
             "Both arms restore the same parent weights, optimizer, simulator and RNG. "
             "This is one parent training trajectory, not an independent-seed replication or final test.", "",
             "| Fixed-budget endpoint | Mean score | Median | Reach 384 | Mean valid moves |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name, summary in summaries.items():
        lines.append(f"| {name} | {summary['mean_score']:.1f} | "
                     f"{summary['median_score']:.1f} | {summary['reach_384']:.1%} | "
                     f"{summary['mean_valid_moves']:.1f} |")
    lines += ["", "| Paired comparison (right minus left) | Mean difference | Bootstrap 95% interval | Wins / losses |",
              "| --- | ---: | ---: | ---: |"]
    for name, contrast in comparisons.items():
        low, high = contrast["paired_bootstrap_95_percentile_interval"]
        lines.append(f"| {name} | {contrast['mean_score_difference']:.1f} | "
                     f"[{low:.1f}, {high:.1f}] | {contrast['wins']} / {contrast['losses']} |")
    lines += ["", "Per-game CSVs are in [evaluation/](evaluation/) and machine-readable results "
              "in [comparison.json](comparison.json).", ""]
    (group_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Comparison: {group_dir / 'comparison.md'}", flush=True)
    return result


def main():
    register_probe_arm()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-fork", type=Path)
    parser.add_argument("--additional-steps", type=int, default=5_000_000)
    parser.add_argument("--group")
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--compare-group")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    if args.resume_run:
        if args.parent_fork or args.compare_group or args.group:
            parser.error("--resume-run cannot be combined with a new probe or comparison")
        runner.resume_fork(args.resume_run, preview=args.preview)
        return
    if args.compare_group:
        if args.parent_fork or args.group:
            parser.error("--compare-group cannot be combined with a new probe")
        if args.preview:
            print(json.dumps(runner.read_json(runner.GROUP_ROOT / args.compare_group / "plan.json"), indent=2))
            return
        if runner.active_workers():
            raise RuntimeError("A recurrent trainer or evaluator is already running")
        with runner.continuation_lock():
            compare_probe(args.compare_group)
        return
    if args.parent_fork is None:
        parser.error("--parent-fork is required for a new probe")
    specification = probe_plan(args.parent_fork, args.additional_steps)
    if args.preview:
        print(json.dumps(specification, indent=2))
        return
    group = args.group or datetime.now().astimezone().strftime("%Y-%m-%d_%H%M%S_mask_lr_probe")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", group):
        parser.error("--group may contain only letters, digits, underscores and hyphens")
    runner.run_batch(specification, ARMS, group)
    with runner.continuation_lock():
        compare_probe(group)


if __name__ == "__main__":
    main()
