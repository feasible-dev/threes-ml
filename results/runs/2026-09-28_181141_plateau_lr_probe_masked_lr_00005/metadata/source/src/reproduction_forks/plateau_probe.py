"""Post-run assessment and matched LR-halving probe from the 330M masked fork."""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from extend_reproduction_fork import extension_plan
from . import runner
from .compare import evaluate_model, paired_result, summarize
from .lr_probe import register_probe_arm


PARENT_STEPS = 330_039_296
GAMES = 300
BASELINE_SEED = 850_000
PILOT_SEED = 860_000
CONTROL = "masked_lr_00010"
TREATMENT = "masked_lr_00005"
ARMS = (CONTROL, TREATMENT)
DEFAULT_PARENT = (runner.RUNS_ROOT /
                  "2026-09-27_193437_masked_lr_00010_long_masked_lr_00010")


def checked_group(name):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Group may contain only letters, digits, underscores and hyphens")
    return name


def pilot_plan(parent_run, additional_steps):
    register_probe_arm()
    specification = extension_plan(parent_run, additional_steps)
    if specification["parent_steps"] != PARENT_STEPS or list(specification["arms"]) != [CONTROL]:
        raise ValueError("Select the completed 330,039,296-transition masked LR 0.00010 endpoint")
    return {**specification, "probe": "plateau_lr_00010_vs_00005",
            "arms": {arm: {"mask_training": True, "effective_lr": runner.ARMS[arm][1]}
                     for arm in ARMS},
            "development_comparison": {"games": GAMES, "seed_start": PILOT_SEED}}


def summary(rows, model):
    result = {"model": str(model), **summarize(rows)}
    tiles = np.array([row["max_tile"] for row in rows.values()])
    for tile in (192, 768, 1536):
        result[f"reach_{tile}"] = float(np.mean(tiles >= tile))
    return result


def write_report(directory, title, description, agents, comparisons, seed):
    lines = [f"# {title}", "", description,
             f"Natural development games: {seed}–{seed + GAMES - 1}. "
             "All policies use legal greedy moves and fresh recurrent state per game.",
             "These are fixed-policy game-level comparisons from one training lineage, "
             "not independent training-seed replications or final-test results.", "",
             "| Policy | Mean score | Median | >=384 | >=768 | >=1536 | Mean valid moves |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, item in agents.items():
        lines.append(f"| {name} | {item['mean_score']:.1f} | {item['median_score']:.1f} | "
                     f"{item['reach_384']:.1%} | {item['reach_768']:.1%} | "
                     f"{item['reach_1536']:.1%} | {item['mean_valid_moves']:.1f} |")
    lines += ["", "| Paired score contrast (right minus left) | Mean difference | "
              "95% bootstrap interval | Wins / losses |",
              "| --- | ---: | ---: | ---: |"]
    for name, contrast in comparisons.items():
        low, high = contrast["paired_bootstrap_95_percentile_interval"]
        lines.append(f"| {name} | {contrast['mean_score_difference']:.1f} | "
                     f"[{low:.1f}, {high:.1f}] | "
                     f"{contrast['wins']} / {contrast['losses']} |")
    lines += ["", "Per-game scores are in [evaluation/](evaluation/); "
              "machine-readable results are in [comparison.json](comparison.json).", ""]
    (directory / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def evaluate_set(directory, paths, *, seed, device, terminal_rank, contrasts,
                 title, description, metadata):
    directory.mkdir(parents=True, exist_ok=True)
    inputs = {name: {"path": str(path), "sha256": runner.sha256(path)}
              for name, path in paths.items()}
    comparison_path = directory / "comparison.json"
    if comparison_path.exists() and (directory / "comparison.md").exists():
        existing = runner.read_json(comparison_path)
        if (existing.get("inputs") != inputs or existing.get("seed_start") != seed or
                existing.get("games") != GAMES or existing.get("metadata") != metadata):
            raise ValueError(f"Existing comparison has different inputs: {comparison_path}")
        print(f"Using completed comparison: {directory / 'comparison.md'}", flush=True)
        return existing
    data_dir = directory / "evaluation"
    data_dir.mkdir(exist_ok=True)
    rows = {}
    agents = {}
    for name, path in paths.items():
        print(f"Evaluating {name}: {GAMES} natural games", flush=True)
        rows[name] = evaluate_model(path, data_dir / f"{name}.csv",
                                    GAMES, seed, terminal_rank, device)
        agents[name] = summary(rows[name], path)
    comparisons = {name: paired_result(rows[left], rows[right], seed=seed + 999 + i)
                   for i, (name, left, right) in enumerate(contrasts)}
    result = {"measured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
              "inputs": inputs, "metadata": metadata, "games": GAMES,
              "seed_start": seed, "seed_end": seed + GAMES - 1,
              "agents": agents, "paired_comparisons": comparisons}
    write_report(directory, title, description, agents, comparisons, seed)
    runner.write_json(comparison_path, result)
    print(f"Comparison: {directory / 'comparison.md'}", flush=True)
    return result


def baseline_assessment(specification, group):
    if runner.active_workers():
        raise RuntimeError("A recurrent trainer or evaluator is already running")
    parent_run = Path(specification["parent_run"])
    parent_record = runner.read_json(parent_run / "metadata/fork.json")
    earlier = Path(parent_record["parent_checkpoint"]) / "model.zip"
    endpoint = Path(specification["parent_checkpoint"]) / "model.zip"
    selected = parent_run / "models/best.zip"
    paths = {"probe_parent_230m": earlier, "fixed_endpoint_330m": endpoint,
             "selected_best_307m": selected}
    config = runner.read_json(parent_run / "metadata/config.json")
    with runner.continuation_lock():
        return evaluate_set(
            runner.PROJECT_ROOT / "results/comparisons" / f"{group}_baseline",
            paths, seed=BASELINE_SEED, device=config["training"]["device"],
            terminal_rank=config["environment"]["terminal_rank"],
            contrasts=(("330m_minus_230m", "probe_parent_230m", "fixed_endpoint_330m"),
                       ("selected_best_minus_330m", "fixed_endpoint_330m", "selected_best_307m")),
            title="Post-run plateau assessment",
            description="The selected-best checkpoint was chosen on the reused 100-game periodic set; "
                        "it is assessed separately from the fixed endpoint here.",
            metadata={"parent_run": str(parent_run), "parent_steps": PARENT_STEPS,
                      "engine_revision": runner.ENGINE_REVISION})


def compare_pilot(group):
    register_probe_arm()
    group_dir = (runner.GROUP_ROOT / checked_group(group)).resolve()
    if not group_dir.is_relative_to(runner.GROUP_ROOT.resolve()):
        raise ValueError("Comparison group must be inside results/fork_batches")
    plan = runner.read_json(group_dir / "plan.json")
    if (plan.get("probe") != "plateau_lr_00010_vs_00005" or
            tuple(plan["arms"]) != ARMS or plan["parent_steps"] != PARENT_STEPS):
        raise ValueError("This is not the matched plateau LR probe")
    checkpoint = Path(plan["parent_checkpoint"])
    if (runner.sha256(checkpoint / "model.zip") != plan["parent_model_sha256"] or
            runner.sha256(checkpoint / "runtime.pt") != plan["parent_runtime_sha256"]):
        raise ValueError("Parent checkpoint changed since the probe was created")
    paths = {"parent_endpoint": checkpoint / "model.zip"}
    for arm in ARMS:
        run = runner.RUNS_ROOT / f"{group}_{arm}"
        record = runner.read_json(run / "metadata/fork.json")
        endpoint = run / "models/checkpoints" / f"step_{plan['endpoint_steps']:012d}"
        if (runner.read_json(run / "metadata/status.json")["status"] != "complete" or
                not (endpoint / "complete.json").is_file() or
                record["arm"] != arm or record["parent_steps"] != PARENT_STEPS or
                record["parent_model_sha256"] != plan["parent_model_sha256"] or
                record["parent_runtime_sha256"] != plan["parent_runtime_sha256"] or
                record["effective_lr"] != runner.ARMS[arm][1]):
            raise ValueError(f"Incomplete or mismatched LR probe arm: {run}")
        paths[arm] = endpoint / "model.zip"
    config = runner.read_json(Path(plan["parent_run"]) / "metadata/config.json")
    if runner.active_workers():
        raise RuntimeError("A recurrent trainer or evaluator is already running")
    with runner.continuation_lock():
        return evaluate_set(
            group_dir, paths, seed=PILOT_SEED, device=config["training"]["device"],
            terminal_rank=config["environment"]["terminal_rank"],
            contrasts=(("control_minus_parent", "parent_endpoint", CONTROL),
                       ("treatment_minus_parent", "parent_endpoint", TREATMENT),
                       ("lr_00005_minus_00010", CONTROL, TREATMENT)),
            title="Matched LR-halving plateau probe",
            description=f"Both masked arms restored the same {PARENT_STEPS:,}-transition "
                        f"checkpoint and each added {plan['rounded_additional_steps']:,} "
                        "attempted transitions. Only the learning rate differs: "
                        "0.00010 versus 0.00005.",
            metadata={"group": group, "parent_run": plan["parent_run"],
                      "parent_steps": PARENT_STEPS, "endpoint_steps": plan["endpoint_steps"],
                      "parent_model_sha256": plan["parent_model_sha256"],
                      "parent_runtime_sha256": plan["parent_runtime_sha256"],
                      "engine_revision": runner.ENGINE_REVISION})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-fork", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--additional-steps", type=int, default=10_000_000)
    parser.add_argument("--group")
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--compare-group")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    register_probe_arm()
    if args.resume_run:
        if args.group or args.compare_group:
            parser.error("--resume-run cannot be combined with --group or --compare-group")
        runner.resume_fork(args.resume_run, preview=args.preview)
        return
    if args.compare_group:
        if args.group:
            parser.error("--compare-group cannot be combined with --group")
        if args.preview:
            print(json.dumps(runner.read_json(runner.GROUP_ROOT / args.compare_group / "plan.json"),
                             indent=2))
            return
        compare_pilot(args.compare_group)
        return
    specification = pilot_plan(args.parent_fork, args.additional_steps)
    group = checked_group(args.group or datetime.now().astimezone().strftime(
        "%Y-%m-%d_%H%M%S_plateau_lr_probe"))
    if args.preview:
        print(json.dumps({"group": group, "plan": specification,
                          "baseline_games": GAMES, "baseline_seed": BASELINE_SEED}, indent=2))
        return
    print(f"Plateau group: {group}", flush=True)
    baseline_assessment(specification, group)
    plan_file = runner.GROUP_ROOT / group / "plan.json"
    if plan_file.exists():
        existing = runner.read_json(plan_file)
        if (existing.get("probe") != specification["probe"] or
                existing.get("arms") != specification["arms"] or
                existing.get("development_comparison") != specification["development_comparison"]):
            raise ValueError(f"Existing group has a different plateau-probe plan: {plan_file}")
    runner.run_batch(specification, ARMS, group)
    compare_pilot(group)


if __name__ == "__main__":
    main()
