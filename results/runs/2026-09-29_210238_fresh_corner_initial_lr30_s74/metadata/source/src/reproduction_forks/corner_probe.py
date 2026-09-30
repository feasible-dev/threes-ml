"""Matched short fork with a temporary edge/corner largest-tile reward."""

import argparse
import json
import math
import re
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from . import runner
from .corner_reward import (CORNER_WEIGHT, DISCOUNT, MIN_RANK, SIDE_FRACTION,
                            CornerRewardVecEnv)
from .plateau_probe import evaluate_set


CONTROL = "masked_lr_00005"
TREATMENT = "masked_lr_00005_edge_corner"
ARMS = (CONTROL, TREATMENT)
GAMES = 300
SEED_START = 870_000
DEFAULT_PARENT = (runner.RUNS_ROOT /
                  "2026-09-28_192644_mask_lr_extension_masked_lr_00005")
EXPERIMENT = "largest_tile_edge_corner_potential_v1"
EXTRA_SOURCE = ("src/reproduction_forks/corner_reward.py",
                "src/reproduction_forks/corner_probe.py")


def register_arm():
    runner.ARMS[TREATMENT] = runner.ARMS[CONTROL]


def reward_record(weight):
    return {"type": EXPERIMENT, "corner_weight": weight,
            "side_weight": weight * SIDE_FRACTION, "discount": DISCOUNT,
            "min_rank": MIN_RANK, "min_tile": 12}


def pilot_plan(parent_run, additional_steps):
    if additional_steps < 1:
        raise ValueError("Additional transitions must be positive")
    register_arm()
    parent = runner.inspect_fork(parent_run)
    if parent["status"] not in ("complete", "interrupted"):
        raise ValueError("Stop the parent run before starting an isolated pilot")
    if parent["status"] == "complete" and parent["remaining_steps"] != 0:
        raise ValueError("The completed parent has no allocated endpoint checkpoint")
    if parent["compatibility_issues"]:
        raise ValueError("Parent/source compatibility issues: " +
                         "; ".join(parent["compatibility_issues"]))
    if (parent["arm"] != CONTROL or not parent["mask_training"] or
            parent["effective_lr"] != runner.ARMS[CONTROL][1]):
        raise ValueError("Select a masked LR 0.00005 fork checkpoint")
    config = runner.read_json(Path(parent["run"]) / "metadata/config.json")
    rollout = config["ppo"]["n_envs"] * config["ppo"]["n_steps"]
    rounded = math.ceil(additional_steps / rollout) * rollout
    specification = {"parent_run": parent["run"], "parent_checkpoint": parent["checkpoint"],
                     "parent_steps": parent["saved_steps"], "parent_status": parent["status"],
                     "parent_checkpoint_status": "complete",
                     "parent_allocation_target_steps": parent["target_steps"],
                     "parent_remaining_steps": parent["remaining_steps"],
                     "compatibility_issues": [],
                     "requested_additional_steps": additional_steps,
                     "rounded_additional_steps": rounded,
                     "endpoint_steps": parent["saved_steps"] + rounded,
                     "rollout_size": rollout, "training_seed": config["training"]["seed"]}
    return {**specification, "experiment": EXPERIMENT,
            "corner_reward": reward_record(CORNER_WEIGHT),
            "arms": {arm: {"mask_training": True,
                           "effective_lr": runner.ARMS[arm][1],
                           "corner_reward": reward_record(CORNER_WEIGHT if arm == TREATMENT else 0.0)}
                     for arm in ARMS},
            "development_comparison": {"games": GAMES, "seed_start": SEED_START}}


@contextmanager
def configured_runner():
    """Select the fork-local environment without editing hash-checked source."""
    original_env = runner.ReproductionVecEnv
    original_config = runner.arm_config
    original_run_arm = runner.run_arm
    original_require_ready = runner.require_ready

    def arm_config(parent_config, specification, arm):
        config = original_config(parent_config, specification, arm)
        config["fork"]["corner_reward"] = reward_record(
            CORNER_WEIGHT if arm == TREATMENT else 0.0)
        return config

    def run_arm(specification, arm, group):
        runner.ReproductionVecEnv = CornerRewardVecEnv if arm == TREATMENT else original_env
        try:
            return original_run_arm(specification, arm, group)
        finally:
            runner.ReproductionVecEnv = original_env

    def require_ready(specification):
        if specification["parent_status"] != "interrupted":
            return original_require_ready(specification)
        current = runner.inspect_fork(specification["parent_run"])
        if (current["status"] != "interrupted" or
                current["checkpoint"] != specification["parent_checkpoint"] or
                current["saved_steps"] != specification["parent_steps"] or
                current["compatibility_issues"]):
            raise ValueError("The interrupted parent or its complete checkpoint changed")
        workers = runner.active_workers()
        if workers:
            raise RuntimeError("A recurrent trainer or evaluator is already running")

    runner.arm_config = arm_config
    runner.run_arm = run_arm
    runner.require_ready = require_ready
    try:
        yield original_env
    finally:
        runner.ReproductionVecEnv = original_env
        runner.arm_config = original_config
        runner.run_arm = original_run_arm
        runner.require_ready = original_require_ready


def verify_group(group):
    directory = (runner.GROUP_ROOT / group).resolve()
    if not directory.is_relative_to(runner.GROUP_ROOT.resolve()):
        raise ValueError("Group must be inside results/fork_batches")
    plan = runner.read_json(directory / "plan.json")
    if (plan.get("experiment") != EXPERIMENT or tuple(plan["arms"]) != ARMS or
            plan.get("corner_reward") != reward_record(CORNER_WEIGHT) or
            plan.get("development_comparison") != {"games": GAMES, "seed_start": SEED_START}):
        raise ValueError("This group does not match the corner-reward pilot")
    checkpoint = Path(plan["parent_checkpoint"])
    if (runner.sha256(checkpoint / "model.zip") != plan["parent_model_sha256"] or
            runner.sha256(checkpoint / "runtime.pt") != plan["parent_runtime_sha256"]):
        raise ValueError("The parent checkpoint changed")
    return directory, plan


def verify_run(run):
    run = Path(run).resolve()
    config = runner.read_json(run / "metadata/config.json")
    arm = config["fork"]["arm"]
    if arm not in ARMS or config["fork"].get("corner_reward") != reward_record(
            CORNER_WEIGHT if arm == TREATMENT else 0.0):
        raise ValueError("This is not a matching corner-reward pilot arm")
    provenance = runner.read_json(run / "metadata/provenance.json")
    source = provenance.get("resumes", [])[-1] if provenance.get("resumes") else provenance
    for relative in EXTRA_SOURCE:
        if source["sha256"].get(relative) != runner.sha256(runner.PROJECT_ROOT / relative):
            raise ValueError(f"Corner-reward source changed since this run: {relative}")
    return arm


def compare_group(group):
    directory, plan = verify_group(group)
    paths = {"parent_endpoint": Path(plan["parent_checkpoint"]) / "model.zip"}
    for arm in ARMS:
        run = runner.RUNS_ROOT / f"{group}_{arm}"
        verify_run(run)
        record = runner.read_json(run / "metadata/fork.json")
        endpoint = run / "models/checkpoints" / f"step_{plan['endpoint_steps']:012d}"
        if (runner.read_json(run / "metadata/status.json")["status"] != "complete" or
                not (endpoint / "complete.json").is_file() or
                record["parent_model_sha256"] != plan["parent_model_sha256"] or
                record["parent_runtime_sha256"] != plan["parent_runtime_sha256"]):
            raise ValueError(f"Incomplete or mismatched arm: {run}")
        paths[arm] = endpoint / "model.zip"
    if runner.active_workers():
        raise RuntimeError("A recurrent trainer or evaluator is already running")
    config = runner.read_json(Path(plan["parent_run"]) / "metadata/config.json")
    with runner.continuation_lock():
        return evaluate_set(
            directory, paths, seed=SEED_START, device=config["training"]["device"],
            terminal_rank=config["environment"]["terminal_rank"],
            contrasts=(("control_minus_parent", "parent_endpoint", CONTROL),
                       ("corner_minus_parent", "parent_endpoint", TREATMENT),
                       ("corner_minus_control", CONTROL, TREATMENT)),
            title="Largest-tile edge/corner reward pilot",
            description=f"Both arms restored one complete parent checkpoint and trained for "
                        f"{plan['rounded_additional_steps']:,} transitions. The treatment alone "
                        f"added potential {CORNER_WEIGHT * SIDE_FRACTION:g} on a side and "
                        f"{CORNER_WEIGHT:g} in a corner for a largest tile of at least 12; "
                        "leaving either location reverses its bonus.",
            metadata={"group": group, "experiment": EXPERIMENT,
                      "corner_reward": reward_record(CORNER_WEIGHT),
                      "parent_steps": plan["parent_steps"],
                      "endpoint_steps": plan["endpoint_steps"],
                      "parent_model_sha256": plan["parent_model_sha256"],
                      "parent_runtime_sha256": plan["parent_runtime_sha256"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-fork", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--additional-steps", type=int, default=5_000_000)
    parser.add_argument("--group")
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--compare-group")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    register_arm()
    if args.resume_run:
        if args.group or args.compare_group:
            parser.error("--resume-run cannot be combined with a group option")
        arm = verify_run(args.resume_run)
        with configured_runner() as base_env:
            runner.ReproductionVecEnv = CornerRewardVecEnv if arm == TREATMENT else base_env
            runner.resume_fork(args.resume_run, preview=args.preview)
        return
    if args.compare_group:
        if args.group:
            parser.error("--compare-group cannot be combined with --group")
        if args.preview:
            _, plan = verify_group(args.compare_group)
            print(json.dumps(plan, indent=2))
        else:
            compare_group(args.compare_group)
        return
    specification = pilot_plan(args.parent_fork, args.additional_steps)
    group = args.group or datetime.now().astimezone().strftime("%Y-%m-%d_%H%M%S_corner_probe")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", group):
        parser.error("--group may contain only letters, digits, underscores and hyphens")
    plan_file = runner.GROUP_ROOT / group / "plan.json"
    if plan_file.exists():
        _, old = verify_group(group)
        for key in ("parent_run", "parent_steps", "endpoint_steps", "arms"):
            if old[key] != specification[key]:
                raise ValueError(f"Existing group has a different {key}")
    if args.preview:
        print(json.dumps({"group": group, "plan": specification}, indent=2))
        return
    with configured_runner():
        runner.run_batch(specification, ARMS, group)
    compare_group(group)


if __name__ == "__main__":
    main()
