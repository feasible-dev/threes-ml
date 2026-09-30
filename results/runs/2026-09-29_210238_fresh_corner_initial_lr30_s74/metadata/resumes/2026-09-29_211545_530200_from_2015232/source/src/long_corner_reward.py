"""Continue the preserved edge/corner reward checkpoint at lower fixed LR."""

import argparse
import json
import math
import re
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from reproduction_forks import runner
from reproduction_forks.corner_probe import (CORNER_WEIGHT, EXPERIMENT, TREATMENT,
                                             reward_record, verify_run)
from reproduction_forks.corner_reward import CornerRewardVecEnv


LONG_ARM = "masked_lr_000025_edge_corner"
LONG_RATE = 0.000025
DEFAULT_PARENT = (runner.RUNS_ROOT /
                  "2026-09-28_223320_edge_corner_398m_masked_lr_00005_edge_corner")
EXTRA_SOURCE = ("src/reproduction_forks/corner_reward.py",
                "src/long_corner_reward.py")


def register_arms():
    runner.ARMS[TREATMENT] = (True, 0.00005)
    runner.ARMS[LONG_ARM] = (True, LONG_RATE)


def long_plan(parent_fork, additional_steps):
    if additional_steps < 1:
        raise ValueError("Additional transitions must be positive")
    register_arms()
    if verify_run(parent_fork) != TREATMENT:
        raise ValueError("Select the edge/corner treatment arm")
    parent = runner.inspect_fork(parent_fork)
    if parent["status"] not in ("complete", "interrupted"):
        raise ValueError("Stop the parent run before starting a long child")
    if parent["status"] == "complete" and parent["remaining_steps"] != 0:
        raise ValueError("The completed parent has no allocated endpoint checkpoint")
    if parent["compatibility_issues"]:
        raise ValueError("Parent/source compatibility issues: " +
                         "; ".join(parent["compatibility_issues"]))
    if (parent["arm"] != TREATMENT or not parent["mask_training"] or
            parent["effective_lr"] != runner.ARMS[TREATMENT][1]):
        raise ValueError("The parent does not match the edge/corner treatment")
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
    return {**specification, "continuation_of": EXPERIMENT,
            "reward": reward_record(CORNER_WEIGHT),
            "evaluation_protocol": "periodic natural validation only; no paired comparison",
            "arms": {LONG_ARM: {"mask_training": True, "effective_lr": LONG_RATE,
                                "corner_reward": reward_record(CORNER_WEIGHT)}}}


@contextmanager
def configured_runner():
    """Keep the reward wrapper in the child without changing frozen fork source."""
    original_env = runner.ReproductionVecEnv
    original_config = runner.arm_config
    original_require_ready = runner.require_ready

    def arm_config(parent_config, specification, arm):
        config = original_config(parent_config, specification, arm)
        config["fork"]["corner_reward"] = reward_record(CORNER_WEIGHT)
        return config

    def require_ready(specification):
        if specification["parent_status"] != "interrupted":
            return original_require_ready(specification)
        current = runner.inspect_fork(specification["parent_run"])
        if (current["status"] != "interrupted" or
                current["checkpoint"] != specification["parent_checkpoint"] or
                current["saved_steps"] != specification["parent_steps"] or
                current["compatibility_issues"]):
            raise ValueError("The interrupted parent or its complete checkpoint changed")
        if runner.active_workers():
            raise RuntimeError("A recurrent trainer or evaluator is already running")

    runner.ReproductionVecEnv = CornerRewardVecEnv
    runner.arm_config = arm_config
    runner.require_ready = require_ready
    try:
        yield
    finally:
        runner.ReproductionVecEnv = original_env
        runner.arm_config = original_config
        runner.require_ready = original_require_ready


def verify_long_run(run):
    run = Path(run).resolve()
    config = runner.read_json(run / "metadata/config.json")
    if (config["fork"]["arm"] != LONG_ARM or
            config["fork"].get("corner_reward") != reward_record(CORNER_WEIGHT) or
            config["fork"]["effective_lr"] != LONG_RATE):
        raise ValueError("This is not a matching low-LR edge/corner continuation")
    provenance = runner.read_json(run / "metadata/provenance.json")
    source = provenance.get("resumes", [])[-1] if provenance.get("resumes") else provenance
    for relative in EXTRA_SOURCE:
        if source["sha256"].get(relative) != runner.sha256(runner.PROJECT_ROOT / relative):
            raise ValueError(f"Long edge/corner source changed since this run: {relative}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-fork", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--additional-steps", type=int, default=500_000_000)
    parser.add_argument("--group")
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    register_arms()
    if args.resume_run:
        if args.group:
            parser.error("--resume-run cannot be combined with --group")
        verify_long_run(args.resume_run)
        with configured_runner():
            runner.resume_fork(args.resume_run, preview=args.preview)
        return
    specification = long_plan(args.parent_fork, args.additional_steps)
    group = args.group or datetime.now().astimezone().strftime(
        "%Y-%m-%d_%H%M%S_edge_corner_lr25_500m")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", group):
        parser.error("--group may contain only letters, digits, underscores and hyphens")
    if args.preview:
        print(json.dumps({"group": group, "plan": specification}, indent=2))
        return
    with configured_runner():
        runner.run_batch(specification, [LONG_ARM], group)


if __name__ == "__main__":
    main()
