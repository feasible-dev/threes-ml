"""Extend the completed half-rate corner-reward probe in an isolated child run."""

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path

from corner_lr_probe import (PROBE_ARM, PROBE_RATE, register_probe_arm,
                             verify_probe_run)
from long_corner_reward import configured_runner
from reproduction_forks import runner
from reproduction_forks.corner_probe import CORNER_WEIGHT, reward_record


DEFAULT_PARENT = (runner.RUNS_ROOT /
                  "2026-09-29_185906_corner_lr125_1m_masked_lr_0000125_edge_corner")
EXTRA_SOURCE = ("src/reproduction_forks/corner_reward.py",
                "src/long_corner_reward.py", "src/corner_lr_probe.py",
                "src/extend_corner_lr_probe.py")


def extension_plan(parent_fork, additional_steps):
    if additional_steps < 1:
        raise ValueError("Additional transitions must be positive")
    register_probe_arm()
    verify_probe_run(parent_fork)
    parent = runner.inspect_fork(parent_fork)
    if parent["status"] != "complete" or parent["remaining_steps"] != 0:
        raise ValueError("Select the completed half-rate corner-reward probe")
    if parent["compatibility_issues"]:
        raise ValueError("Parent/source compatibility issues: " +
                         "; ".join(parent["compatibility_issues"]))
    if (parent["arm"] != PROBE_ARM or not parent["mask_training"] or
            parent["effective_lr"] != PROBE_RATE):
        raise ValueError("The parent does not match the half-rate reward probe")
    config = runner.read_json(Path(parent["run"]) / "metadata/config.json")
    rollout = config["ppo"]["n_envs"] * config["ppo"]["n_steps"]
    rounded = math.ceil(additional_steps / rollout) * rollout
    return {"parent_run": parent["run"], "parent_checkpoint": parent["checkpoint"],
            "parent_steps": parent["saved_steps"], "parent_status": parent["status"],
            "parent_checkpoint_status": "complete",
            "parent_allocation_target_steps": parent["target_steps"],
            "parent_remaining_steps": 0, "compatibility_issues": [],
            "requested_additional_steps": additional_steps,
            "rounded_additional_steps": rounded,
            "endpoint_steps": parent["saved_steps"] + rounded,
            "rollout_size": rollout, "training_seed": config["training"]["seed"],
            "continuation_of": str(parent_fork),
            "reward": reward_record(CORNER_WEIGHT),
            "evaluation_protocol": "periodic natural validation only; no paired comparison",
            "arms": {PROBE_ARM: {"mask_training": True, "effective_lr": PROBE_RATE,
                                "corner_reward": reward_record(CORNER_WEIGHT)}}}


def verify_extension_run(run):
    run = Path(run).resolve()
    config = runner.read_json(run / "metadata/config.json")
    if (config["fork"]["arm"] != PROBE_ARM or
            config["fork"].get("corner_reward") != reward_record(CORNER_WEIGHT) or
            config["fork"]["effective_lr"] != PROBE_RATE):
        raise ValueError("This is not a matching long half-rate corner continuation")
    provenance = runner.read_json(run / "metadata/provenance.json")
    source = provenance.get("resumes", [])[-1] if provenance.get("resumes") else provenance
    for relative in EXTRA_SOURCE:
        if source["sha256"].get(relative) != runner.sha256(runner.PROJECT_ROOT / relative):
            raise ValueError(f"Long extension source changed since this run: {relative}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-fork", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--additional-steps", type=int, default=100_000_000)
    parser.add_argument("--group")
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    register_probe_arm()
    if args.resume_run:
        if args.group:
            parser.error("--resume-run cannot be combined with --group")
        verify_extension_run(args.resume_run)
        with configured_runner():
            runner.resume_fork(args.resume_run, preview=args.preview)
        return
    specification = extension_plan(args.parent_fork, args.additional_steps)
    group = args.group or datetime.now().astimezone().strftime(
        "%Y-%m-%d_%H%M%S_corner_lr125_100m")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", group):
        parser.error("--group may contain only letters, digits, underscores and hyphens")
    if args.preview:
        print(json.dumps({"group": group, "plan": specification}, indent=2))
        return
    with configured_runner():
        runner.run_batch(specification, [PROBE_ARM], group)


if __name__ == "__main__":
    main()
