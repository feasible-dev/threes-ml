"""Branch the stopped edge/corner run for a short half-rate continuation."""

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path

from long_corner_reward import (LONG_ARM, LONG_RATE, configured_runner,
                                register_arms, verify_long_run)
from reproduction_forks import runner
from reproduction_forks.corner_probe import CORNER_WEIGHT, reward_record


PROBE_ARM = "masked_lr_0000125_edge_corner"
PROBE_RATE = 0.0000125
DEFAULT_PARENT = (runner.RUNS_ROOT /
                  "2026-09-28_230211_edge_corner_lr25_500m_masked_lr_000025_edge_corner")
EXTRA_SOURCE = ("src/reproduction_forks/corner_reward.py",
                "src/long_corner_reward.py", "src/corner_lr_probe.py")


def register_probe_arm():
    register_arms()
    runner.ARMS[PROBE_ARM] = (True, PROBE_RATE)


def probe_plan(parent_fork, additional_steps):
    if additional_steps < 1:
        raise ValueError("Additional transitions must be positive")
    register_probe_arm()
    verify_long_run(parent_fork)
    parent = runner.inspect_fork(parent_fork)
    if parent["status"] not in ("complete", "interrupted"):
        raise ValueError("Stop the parent run before starting the probe")
    if parent["status"] == "complete" and parent["remaining_steps"] != 0:
        raise ValueError("The completed parent has no allocated endpoint checkpoint")
    if parent["compatibility_issues"]:
        raise ValueError("Parent/source compatibility issues: " +
                         "; ".join(parent["compatibility_issues"]))
    if (parent["arm"] != LONG_ARM or not parent["mask_training"] or
            parent["effective_lr"] != LONG_RATE):
        raise ValueError("Select the masked edge/corner run at LR 0.000025")
    config = runner.read_json(Path(parent["run"]) / "metadata/config.json")
    rollout = config["ppo"]["n_envs"] * config["ppo"]["n_steps"]
    rounded = math.ceil(additional_steps / rollout) * rollout
    return {"parent_run": parent["run"], "parent_checkpoint": parent["checkpoint"],
            "parent_steps": parent["saved_steps"], "parent_status": parent["status"],
            "parent_checkpoint_status": "complete",
            "parent_allocation_target_steps": parent["target_steps"],
            "parent_remaining_steps": parent["remaining_steps"],
            "compatibility_issues": [],
            "requested_additional_steps": additional_steps,
            "rounded_additional_steps": rounded,
            "endpoint_steps": parent["saved_steps"] + rounded,
            "rollout_size": rollout, "training_seed": config["training"]["seed"],
            "continuation_of": str(parent_fork),
            "reward": reward_record(CORNER_WEIGHT),
            "evaluation_protocol": "periodic natural validation only; no paired comparison",
            "arms": {PROBE_ARM: {"mask_training": True, "effective_lr": PROBE_RATE,
                                "corner_reward": reward_record(CORNER_WEIGHT)}}}


def verify_probe_run(run):
    run = Path(run).resolve()
    config = runner.read_json(run / "metadata/config.json")
    if (config["fork"]["arm"] != PROBE_ARM or
            config["fork"].get("corner_reward") != reward_record(CORNER_WEIGHT) or
            config["fork"]["effective_lr"] != PROBE_RATE):
        raise ValueError("This is not a matching half-rate edge/corner probe")
    provenance = runner.read_json(run / "metadata/provenance.json")
    source = provenance.get("resumes", [])[-1] if provenance.get("resumes") else provenance
    for relative in EXTRA_SOURCE:
        if source["sha256"].get(relative) != runner.sha256(runner.PROJECT_ROOT / relative):
            raise ValueError(f"Probe source changed since this run: {relative}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-fork", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--additional-steps", type=int, default=1_000_000)
    parser.add_argument("--group")
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    register_probe_arm()
    if args.resume_run:
        if args.group:
            parser.error("--resume-run cannot be combined with --group")
        verify_probe_run(args.resume_run)
        with configured_runner():
            runner.resume_fork(args.resume_run, preview=args.preview)
        return
    specification = probe_plan(args.parent_fork, args.additional_steps)
    group = args.group or datetime.now().astimezone().strftime(
        "%Y-%m-%d_%H%M%S_corner_lr125_1m")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", group):
        parser.error("--group may contain only letters, digits, underscores and hyphens")
    if args.preview:
        print(json.dumps({"group": group, "plan": specification}, indent=2))
        return
    with configured_runner():
        runner.run_batch(specification, [PROBE_ARM], group)


if __name__ == "__main__":
    main()
