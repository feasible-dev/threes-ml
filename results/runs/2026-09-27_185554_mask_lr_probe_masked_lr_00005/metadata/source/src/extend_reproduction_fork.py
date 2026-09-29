"""Branch a completed masking/LR fork into a new, longer training allocation.

The completed pilot remains immutable. The child restores its endpoint weights,
optimizer, simulator, recurrent memory and RNG, then collects fresh rollouts.
"""

import argparse
import math
import re
from datetime import datetime
from pathlib import Path

from reproduction_forks.runner import ARMS, inspect_fork, read_json, run_batch


def extension_plan(parent_fork, additional_steps):
    if additional_steps < 1:
        raise ValueError("Additional transitions must be positive")
    parent = inspect_fork(parent_fork)
    if parent["status"] != "complete" or parent["remaining_steps"] != 0:
        raise ValueError("The parent fork must have completed its allocated endpoint")
    if parent["compatibility_issues"]:
        raise ValueError("Parent/source compatibility issues: " +
                         "; ".join(parent["compatibility_issues"]))
    run = Path(parent["run"])
    config = read_json(run / "metadata/config.json")
    rollout = config["ppo"]["n_envs"] * config["ppo"]["n_steps"]
    rounded = math.ceil(additional_steps / rollout) * rollout
    arm = parent["arm"]
    mask, rate = ARMS[arm]
    if parent["mask_training"] != mask or parent["effective_lr"] != rate:
        raise ValueError("Parent fork settings disagree with its recorded arm")
    return {"parent_run": str(run), "parent_checkpoint": parent["checkpoint"],
            "parent_steps": parent["saved_steps"], "parent_status": "complete",
            "compatibility_issues": [], "parent_type": "completed_mask_lr_fork",
            "requested_additional_steps": additional_steps,
            "rounded_additional_steps": rounded,
            "endpoint_steps": parent["saved_steps"] + rounded,
            "rollout_size": rollout, "training_seed": config["training"]["seed"],
            "arms": {arm: {"mask_training": mask, "effective_lr": rate}},
            "development_comparison": {"games": 300, "seed_start": 830000}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-fork", type=Path, required=True)
    parser.add_argument("--additional-steps", type=int, default=20_000_000)
    parser.add_argument("--group")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    specification = extension_plan(args.parent_fork, args.additional_steps)
    if args.preview:
        import json
        print(json.dumps(specification, indent=2))
        return
    group = args.group or datetime.now().astimezone().strftime("%Y-%m-%d_%H%M%S_mask_lr_extension")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", group):
        parser.error("--group may contain only letters, digits, underscores and hyphens")
    run_batch(specification, list(specification["arms"]), group)


if __name__ == "__main__":
    main()
