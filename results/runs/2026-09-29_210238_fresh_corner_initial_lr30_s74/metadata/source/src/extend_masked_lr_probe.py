"""Continue the completed masked LR 0.00010 probe in a new long-run child."""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from extend_reproduction_fork import extension_plan
from reproduction_forks import runner
from reproduction_forks.lr_probe import TREATMENT, register_probe_arm


def long_plan(parent_fork, additional_steps):
    register_probe_arm()
    specification = extension_plan(parent_fork, additional_steps)
    if list(specification["arms"]) != [TREATMENT]:
        raise ValueError("The long LR 0.00010 continuation requires a completed masked_lr_00010 parent")
    return {**specification, "continuation_of": "masked_lr_00010_probe",
            "development_comparison": {"games": 300, "seed_start": 850000}}


def main():
    register_probe_arm()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-fork", type=Path)
    parser.add_argument("--additional-steps", type=int, default=100_000_000)
    parser.add_argument("--group")
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    if args.resume_run:
        if args.parent_fork or args.group:
            parser.error("--resume-run cannot be combined with a new long continuation")
        runner.resume_fork(args.resume_run, preview=args.preview)
        return
    if args.parent_fork is None:
        parser.error("--parent-fork is required for a new long continuation")
    specification = long_plan(args.parent_fork, args.additional_steps)
    if args.preview:
        print(json.dumps(specification, indent=2))
        return
    group = args.group or datetime.now().astimezone().strftime("%Y-%m-%d_%H%M%S_masked_lr_00010_long")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", group):
        parser.error("--group may contain only letters, digits, underscores and hyphens")
    runner.run_batch(specification, [TREATMENT], group)


if __name__ == "__main__":
    main()
