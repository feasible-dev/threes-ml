"""Fresh masked PPO training with stronger potential for a largest tile on an edge."""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from reproduction import training as core
from reproduction.environment import NativeBatch, ReproductionVecEnv
from reproduction_forks.algorithm import ForkRecurrentPPO
from reproduction_forks.corner_reward import corner_delta
from reproduction_forks.masking import MaskedPublishedPolicy
from resume_reproduction import active_workers, continuation_lock
from run_layout import PROJECT_ROOT, RUNS_ROOT
from run_provenance import snapshot_source


CONFIG = PROJECT_ROOT / "configs/fresh_corner_strong_lr00015.json"
CORNER_WEIGHT = 0.20
SIDE_WEIGHT = 0.05
SOURCE_FILES = ("src/fresh_corner_strong.py", "src/reproduction_forks/corner_reward.py",
                "src/reproduction_forks/algorithm.py", "src/reproduction_forks/masking.py",
                "src/reproduction/environment.py", "src/reproduction/policy.py",
                "src/reproduction/training.py", "src/reproduction/evaluation.py")
REWARD = {"type": "largest_tile_edge_corner_potential_v1", "corner_weight": CORNER_WEIGHT,
          "side_weight": SIDE_WEIGHT, "discount": 0.999, "min_rank": 5, "min_tile": 12}


class StrongCornerNativeBatch(NativeBatch):
    def step(self, actions):
        before = self.board.copy()
        observations, rewards, done, stats = super().step(actions)
        rewards = np.clip(rewards + corner_delta(before, self.board, done, weight=CORNER_WEIGHT),
                          -1, 1).astype(np.float32)
        return observations, rewards, done, stats


class StrongCornerVecEnv(ReproductionVecEnv):
    def __init__(self, count, run_seed, episode_path=None, *, bootstrap_timeouts=False, **rules):
        super().__init__(count, run_seed, episode_path,
                         bootstrap_timeouts=bootstrap_timeouts, **rules)
        self.native = StrongCornerNativeBatch(count, **rules)


class FreshMaskedCornerPPO(ForkRecurrentPPO):
    def train(self):
        self.fork_mask_training = True
        return super().train()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=18_000_000)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    config = core.load_config(CONFIG)
    if args.steps < 1 or not 0 <= args.seed < 2**31:
        parser.error("Steps must be positive and seed must fit in 31 bits")
    rollout = config["ppo"]["n_envs"] * config["ppo"]["n_steps"]
    if args.resume_run:
        run = args.resume_run.resolve()
        if not run.is_relative_to(RUNS_ROOT.resolve()):
            parser.error("Resume run must be inside results/runs")
        manifest = read_json(run / "metadata/experiment.json")
        frozen = read_json(run / "metadata/config.json")
        if manifest.get("reward") != REWARD or not manifest.get("mask_training") or frozen["ppo"]["learning_rate"] != 0.00015:
            parser.error("This is not a matching strong-corner run")
        provenance = read_json(run / "metadata/provenance.json")
        source = provenance.get("resumes", [])[-1] if provenance.get("resumes") else provenance
        from reproduction_forks.runner import sha256
        if any(source["sha256"].get(name) != sha256(PROJECT_ROOT / name) for name in SOURCE_FILES):
            parser.error("Strong-corner source changed since this run")
        markers = sorted((run / "models/checkpoints").glob("step_*/complete.json"))
        if not markers:
            parser.error("No complete checkpoint to resume")
        saved = read_json(markers[-1])["steps"]
        remaining = manifest["target_steps"] - saved
        if remaining <= 0:
            parser.error("The original allocation is complete")
        plan = {"run": str(run), "checkpoint": str(markers[-1].parent),
                "saved_steps": saved, "remaining_steps": remaining,
                "target_steps": manifest["target_steps"], "learning_rate": 0.00015,
                "reward": REWARD, "mask_training": True}
    else:
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H%M%S")
        run = RUNS_ROOT / f"{stamp}_fresh_corner_strong_lr15_s{args.seed}"
        rounded = math.ceil(args.steps / rollout) * rollout
        plan = {"run": str(run), "requested_steps": args.steps,
                "rounded_steps": rounded, "target_steps": rounded,
                "seed": args.seed, "learning_rate": 0.00015,
                "reward": REWARD, "mask_training": True,
                "initialization": "random weights", "curriculum_scaffold": 0.5}
        manifest = {"experiment": "fresh_masked_strong_corner_v1",
                    "reward": REWARD, "mask_training": True,
                    "initialization": "random weights", "target_steps": rounded,
                    "requested_steps": args.steps,
                    "created_at": datetime.now().astimezone().isoformat(timespec="seconds")}
    if args.preview:
        print(json.dumps(plan, indent=2))
        return
    if active_workers():
        raise RuntimeError("Another recurrent trainer or evaluator is running")
    if not args.resume_run and run.exists():
        parser.error(f"Run already exists: {run}")

    original_snapshot = core.snapshot_source
    original_tracking = core.Tracking

    def save_source_and_manifest(run_dir):
        snapshot_source(run_dir)
        (run_dir / "metadata/experiment.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    class CornerTracking(original_tracking):
        def __init__(self, env, run_dir, configuration, start_steps):
            super().__init__(env, run_dir, configuration, start_steps)
            method_path = run_dir / "metadata/method.json"
            method = read_json(method_path)
            method["method"] = "Masked recurrent PPO with potential-based largest-tile edge/corner reward"
            method["corner_reward"] = REWARD
            method["mask_training"] = True
            method_path.write_text(json.dumps(method, indent=2) + "\n", encoding="utf-8")

    core.ReproductionVecEnv = StrongCornerVecEnv
    core.PublishedPolicy = MaskedPublishedPolicy
    core.TrackedRecurrentPPO = FreshMaskedCornerPPO
    core.snapshot_source = save_source_and_manifest
    core.Tracking = CornerTracking
    arguments = (["--resume", str(run), "--steps", str(remaining)] if args.resume_run else
                 ["--config", str(CONFIG), "--run-dir", str(run),
                  "--steps", str(args.steps), "--seed", str(args.seed)])
    previous_argv = sys.argv
    try:
        with continuation_lock():
            sys.argv = [previous_argv[0], *arguments]
            core.main()
    finally:
        sys.argv = previous_argv
        core.snapshot_source = original_snapshot
        core.Tracking = original_tracking


if __name__ == "__main__":
    main()
