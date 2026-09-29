"""Launch isolated masking-by-LR forks from one complete v2 checkpoint.

Preview is read-only. Training uses new run directories, restores the parent
optimizer and live simulator state, and collects fresh rollouts for every arm.
"""

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import shutil
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.utils import update_learning_rate

from experiment_catalog import update_catalog
from reproduction.environment import ENGINE_REVISION, ReproductionVecEnv
from reproduction.policy import PublishedPolicy
from reproduction.training import Tracking
from resume_reproduction import active_workers, continuation_lock, inspect_run
from run_layout import PROJECT_ROOT, RUNS_ROOT
from run_provenance import snapshot_resume, snapshot_source
from .algorithm import ForkRecurrentPPO
from .masking import MaskedPublishedPolicy


ARMS = {
    "unmasked_lr_00015": (False, 0.00015),
    "masked_lr_00015": (True, 0.00015),
    "unmasked_lr_00005": (False, 0.00005),
    "masked_lr_00005": (True, 0.00005),
}
SCHEMA = "recurrent_mask_lr_fork_v1"
GROUP_ROOT = PROJECT_ROOT / "results" / "fork_batches"
FORK_CRITICAL_SOURCE = (
    "src/reproduction/environment.py", "src/reproduction/policy.py",
    "src/reproduction/training.py", "src/reproduction/evaluation.py",
    "src/reproduction_forks/masking.py", "src/reproduction_forks/algorithm.py",
    "src/reproduction_forks/runner.py")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plan(parent_run, parent_steps, requested_steps):
    parent_run = Path(parent_run).resolve()
    inspected = inspect_run(parent_run, requested_steps)
    if inspected["from_steps"] != parent_steps:
        raise ValueError("Select the exact latest complete fixed-budget checkpoint, not a validation peak")
    if requested_steps < 1:
        raise ValueError("Additional transitions must be positive")
    config = read_json(parent_run / "metadata/config.json")
    method = read_json(parent_run / "metadata/method.json")
    if method.get("engine_revision") != ENGINE_REVISION:
        raise ValueError("The parent uses a different simulator revision")
    checkpoint = Path(inspected["checkpoint"])
    rollout = config["ppo"]["n_envs"] * config["ppo"]["n_steps"]
    rounded = math.ceil(requested_steps / rollout) * rollout
    return {"parent_run": str(parent_run), "parent_checkpoint": str(checkpoint),
            "parent_steps": parent_steps, "parent_status": inspected["status"],
            "compatibility_issues": inspected["compatibility_issues"],
            "requested_additional_steps": requested_steps,
            "rounded_additional_steps": rounded, "endpoint_steps": parent_steps + rounded,
            "rollout_size": rollout, "training_seed": config["training"]["seed"],
            "arms": {name: {"mask_training": mask, "effective_lr": rate}
                     for name, (mask, rate) in ARMS.items()},
            "development_comparison": {"games": 300, "seed_start": 820000}}


def require_ready(specification):
    if specification["parent_status"] != "complete":
        raise ValueError("The selected parent run is not complete; finish the allocated batch first")
    if specification["compatibility_issues"]:
        raise ValueError("Parent/source compatibility issues: " +
                         "; ".join(specification["compatibility_issues"]))
    workers = active_workers()
    if workers:
        raise RuntimeError("A recurrent trainer or evaluator is already running")


def arm_config(parent_config, specification, arm):
    config = copy.deepcopy(parent_config)
    mask, rate = ARMS[arm]
    config["schema"] = SCHEMA
    config["ppo"]["learning_rate"] = rate
    config["training"]["steps"] = specification["rounded_additional_steps"]
    config["training"]["lr_schedule"] = {"milestones": [], "factor": 1.0}
    config["fork"] = {"arm": arm, "mask_training": mask,
                      "parent_steps": specification["parent_steps"],
                      "effective_lr": rate}
    return config


def restore_parent_for_fork(checkpoint, env, mask_training, effective_lr, device):
    """Read parent weights/optimizer/runtime without changing any parent file."""
    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint / "runtime.pt", map_location="cpu", weights_only=False)
    policy_class = MaskedPublishedPolicy if mask_training else PublishedPolicy
    model = ForkRecurrentPPO.load(
        checkpoint / "model.zip", env=env, device=device, force_reset=False,
        custom_objects={"policy_class": policy_class})
    restore_runtime_state(state, model, env, mask_training, effective_lr, device,
                          reset_instrumentation=True)
    return model


def restore_runtime_state(state, model, env, mask_training, effective_lr, device,
                          *, reset_instrumentation=False):
    if state["steps"] != model.num_timesteps or state["engine_revision"] != ENGINE_REVISION:
        raise ValueError("Checkpoint model, runtime, or simulator revision disagree")
    if isinstance(model.policy, MaskedPublishedPolicy) != mask_training:
        raise ValueError("Checkpoint policy class and mask setting disagree")
    for name, array in state["native"].items():
        getattr(env.native, name)[:] = array
    env.counter = state["counter"]
    env.returns = state["returns"].copy()
    env.invalid_count = 0 if reset_instrumentation else state["invalid_count"]
    env.action_count = 0 if reset_instrumentation else state["action_count"]
    recurrent = state["last_lstm_states"]
    model._last_lstm_states = RNNStates(
        tuple(x.to(device) for x in recurrent.pi),
        tuple(x.to(device) for x in recurrent.vf))
    model._last_obs = state["last_obs"].copy()
    model._last_episode_starts = state["last_episode_starts"].copy()
    model.learning_rate = effective_lr
    model.lr_schedule = lambda _: effective_lr
    model.policy.lr_schedule = model.lr_schedule
    model.absolute_lr_schedule = {"milestones": [], "factor": 1.0}
    model.fork_mask_training = mask_training
    update_learning_rate(model.policy.optimizer, effective_lr)
    torch.set_rng_state(state["torch_rng"])
    if device == "cuda" and state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])


def inspect_fork(run):
    run = Path(run).resolve()
    if not run.is_relative_to(RUNS_ROOT.resolve()):
        raise ValueError("Fork run must be inside results/runs")
    config = read_json(run / "metadata/config.json")
    fork = read_json(run / "metadata/fork.json")
    if config.get("schema") != SCHEMA or fork.get("arm") not in ARMS:
        raise ValueError("This is not a masking-by-LR fork run")
    markers = sorted((run / "models/checkpoints").glob("step_*/complete.json"))
    if not markers:
        raise ValueError("No complete fork checkpoint exists")
    checkpoint = markers[-1].parent
    start = read_json(checkpoint / "complete.json")["steps"]
    target = fork["parent_steps"] + fork["rounded_additional_steps"]
    if start > target:
        raise ValueError("Saved fork checkpoint exceeds its allocated endpoint")
    provenance = read_json(run / "metadata/provenance.json")
    source = provenance.get("resumes", [])[-1] if provenance.get("resumes") else provenance
    mismatches = [name for name in FORK_CRITICAL_SOURCE
                  if source.get("sha256", {}).get(name) != sha256(PROJECT_ROOT / name)]
    method = read_json(run / "metadata/method.json")
    import importlib.metadata
    for name, package in (("torch", "torch"), ("sb3_contrib", "sb3-contrib"), ("numba", "numba")):
        actual = importlib.metadata.version(package)
        if method.get("versions", {}).get(name) != actual:
            mismatches.append(f"{package} version changed")
    return {"run": str(run), "checkpoint": str(checkpoint), "saved_steps": start,
            "target_steps": target, "remaining_steps": target - start,
            "arm": fork["arm"], "mask_training": fork["mask_training"],
            "effective_lr": fork["effective_lr"], "status": read_json(run / "metadata/status.json")["status"],
            "compatibility_issues": mismatches}


def prepare_fork_resume_session(specification):
    run = Path(specification["run"])
    start = specification["saved_steps"]
    label = datetime.now().astimezone().strftime("%Y-%m-%d_%H%M%S_%f") + f"_from_{start}"
    session_dir = run / "metadata/continuations" / label
    session_dir.mkdir(parents=True)
    for relative in ("data/metrics.csv", "data/episodes.csv", "data/fork_updates.csv",
                     "models/best.zip", "models/best.json",
                     "metadata/status.json", "metadata/config.json", "metadata/method.json",
                     "metadata/provenance.json", "metadata/fork.json"):
        source = run / relative
        if source.exists():
            target = session_dir / "before" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    if (run / "evaluation").exists():
        shutil.copytree(run / "evaluation", session_dir / "before/evaluation")
    write_json(session_dir / "session.json", {**specification,
               "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
               "status": "prepared"})
    return session_dir


def truncate_fork_diagnostics(path, saved_steps):
    """Keep only updates backed by a complete checkpoint before appending."""
    if not path.exists():
        return
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = [row for row in reader if int(row["steps"]) <= saved_steps]
    if fields is None:
        raise ValueError(f"Missing fork diagnostics header: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def resume_fork(run, *, preview=False):
    specification = inspect_fork(run)
    if preview:
        print(json.dumps(specification, indent=2))
        return
    if specification["compatibility_issues"]:
        raise ValueError("Fork/source compatibility issues: " +
                         "; ".join(specification["compatibility_issues"]))
    if specification["remaining_steps"] == 0:
        raise ValueError("The fork already has a complete endpoint checkpoint")
    workers = active_workers()
    if workers:
        raise RuntimeError("A recurrent trainer or evaluator is already running")
    with continuation_lock():
        session_dir = prepare_fork_resume_session(specification)
        run = Path(specification["run"])
        config = read_json(run / "metadata/config.json")
        checkpoint = Path(specification["checkpoint"])
        device = config["training"]["device"]
        torch.set_num_threads(config["training"]["torch_threads"])
        env = ReproductionVecEnv(config["ppo"]["n_envs"], config["training"]["seed"],
                                 run / "data/episodes.csv", **config["environment"])
        status = "interrupted"
        with (session_dir / "terminal.log").open("w", encoding="utf-8", buffering=1) as log:
            out, err = Tee(sys.stdout, log), Tee(sys.stderr, log)
            try:
                with redirect_stdout(out), redirect_stderr(err):
                    state = torch.load(checkpoint / "runtime.pt", map_location="cpu", weights_only=False)
                    model = ForkRecurrentPPO.load(checkpoint / "model.zip", env=env,
                                                  device=device, force_reset=False)
                    restore_runtime_state(state, model, env, specification["mask_training"],
                                          specification["effective_lr"], device)
                    if env.log:
                        env.log.flush()
                        env.log.truncate(state["episode_log_size"])
                        env.log.seek(0, 2)
                    for path in (run / "evaluation").glob("step_*.csv"):
                        if int(path.stem.split("_")[1]) > specification["saved_steps"]:
                            path.unlink()
                    for name in ("best.zip", "best.json"):
                        saved = checkpoint / name
                        active = run / "models" / name
                        if saved.exists():
                            shutil.copy2(saved, active)
                        elif active.exists():
                            active.unlink()
                    snapshot_resume(run, specification["saved_steps"])
                    tracking = Tracking(env, run, config, model.num_timesteps)
                    model.tracking = tracking
                    model.fork_diagnostics_path = run / "data/fork_updates.csv"
                    truncate_fork_diagnostics(model.fork_diagnostics_path,
                                              specification["saved_steps"])
                    write_json(run / "metadata/status.json", {"status": "in progress"})
                    print(f"Resuming fork {specification['arm']} from {model.num_timesteps:,} "
                          f"to {specification['target_steps']:,} transitions", flush=True)
                    model.learn(total_timesteps=specification["remaining_steps"],
                                callback=tracking, reset_num_timesteps=False)
                    status = "complete"
                    print(f"Completed fork {specification['arm']}", flush=True)
            finally:
                env.close()
                write_json(run / "metadata/status.json", {"status": status})
                write_json(session_dir / "session.json", {**specification,
                           "started_at": read_json(session_dir / "session.json")["started_at"],
                           "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                           "status": status})
                if (run / "data/metrics.csv").exists():
                    update_catalog(RUNS_ROOT)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def run_arm(specification, arm, group):
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    run_dir = RUNS_ROOT / f"{group}_{arm}"
    if run_dir.exists():
        existing = read_json(run_dir / "metadata/status.json")
        if existing.get("status") == "complete":
            record = read_json(run_dir / "metadata/fork.json")
            if (record.get("arm") != arm or record.get("group") != group or
                    record.get("parent_model_sha256") != specification["parent_model_sha256"] or
                    record.get("parent_runtime_sha256") != specification["parent_runtime_sha256"] or
                    not (run_dir / "models/checkpoints" /
                         f"step_{specification['endpoint_steps']:012d}" / "complete.json").is_file()):
                raise ValueError(f"Existing completed arm has different provenance or endpoint: {run_dir}")
            print(f"Skipping complete arm: {run_dir}", flush=True)
            return run_dir
        raise ValueError(f"An unfinished arm already exists: {run_dir}; resume it explicitly")
    parent_run = Path(specification["parent_run"])
    parent_config = read_json(parent_run / "metadata/config.json")
    config = arm_config(parent_config, specification, arm)
    mask, rate = ARMS[arm]
    for name in ("metadata", "data", "evaluation", "models", "report"):
        (run_dir / name).mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "metadata/config.json", config)
    checkpoint = Path(specification["parent_checkpoint"])
    fork_record = {"group": group, "arm": arm, "parent_run": specification["parent_run"],
                   "parent_checkpoint": str(checkpoint), "parent_steps": specification["parent_steps"],
                   "parent_model_sha256": specification["parent_model_sha256"],
                   "parent_runtime_sha256": specification["parent_runtime_sha256"],
                   "mask_training": mask, "effective_lr": rate,
                   "optimizer_state": "restored without reset",
                   "simulator_and_recurrent_state": "restored; fresh rollouts under the arm distribution",
                   "requested_additional_steps": specification["requested_additional_steps"],
                   "rounded_additional_steps": specification["rounded_additional_steps"],
                   "started_at": datetime.now().astimezone().isoformat(timespec="seconds")}
    write_json(run_dir / "metadata/fork.json", fork_record)
    write_json(run_dir / "metadata/status.json", {"status": "prepared"})
    snapshot_source(run_dir)
    device = config["training"]["device"]
    torch.set_num_threads(config["training"]["torch_threads"])
    env = ReproductionVecEnv(config["ppo"]["n_envs"], config["training"]["seed"],
                             run_dir / "data/episodes.csv", **config["environment"])
    status = "interrupted"
    with (run_dir / "metadata/terminal.log").open("w", encoding="utf-8", buffering=1) as log:
        out, err = Tee(sys.stdout, log), Tee(sys.stderr, log)
        try:
            with redirect_stdout(out), redirect_stderr(err):
                model = restore_parent_for_fork(checkpoint, env, mask, rate, device)
                parent_method = read_json(parent_run / "metadata/method.json")
                method = {**parent_method,
                          "method": "RecurrentPPO masking-by-LR fork, shared actor/critic gradients",
                          "initialization": f"local random-weight checkpoint at {specification['parent_steps']:,} transitions",
                          "device": device,
                          "fork": {"parent_run": parent_run.name,
                                   "parent_steps": specification["parent_steps"],
                                   "mask_training": mask, "effective_lr": rate}}
                write_json(run_dir / "metadata/method.json", method)
                tracking = Tracking(env, run_dir, config, model.num_timesteps)
                model.tracking = tracking
                model.fork_diagnostics_path = run_dir / "data/fork_updates.csv"
                write_json(run_dir / "metadata/status.json", {"status": "in progress"})
                print(f"Fork {arm}: {run_dir}\nParent: {checkpoint}\n"
                      f"Mask: {mask} | fixed LR: {rate} | new transitions: "
                      f"{specification['rounded_additional_steps']:,}", flush=True)
                model.learn(total_timesteps=specification["rounded_additional_steps"],
                            callback=tracking, reset_num_timesteps=False)
                status = "complete"
                print(f"Completed fork {arm} at {model.num_timesteps:,} transitions", flush=True)
        finally:
            env.close()
            write_json(run_dir / "metadata/status.json", {"status": status})
            if (run_dir / "data/metrics.csv").exists() and (run_dir / "metadata/method.json").exists():
                update_catalog(RUNS_ROOT)
    return run_dir


def run_batch(specification, arms, group):
    require_ready(specification)
    GROUP_ROOT.mkdir(parents=True, exist_ok=True)
    group_dir = GROUP_ROOT / group
    if group_dir.exists():
        existing = read_json(group_dir / "plan.json")
        if {key: existing[key] for key in ("parent_run", "parent_steps", "endpoint_steps")} != {
                key: specification[key] for key in ("parent_run", "parent_steps", "endpoint_steps")}:
            raise ValueError("The existing group selects a different parent or budget")
        checkpoint = Path(existing["parent_checkpoint"])
        if (sha256(checkpoint / "model.zip") != existing["parent_model_sha256"] or
                sha256(checkpoint / "runtime.pt") != existing["parent_runtime_sha256"]):
            raise ValueError("Parent checkpoint changed since the fork group was created")
    else:
        group_dir.mkdir()
        specification = {**specification,
                         "parent_model_sha256": sha256(Path(specification["parent_checkpoint"]) / "model.zip"),
                         "parent_runtime_sha256": sha256(Path(specification["parent_checkpoint"]) / "runtime.pt"),
                         "group": group, "created_at": datetime.now().astimezone().isoformat(timespec="seconds")}
        write_json(group_dir / "plan.json", specification)
    specification = read_json(group_dir / "plan.json")
    with continuation_lock():
        for arm in arms:
            run_arm(specification, arm, group)
        if len(arms) == len(ARMS):
            from .compare import compare_group
            compare_group(group_dir)
    print(f"Fork group: {group_dir}", flush=True)


def compare_existing_group(group, *, preview=False):
    group_dir = (GROUP_ROOT / group).resolve()
    if not group_dir.is_relative_to(GROUP_ROOT.resolve()):
        raise ValueError("Comparison group must be inside results/fork_batches")
    plan_file = group_dir / "plan.json"
    if not plan_file.is_file():
        raise ValueError(f"No fork group plan: {plan_file}")
    plan_data = read_json(plan_file)
    if preview:
        print(json.dumps({"group": group, "plan": plan_data,
                          "comparison": "300 paired natural games per endpoint/best model"}, indent=2))
        return
    workers = active_workers()
    if workers:
        raise RuntimeError("A recurrent trainer or evaluator is already running")
    with continuation_lock():
        from .compare import compare_group
        compare_group(group_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", type=Path)
    parser.add_argument("--parent-steps", type=int)
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--compare-group")
    parser.add_argument("--additional-steps", type=int, default=5_000_000)
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--group")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    if args.compare_group:
        if args.parent_run or args.parent_steps or args.resume_run or args.arm or args.all or args.group:
            parser.error("--compare-group cannot be combined with training selection")
        compare_existing_group(args.compare_group, preview=args.preview)
        return
    if args.resume_run:
        if args.parent_run or args.parent_steps or args.arm or args.all or args.group:
            parser.error("--resume-run cannot be combined with a new fork selection")
        resume_fork(args.resume_run, preview=args.preview)
        return
    if not args.parent_run or args.parent_steps is None:
        parser.error("--parent-run and --parent-steps are required for new forks")
    if args.all == bool(args.arm) and not args.preview:
        parser.error("Select --all or exactly one --arm")
    specification = plan(args.parent_run, args.parent_steps, args.additional_steps)
    if args.preview:
        print(json.dumps(specification, indent=2))
        return
    group = args.group or datetime.now().astimezone().strftime("%Y-%m-%d_%H%M%S_mask_lr")
    arms = list(ARMS) if args.all else [args.arm]
    run_batch(specification, arms, group)


if __name__ == "__main__":
    main()
