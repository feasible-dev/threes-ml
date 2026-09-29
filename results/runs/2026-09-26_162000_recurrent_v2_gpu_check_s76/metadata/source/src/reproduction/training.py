"""Recurrent PPO training, checkpoint continuation, and observable experiment tracking."""

import argparse
import csv
import json
import math
import random
import shutil
from collections import deque
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import update_learning_rate

from dashboard import write_report
from experiment_catalog import update_catalog, write_run_card
from run_layout import RUNS_ROOT
from run_provenance import snapshot_resume, snapshot_source
from .environment import ENGINE_REVISION, ReproductionVecEnv
from .evaluation import evaluate
from .policy import PublishedPolicy


FIELDS = ("steps", "updates", "seconds", "steps_per_second", "train_score_mean", "train_episodes",
          "value_loss", "policy_loss", "entropy_loss", "approx_kl", "explained_variance",
          "invalid_rate", "invalid_rate_recent", "curriculum_fraction_recent", "learning_rate",
          "eval_mean", "eval_median", "eval_max", "eval_games", "eval_384",
          "eval_3072", "eval_6144", "eval_truncated", "gpu_memory_mb")


def load_config(path):
    c = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    required = {"schema", "model", "environment", "ppo", "training", "evaluation"}
    if set(c) != required or c["schema"] not in ("recurrent_reproduction_v1", "recurrent_reproduction_v2"):
        raise ValueError("Expected a recurrent_reproduction_v1 or v2 configuration")
    sections = {
        "model": {"hidden_size"},
        "environment": {"heuristic", "snake_weight", "scaffold", "terminal_rank"},
        "ppo": {"n_envs", "n_steps", "batch_size", "n_epochs", "learning_rate", "gamma", "gae_lambda",
                "ent_coef", "clip_range", "vf_coef", "max_grad_norm", "target_kl"},
        "training": {"seed", "steps", "device", "torch_threads", "checkpoint_interval"},
        "evaluation": {"games", "seed", "interval", "batch_size"},
    }
    if c["schema"] == "recurrent_reproduction_v2":
        sections["environment"] |= {"reward_scaler", "bootstrap_timeouts"}
        sections["training"] |= {"lr_schedule"}
    for section, names in sections.items():
        if set(c[section]) != names:
            raise ValueError(f"Unexpected or missing keys in {section}; expected {sorted(names)}")
    for section, names in {"model": ["hidden_size"], "ppo": ["n_envs", "n_steps", "batch_size", "n_epochs"],
                           "training": ["steps", "torch_threads", "checkpoint_interval"],
                           "evaluation": ["games", "interval", "batch_size"]}.items():
        for name in names:
            if type(c[section][name]) is not int or c[section][name] < 1:
                raise ValueError(f"{section}.{name} must be a positive integer")
    p = c["ppo"]
    if p["batch_size"] < 2 or p["n_envs"] * p["n_steps"] % p["batch_size"]:
        raise ValueError("batch_size must divide n_envs*n_steps and be >= 2")
    if not 0 <= c["training"]["seed"] < 2**31:
        raise ValueError("Training run seed must be in [0,2**31)")
    if not 0 <= c["evaluation"]["seed"] < 2**63 - c["evaluation"]["games"]:
        raise ValueError("Evaluation seeds must be below 2**63")
    if c["training"]["device"] not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    for name in ("gamma", "gae_lambda"):
        if not 0 < p[name] <= 1:
            raise ValueError(f"{name} must be in (0,1]")
    for name in ("learning_rate", "clip_range", "max_grad_norm", "target_kl"):
        if not isinstance(p[name], (int, float)) or not math.isfinite(p[name]) or p[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("ent_coef", "vf_coef"):
        if not math.isfinite(p[name]) or p[name] < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if type(c["environment"]["heuristic"]) is not bool:
        raise ValueError("heuristic must be a boolean")
    if not 0 <= c["environment"]["scaffold"] <= 1 or c["environment"]["terminal_rank"] not in (15, 16):
        raise ValueError("Invalid curriculum probability or terminal rank")
    if not math.isfinite(c["environment"]["snake_weight"]) or c["environment"]["snake_weight"] < 0:
        raise ValueError("snake_weight must be finite and nonnegative")
    if c["schema"] == "recurrent_reproduction_v2":
        if type(c["environment"]["bootstrap_timeouts"]) is not bool:
            raise ValueError("bootstrap_timeouts must be a boolean")
        scale = c["environment"]["reward_scaler"]
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("reward_scaler must be positive and finite")
        schedule = c["training"]["lr_schedule"]
        if set(schedule) != {"milestones", "factor"}:
            raise ValueError("lr_schedule expects milestones and factor")
        milestones = schedule["milestones"]
        if (not isinstance(milestones, list) or
                any(type(s) is not int or s < 1 for s in milestones) or
                milestones != sorted(set(milestones)) or not 0 < schedule["factor"] <= 1):
            raise ValueError("Invalid absolute-step learning-rate schedule")
    return c


def scheduled_learning_rate(initial, schedule, steps):
    """Absolute transition milestones survive changes to the resume budget."""
    return initial * schedule["factor"] ** sum(steps >= s for s in schedule["milestones"])


def save_boundary(model, env, run_dir):
    destination = run_dir / "models" / "checkpoints" / f"step_{model.num_timesteps:012d}"
    destination.mkdir(parents=True, exist_ok=True)
    model.save(destination / "model.zip", exclude=["tracking"])
    if env.log:
        env.log.flush()
    state = {
        "engine_revision": ENGINE_REVISION,
        "steps": model.num_timesteps,
        "native": {name: getattr(env.native, name).copy() for name in ("board", "bag", "meta", "rng", "seeds")},
        "counter": env.counter, "returns": env.returns, "invalid_count": env.invalid_count,
        "action_count": env.action_count,
        "last_lstm_states": model._last_lstm_states,
        "last_obs": model._last_obs, "last_episode_starts": model._last_episode_starts,
        "episode_log_size": env.episode_path.stat().st_size if env.episode_path else 0,
        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
    }
    torch.save(state, destination / "runtime.pt")
    # Keep checkpoint selection consistent if later evaluations are discarded
    # when resuming an earlier complete boundary after interruption.
    for name in ("best.zip", "best.json"):
        source = run_dir / "models" / name
        if source.exists():
            shutil.copy2(source, destination / name)
    (destination / "complete.json").write_text(json.dumps({"steps": model.num_timesteps}), encoding="utf-8")
    return destination


def restore_boundary(env, run_dir, device):
    complete = sorted((run_dir / "models" / "checkpoints").glob("step_*/complete.json"))
    if not complete:
        raise ValueError("No complete update-boundary checkpoint exists")
    directory = complete[-1].parent
    state = torch.load(directory / "runtime.pt", map_location="cpu", weights_only=False)
    if state.get("engine_revision") != ENGINE_REVISION:
        raise ValueError("This checkpoint uses older game rules. Start a new run; use its frozen source for historical continuation.")
    model = TrackedRecurrentPPO.load(directory / "model.zip", env=env, device=device, force_reset=False)
    if state["steps"] != model.num_timesteps:
        raise ValueError("Checkpoint and simulator state disagree")
    for name, array in state["native"].items():
        getattr(env.native, name)[:] = array
    env.counter, env.returns = state["counter"], state["returns"]
    env.invalid_count, env.action_count = state["invalid_count"], state["action_count"]
    if env.log:
        env.log.flush()
        env.log.truncate(state["episode_log_size"])
    torch.set_rng_state(state["torch_rng"])
    if device == "cuda" and state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    # RecurrentPPO._setup_model overwrites the hidden state stored in model.zip.
    # Restore it from the runtime snapshot after SB3 has finished constructing the model.
    from sb3_contrib.common.recurrent.type_aliases import RNNStates
    if "last_lstm_states" not in state:
        raise ValueError("Checkpoint predates complete recurrent-state snapshots; exact continuation unavailable")
    recurrent = state["last_lstm_states"]
    model._last_lstm_states = RNNStates(
        tuple(x.to(device) for x in recurrent.pi),
        tuple(x.to(device) for x in recurrent.vf))
    model._last_obs, model._last_episode_starts = state["last_obs"], state["last_episode_starts"]
    for name in ("best.zip", "best.json"):
        if (directory / name).exists():
            shutil.copy2(directory / name, run_dir / "models" / name)
    print(f"Resuming weights, optimizer, simulator, memory and RNG at {model.num_timesteps:,} steps", flush=True)
    return model


class TrackedRecurrentPPO(RecurrentPPO):
    tracking = None
    absolute_lr_schedule = None

    def _update_learning_rate(self, optimizers):
        if self.absolute_lr_schedule is None:
            return super()._update_learning_rate(optimizers)
        rate = scheduled_learning_rate(float(self.learning_rate), self.absolute_lr_schedule, self.num_timesteps)
        self.logger.record("train/learning_rate", rate)
        for optimizer in optimizers if isinstance(optimizers, list) else [optimizers]:
            update_learning_rate(optimizer, rate)

    def train(self):
        super().train()
        if self.tracking:
            self.tracking.after_update(self)


class Tracking(BaseCallback):
    def __init__(self, env, run_dir, config, start_steps):
        super().__init__()
        self.env, self.run_dir, self.config = env, run_dir, config
        self.start = perf_counter()
        self.start_steps = start_steps
        self.last_print = self.start
        self.last_eval = start_steps
        self.last_save = start_steps
        self.best = -math.inf
        self.scores = deque(maxlen=100)
        self.previous_seconds = 0.0
        self.last_invalid = env.invalid_count
        self.last_actions = env.action_count
        self.curriculum_actions = 0
        self.window_actions = 0
        path = run_dir / "data" / "metrics.csv"
        if path.exists():
            with path.open(encoding="utf-8", newline="") as stream:
                rows = [row for row in csv.DictReader(stream) if int(row["steps"]) <= start_steps]
            if rows:
                self.previous_seconds = float(rows[-1]["seconds"])
                self.best = max((float(row["eval_mean"]) for row in rows if row["eval_mean"]), default=-math.inf)
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
        else:
            with path.open("w", encoding="utf-8", newline="") as stream:
                csv.DictWriter(stream, fieldnames=FIELDS).writeheader()

    def _on_step(self):
        # Autoreset has already happened: use rollout episode info for finished
        # slots and the current flag for slots still playing.
        for i, info in enumerate(self.locals["infos"]):
            self.curriculum_actions += int(info.get("curriculum", self.env.native.meta[i, 7]))
        self.window_actions += self.env.num_envs
        for info in self.locals["infos"]:
            if "score" in info and not info.get("curriculum"):
                self.scores.append(info["score"])
        return True

    def after_update(self, model):
        values = model.logger.name_to_value
        elapsed = perf_counter() - self.start
        first = model.num_timesteps == self.start_steps + model.n_envs * model.n_steps
        final = model.num_timesteps >= model._total_timesteps
        evaluation_due = first or final or model.num_timesteps - self.last_eval >= self.config["evaluation"]["interval"]
        row = {key: "" for key in FIELDS}
        row.update(steps=model.num_timesteps, updates=model.num_timesteps // (model.n_envs * model.n_steps),
                   seconds=round(self.previous_seconds + elapsed, 3),
                   steps_per_second=round((model.num_timesteps - self.start_steps) / max(elapsed, .001), 1),
                   train_score_mean=float(np.mean(self.scores)) if self.scores else "",
                   train_episodes=self.env.counter - model.n_envs,
                   invalid_rate=self.env.invalid_count / max(1, self.env.action_count),
                   invalid_rate_recent=(self.env.invalid_count - self.last_invalid) / max(1, self.env.action_count - self.last_actions),
                   curriculum_fraction_recent=self.curriculum_actions / max(1, self.window_actions),
                   learning_rate=float(values.get("train/learning_rate", model.learning_rate)),
                   gpu_memory_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1) if model.device.type == "cuda" else 0)
        self.last_invalid, self.last_actions = self.env.invalid_count, self.env.action_count
        self.curriculum_actions = self.window_actions = 0
        for column, key in (("value_loss", "value_loss"), ("policy_loss", "policy_gradient_loss"),
                            ("entropy_loss", "entropy_loss"), ("approx_kl", "approx_kl"),
                            ("explained_variance", "explained_variance")):
            row[column] = float(values.get("train/" + key, float("nan")))
        if not all(math.isfinite(row[key]) for key in ("value_loss", "policy_loss", "entropy_loss", "approx_kl")):
            raise FloatingPointError("Non-finite optimization metrics; stopping before checkpointing")
        if evaluation_due:
            ev = self.config["evaluation"]
            measured = evaluate(model.policy, ev["games"], ev["seed"], batch_size=ev["batch_size"],
                                terminal_rank=self.config["environment"]["terminal_rank"],
                                output=self.run_dir / "evaluation" / f"step_{model.num_timesteps:012d}.csv")
            row.update(measured)
            self.last_eval = model.num_timesteps
            if measured["eval_mean"] > self.best:
                self.best = measured["eval_mean"]
                model.save(self.run_dir / "models" / "best.zip", exclude=["tracking"])
                (self.run_dir / "models" / "best.json").write_text(
                    json.dumps({"steps": model.num_timesteps, **measured}, indent=2), encoding="utf-8")
        with (self.run_dir / "data" / "metrics.csv").open("a", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=FIELDS).writerow(row)
        if first or final or model.num_timesteps - self.last_save >= self.config["training"]["checkpoint_interval"]:
            save_boundary(model, self.env, self.run_dir)
            self.last_save = model.num_timesteps
        if evaluation_due or perf_counter() - self.last_print >= 15:
            text = (f"{model.num_timesteps:>12,} steps | {row['steps_per_second']:>7,.0f} steps/s | "
                    f"Vloss {row['value_loss']:.3f} | entropy {-row['entropy_loss']:.3f} | "
                    f"invalid recent {row['invalid_rate_recent']:.1%} | lr {row['learning_rate']:.2g} | "
                    f"curriculum {row['curriculum_fraction_recent']:.0%}")
            if evaluation_due:
                text += (f" | validation {row['eval_mean']:.1f} | >=384 {row['eval_384']:.1%} | >=3072 {row['eval_3072']:.1%} "
                         f">=6144 {row['eval_6144']:.1%}")
            print(text, flush=True)
            self.last_print = perf_counter()
            write_report(self.run_dir)
            write_run_card(self.run_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/reproduction_base.json"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, help="New transitions for this invocation")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if args.resume and args.run_dir:
        parser.error("Use --resume or --run-dir")
    config = load_config(args.resume / "metadata" / "config.json" if args.resume else args.config)
    if args.seed is not None:
        if args.resume and args.seed != config["training"]["seed"]:
            parser.error("A resumed run keeps its training seed")
        config["training"]["seed"] = args.seed
    if args.device:
        config["training"]["device"] = args.device
    steps = args.steps if args.steps is not None else config["training"]["steps"]
    if steps <= 0 or not 0 <= config["training"]["seed"] < 2**31:
        parser.error("Steps must be positive and seed must fit in 31 bits")
    if not args.resume:
        config["training"]["steps"] = steps
    device = config["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable. Run setup_reproduction.ps1 and use .venv-repro Python.")
    torch.set_num_threads(config["training"]["torch_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run_dir = args.resume or args.run_dir or RUNS_ROOT / datetime.now().strftime("%Y-%m-%d_%H%M%S_recurrent")
    if not args.resume:
        if run_dir.exists():
            parser.error(f"Directory exists: {run_dir}")
        for name in ("metadata", "data", "evaluation", "models", "report"):
            (run_dir / name).mkdir(parents=True, exist_ok=True)
        (run_dir / "metadata" / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        snapshot_source(run_dir)
    env = ReproductionVecEnv(config["ppo"]["n_envs"], config["training"]["seed"],
                             run_dir / "data" / "episodes.csv", **config["environment"])
    status = "interrupted"
    try:
        if args.resume:
            model = restore_boundary(env, run_dir, device)
            snapshot_resume(run_dir, model.num_timesteps)
        else:
            options = {key: value for key, value in config["ppo"].items() if key != "n_envs"}
            model = TrackedRecurrentPPO(PublishedPolicy, env,
                                       policy_kwargs=config["model"], **options,
                                       seed=config["training"]["seed"], device=device, verbose=0)
            model.absolute_lr_schedule = config["training"].get("lr_schedule")
            import sb3_contrib, numba
            method = {"engine_revision": ENGINE_REVISION,
                      "method": "RecurrentPPO, shared actor/critic gradients", "initialization": "random; no imported weights",
                      "parameters": sum(p.numel() for p in model.policy.parameters()),
                      "device": str(model.device), "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                      "versions": {"torch": torch.__version__, "sb3_contrib": sb3_contrib.__version__, "numba": numba.__version__},
                      "architecture_source": "ONNX SHA256 63bff54b23de09801fb2bc77ca8856ed781c2abaa2efe896c3da15e5d810eff2",
                      "rule_source": "threes-web 9527295de72333c91592fff45e8ccbcabd6ba80d/threes.h",
                      "limitations": ["SB3 recurrent PPO instead of PufferLib; final training hyperparameters unavailable",
                                      "Independent native simulator; separate RNG and corrected optional curriculum initialization",
                                      "Target success rate must be measured; architecture equivalence alone does not establish training replication"]}
            (run_dir / "metadata" / "method.json").write_text(json.dumps(method, indent=2), encoding="utf-8")
        tracking = Tracking(env, run_dir, config, model.num_timesteps)
        model.tracking = tracking
        (run_dir / "metadata" / "status.json").write_text(json.dumps({"status": "in progress"}), encoding="utf-8")
        print(f"Run: {run_dir.resolve()}\nDevice: {device} | {config['ppo']['n_envs']} environments | "
              f"{sum(p.numel() for p in model.policy.parameters()):,} parameters | {steps:,} additional transitions", flush=True)
        print("Training starts from random weights for a new run. First call includes simulator compilation.", flush=True)
        model.learn(total_timesteps=steps, callback=tracking, reset_num_timesteps=not args.resume)
        status = "complete"
    except KeyboardInterrupt:
        print("Interrupted. Resume from the last complete update-boundary checkpoint.", flush=True)
        raise
    finally:
        env.close()
        (run_dir / "metadata" / "status.json").write_text(json.dumps({"status": status}), encoding="utf-8")
        if (run_dir / "data" / "metrics.csv").exists() and (run_dir / "metadata" / "method.json").exists():
            write_report(run_dir)
            update_catalog(RUNS_ROOT)
    print(f"Completed. Report: {run_dir / 'report' / 'report.html'}", flush=True)
