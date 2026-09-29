"""Entraînement PPO avec métriques en direct, rapport et reprise."""

import argparse
import csv
import json
from collections import deque
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from dashboard import write_report
from experiment_catalog import update_catalog, write_run_card
from random_agent import positive_int
from rl_env import ThreesEnv
from rewards import REWARD_DESCRIPTIONS
from experiment_config import load_config, normalize_config, write_config
from run_layout import (PROJECT_ROOT, RUNS_ROOT, checkpoints_path, config_path,
                        method_path, metrics_path, model_path, report_path)
from run_provenance import snapshot_resume, snapshot_source

FIELDS = ("steps", "updates", "seconds", "steps_per_second", "train_score_mean",
          "train_episodes", "value_loss", "policy_loss", "entropy_loss", "approx_kl",
          "explained_variance", "eval_mean", "eval_median", "eval_max", "eval_games")


def evaluate_games(model: MaskablePPO, games: int, seed: int) -> list[tuple[int, int]]:
    env = ThreesEnv()
    scores = []
    try:
        for game_seed in range(seed, seed + games):
            observation, _ = env.reset(seed=game_seed)
            for _ in range(10_000):
                action, _ = model.predict(
                    observation, action_masks=env.action_masks(), deterministic=True
                )
                observation, _, done, _, info = env.step(int(action))
                if done:
                    scores.append((info["score"], max(max(row) for row in env.game.board)))
                    break
            else:
                raise RuntimeError(f"Game {game_seed} did not finish after 10,000 moves")
    finally:
        env.close()
    return scores


def evaluate(model: MaskablePPO, games: int, seed: int) -> list[int]:
    return [score for score, _ in evaluate_games(model, games, seed)]


class TrackedPPO(MaskablePPO):
    """Capture les pertes juste après chaque mise à jour des poids."""

    def train(self) -> None:
        super().train()
        callback = getattr(self, "metrics_callback", None)
        if callback is not None:
            callback.record_update(self)


class MetricsCallback(BaseCallback):
    """Journalise les mises à jour et déclenche les évaluations périodiques."""

    def __init__(self, run_dir: Path, eval_interval: int, eval_games: int, eval_seed: int):
        super().__init__()
        self.run_dir = run_dir
        self.path = metrics_path(run_dir)
        self.eval_interval = eval_interval
        self.eval_games = eval_games
        self.eval_seed = eval_seed
        self.start = perf_counter()
        self.start_steps = 0
        self.last_eval_steps = -1
        self.episode_scores = deque(maxlen=100)
        self.episode_count = 0
        self.last_eval = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8-sig", newline="") as stream:
                previous = list(csv.DictReader(stream))
            if previous:
                last = previous[-1]
                self.episode_count = int(last["train_episodes"])
                self.previous_seconds = float(last["seconds"])
                evaluations = [row for row in previous if row["eval_mean"]]
                if evaluations:
                    self.last_eval_steps = int(evaluations[-1]["steps"])
            else:
                self.previous_seconds = 0.0
        else:
            self.previous_seconds = 0.0

    def _on_training_start(self) -> None:
        self.start_steps = self.model.num_timesteps

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.episode_scores.append(info["score"])
                self.episode_count += 1
        return True

    def record_update(self, model: MaskablePPO) -> None:
        values = model.logger.name_to_value
        segment_elapsed = perf_counter() - self.start
        elapsed = self.previous_seconds + segment_elapsed
        row = {
            "steps": model.num_timesteps,
            "updates": model.num_timesteps // model.n_steps,
            "seconds": round(elapsed, 3),
            "steps_per_second": round((model.num_timesteps - self.start_steps) / max(segment_elapsed, 0.001), 1),
            "train_score_mean": round(mean(self.episode_scores), 2) if self.episode_scores else "",
            "train_episodes": self.episode_count,
            "value_loss": values.get("train/value_loss", ""),
            "policy_loss": values.get("train/policy_gradient_loss", ""),
            "entropy_loss": values.get("train/entropy_loss", ""),
            "approx_kl": values.get("train/approx_kl", ""),
            "explained_variance": values.get("train/explained_variance", ""),
            "eval_mean": "", "eval_median": "", "eval_max": "", "eval_games": "",
        }
        save_now = (self.last_eval_steps < 0 or
                    model.num_timesteps - self.last_eval_steps >= self.eval_interval or
                    model.num_timesteps >= model._total_timesteps)
        if save_now:
            scores = evaluate(model, self.eval_games, self.eval_seed)
            self.last_eval = {
                "eval_mean": round(mean(scores), 2),
                "eval_median": round(median(scores), 2),
                "eval_max": max(scores),
                "eval_games": len(scores),
            }
            row.update(self.last_eval)
            self.last_eval_steps = model.num_timesteps
            print(f"{model.num_timesteps:>8} steps | eval score {mean(scores):>7.1f} | "
                  f"value loss {row['value_loss']:.3f} | {row['steps_per_second']:.0f} steps/s", flush=True)
        with self.path.open("a", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=FIELDS).writerow(row)
        if save_now:
            checkpoint_dir = checkpoints_path(self.run_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model, checkpoint_dir / f"step_{model.num_timesteps:010d}.zip")


def save_checkpoint(model: MaskablePPO, path: Path) -> None:
    """Écrit un checkpoint sans sérialiser le callback et son environnement."""
    callback = model.metrics_callback
    model.metrics_callback = None
    try:
        model.save(path)
    finally:
        model.metrics_callback = callback


def write_method_manifest(path: Path, model: MaskablePPO, reward_id: str) -> None:
    """Fige les détails effectifs que config.json ne décrit pas."""
    policy = model.policy
    architecture = model.policy_kwargs["net_arch"]
    manifest = {
        "method": "MaskablePPO",
        "policy": "MlpPolicy",
        "observation": {
            "dimension": int(model.observation_space.shape[0]),
            "board_ranks": "16 board cells, tile ranks divided by 15",
            "next_hint": "16 indicators for possible ranks of the next tile",
            "bag_remaining": "3 remaining-card counts for tiles 1, 2, and 3, divided by 4",
        },
        "action_mask": "4 legal-action indicators supplied separately from the observation",
        "network": {
            "policy_hidden": [int(width) for width in (architecture if isinstance(architecture, list) else architecture["pi"])],
            "value_hidden": [int(width) for width in (architecture if isinstance(architecture, list) else architecture["vf"])],
            "activation": policy.activation_fn.__name__,
            "policy_outputs": int(model.action_space.n),
            "value_outputs": 1,
            "trainable_parameters": sum(parameter.numel() for parameter in policy.parameters()),
        },
        "reward_id": reward_id,
        "reward": REWARD_DESCRIPTIONS[reward_id],
        "terminal_bonus": 0,
        "terminal_penalty": 0,
        "ppo": {
            "n_steps": int(model.n_steps),
            "batch_size": int(model.batch_size),
            "n_epochs": int(model.n_epochs),
            "learning_rate": float(model.learning_rate),
            "gamma": float(model.gamma),
            "gae_lambda": float(model.gae_lambda),
            "clip_range": float(model.clip_range(1.0)),
            "clip_range_vf": None if model.clip_range_vf is None else float(model.clip_range_vf(1.0)),
            "normalize_advantage": bool(model.normalize_advantage),
            "ent_coef": float(model.ent_coef),
            "vf_coef": float(model.vf_coef),
            "max_grad_norm": float(model.max_grad_norm),
            "target_kl": None if model.target_kl is None else float(model.target_kl),
        },
        "device": str(model.device),
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_arguments() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="Entraîner PPO avec tableau de bord et reprise")
    parser.add_argument("--config", type=Path, help="configuration JSON (défaut : configs/ppo.json)")
    parser.add_argument("--steps", type=positive_int, help="pas supplémentaires pour cette invocation")
    parser.add_argument("--run-dir", type=Path, help="dossier du nouveau run")
    parser.add_argument("--resume", type=Path, help="reprendre le checkpoint d'un dossier de run")
    parser.add_argument("--seed", type=int, help="graine du nouveau run")
    parser.add_argument("--eval-games", type=positive_int, help="parties par évaluation")
    args = parser.parse_args()
    if args.resume and args.run_dir:
        parser.error("--resume et --run-dir sont incompatibles")
    return parser, args


def prepare_run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[Path, dict]:
    """Charge une configuration historique ou prépare un dossier de run neuf."""
    if args.resume:
        run_dir = args.resume
        config = load_config(config_path(run_dir))
        if args.config and load_config(args.config) != config:
            parser.error("La configuration fournie diffère de celle du checkpoint")
        if args.seed is not None and args.seed != config["seed"]:
            parser.error("La graine diffère de celle du checkpoint")
        if args.eval_games is not None and args.eval_games != config["eval_games"]:
            parser.error("Le nombre de parties d'évaluation diffère de celui du run")
    else:
        config = load_config(args.config or PROJECT_ROOT / "configs" / "ppo.json")
        if args.seed is not None:
            config["seed"] = args.seed
        if args.eval_games is not None:
            config["eval_games"] = args.eval_games
        if args.steps is not None:
            config["train_steps"] = args.steps
        config = normalize_config(config)
        run_dir = args.run_dir or RUNS_ROOT / datetime.now().strftime("%Y-%m-%d_%H%M%S_ppo")
        if run_dir.exists():
            parser.error(f"Le dossier existe déjà : {run_dir}")
        run_dir.mkdir(parents=True)
        for directory in (run_dir / "metadata", run_dir / "data", run_dir / "report",
                          run_dir / "models", run_dir / "evaluation"):
            directory.mkdir()
        write_config(config_path(run_dir), config)
        snapshot_source(run_dir)
        with metrics_path(run_dir).open("w", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=FIELDS).writeheader()
    return run_dir, config


def restore_model(run_dir: Path, config: dict, env: ThreesEnv,
                  parser: argparse.ArgumentParser) -> TrackedPPO:
    """Repart du modèle le plus avancé et coupe les métriques qui le dépassent."""
    checkpoint = model_path(run_dir)
    available = sorted(checkpoints_path(run_dir).glob("step_*.zip"))
    last_periodic = available[-1] if available else None
    if not checkpoint.is_file() and last_periodic is None:
        parser.error(f"Checkpoint introuvable dans {run_dir}")
    chosen = checkpoint if checkpoint.is_file() else last_periodic
    model = TrackedPPO.load(chosen, env=env, device="cpu")
    if last_periodic and int(last_periodic.stem.removeprefix("step_")) > model.num_timesteps:
        chosen = last_periodic
        model = TrackedPPO.load(chosen, env=env, device="cpu")
    if (model.policy_kwargs.get("net_arch") != config["net_arch"] or
            model.n_steps != config["n_steps"] or model.batch_size != config["batch_size"] or
            model.n_epochs != config["n_epochs"] or model.gamma != config["gamma"] or
            model.learning_rate != config["learning_rate"]):
        parser.error("L'architecture ou les paramètres PPO du checkpoint diffèrent de config.json")

    metrics_file = metrics_path(run_dir)
    with metrics_file.open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("steps") and
                int(row["steps"]) <= model.num_timesteps]
    with metrics_file.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Resuming from {model.num_timesteps} steps: {chosen}")
    return model


def main() -> None:
    parser, args = parse_arguments()
    run_dir, config = prepare_run(args, parser)
    torch.set_num_threads(1)
    env = ThreesEnv(config["reward_id"])
    try:
        if args.resume:
            model = restore_model(run_dir, config, env, parser)
        else:
            model = TrackedPPO(
                "MlpPolicy", env, policy_kwargs={"net_arch": config["net_arch"]},
                n_steps=config["n_steps"], batch_size=config["batch_size"],
                n_epochs=config["n_epochs"], learning_rate=config["learning_rate"],
                gamma=config["gamma"], seed=config["seed"], device="cpu", verbose=0,
            )
            write_method_manifest(method_path(run_dir), model, config["reward_id"])
            write_run_card(run_dir)

        train_model(model, run_dir, config, args)
    finally:
        env.close()


def train_model(model: TrackedPPO, run_dir: Path, config: dict,
                args: argparse.Namespace) -> None:
    """Relie le suivi, l'entraînement et les artefacts de fin de run."""
    callback = MetricsCallback(run_dir, config["eval_interval"],
                               config["eval_games"], config["eval_seed"])
    model.metrics_callback = callback
    if args.resume:
        snapshot_resume(run_dir, model.num_timesteps)
    steps = args.steps or config["train_steps"]
    print(f"Run: {run_dir.resolve()}\nDashboard: python src/dashboard.py {run_dir}", flush=True)
    checkpoint = model_path(run_dir)
    try:
        model.learn(total_timesteps=steps, callback=callback, reset_num_timesteps=not args.resume)
        model.metrics_callback = None
        save_checkpoint(model, checkpoint)
    finally:
        # Le rapport et la fiche restent consultables après Ctrl+C ou une erreur.
        try:
            write_report(run_dir)
            update_catalog(RUNS_ROOT)
        except Exception as error:
            print(f"Report or catalog update failed: {error}", flush=True)
    print(f"Checkpoint: {checkpoint}\nReport: {report_path(run_dir)}")
    if callback.last_eval:
        print(f"Latest evaluation ({config['eval_games']} games): "
              f"mean {callback.last_eval['eval_mean']}, median {callback.last_eval['eval_median']}")


if __name__ == "__main__":
    main()

