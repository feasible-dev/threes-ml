"""Read-only recurrent PPO audit from a complete engine-v2 checkpoint.

The rollout is collected in a private in-memory environment. This script does
not call train(), save a model, or modify the parent run directory.
"""

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.callbacks import BaseCallback

from reproduction.environment import NativeBatch, ReproductionVecEnv, legal_masks, moved_board, shaped_reward
from reproduction.training import TrackedRecurrentPPO


class AuditCallback(BaseCallback):
    def _on_step(self):
        return True


def norm(grads):
    return math.sqrt(sum(float(torch.sum(g.detach().double() ** 2)) for g in grads if g is not None))


def alignment(left, right):
    dot = sum(float(torch.sum(a.detach().double() * b.detach().double()))
              for a, b in zip(left, right) if a is not None and b is not None)
    a, b = norm(left), norm(right)
    return dot / (a * b) if a and b else None


def gradient_audit(policy, sample, value_coefficient):
    valid = sample.mask > 0
    values, log_probs, entropy = policy.evaluate_actions(
        sample.observations, sample.actions.long().flatten(),
        sample.lstm_states, sample.episode_starts)
    advantage = sample.advantages[valid]
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    ratio = torch.exp(log_probs[valid] - sample.old_log_prob[valid])
    policy_loss = -(advantage * ratio).mean()
    value_loss = value_coefficient * torch.mean((values.flatten()[valid] - sample.returns[valid]) ** 2)
    entropy_loss = -entropy[valid].mean()
    shared = [(name, parameter) for name, parameter in policy.named_parameters()
              if name.startswith("features_extractor.") or name.startswith("lstm_actor.")]
    parameters = [parameter for _, parameter in shared]
    actor = torch.autograd.grad(policy_loss, parameters, retain_graph=True, allow_unused=True)
    critic = torch.autograd.grad(value_loss, parameters, retain_graph=True, allow_unused=True)
    result = {"policy_loss": float(policy_loss.detach()),
              "weighted_value_loss": float(value_loss.detach()),
              "actor_critic_cosine": alignment(actor, critic),
              "shared": {}}
    for prefix in ("features_extractor.", "lstm_actor."):
        indices = [i for i, (name, _) in enumerate(shared) if name.startswith(prefix)]
        result["shared"][prefix[:-1]] = {
            "actor_norm": norm([actor[i] for i in indices]),
            "weighted_critic_norm": norm([critic[i] for i in indices]),
            "cosine": alignment([actor[i] for i in indices], [critic[i] for i in indices]),
        }
    return result, (policy_loss, value_loss, entropy_loss)


def episode_summary(path):
    counts = {"episodes": 0, "natural": 0, "curriculum": 0,
              "timeouts": 0, "natural_timeouts": 0, "curriculum_timeouts": 0}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            curriculum = bool(int(row["curriculum"]))
            timeout = bool(int(row["truncated"]))
            counts["episodes"] += 1
            kind = "curriculum" if curriculum else "natural"
            counts[kind] += 1
            counts["timeouts"] += timeout
            counts[kind + "_timeouts"] += timeout
    return counts


def greedy_natural_games(policy, seed, games, environment_config, gamma, mask_actions):
    env = NativeBatch(games, terminal_rank=environment_config["terminal_rank"],
                      heuristic=environment_config["heuristic"],
                      snake_weight=environment_config["snake_weight"],
                      reward_scaler=environment_config["reward_scaler"], scaffold=0.0)
    env.reset(np.arange(games), np.arange(seed, seed + games, dtype=np.uint64))
    active = np.ones(games, dtype=bool)
    device = next(policy.parameters()).device
    hidden = torch.zeros((games, policy.lstm_actor.hidden_size), device=device)
    cell = torch.zeros_like(hidden)
    invalid = np.zeros(games, dtype=np.int64)
    returns = np.zeros(games)
    discounted_returns = np.zeros(games)
    merge_rewards = np.zeros(games)
    shaping_rewards = np.zeros(games)
    clipped_rewards = np.zeros(games, dtype=np.int64)
    rows = [None] * games
    policy.set_training_mode(False)
    with torch.no_grad():
        while active.any():
            obs = torch.as_tensor(env.observe(), device=device)
            logits, _, hidden, cell = policy.raw_forward(obs, hidden, cell)
            legal = legal_masks(env.board)
            if mask_actions:
                logits = logits.masked_fill(~torch.as_tensor(legal, device=device), -1e9)
            actions = logits.argmax(dim=-1).cpu().numpy()
            invalid += active & ~legal[np.arange(games), actions]
            merges = np.zeros(games)
            for i in np.flatnonzero(active & legal[np.arange(games), actions]):
                _, _, merges[i] = moved_board(env.board[i], int(actions[i]))
            _, rewards, done, stats = env.step(actions)
            returns[active] += rewards[active]
            discounted_returns[active] += np.power(gamma, stats[active, 3] - 1) * rewards[active]
            merge_rewards[active] += merges[active] * environment_config["reward_scaler"]
            for i in np.flatnonzero(active & legal[np.arange(games), actions]):
                shaping = (shaped_reward(env.board[i], environment_config["snake_weight"])
                           if environment_config["heuristic"] else 0.0)
                shaping_rewards[i] += shaping * environment_config["reward_scaler"]
                clipped_rewards[i] += int((merges[i] + shaping) * environment_config["reward_scaler"] > 1.0)
            for i in np.flatnonzero(done & active):
                rows[i] = {"seed": seed + int(i), "score": int(stats[i, 0]),
                           "valid_moves": int(stats[i, 2]), "attempts": int(stats[i, 3]),
                           "invalid_attempts": int(invalid[i]), "timeout": bool(stats[i, 5]),
                           "return": float(returns[i]),
                           "discounted_return": float(discounted_returns[i]),
                           "merge_reward_before_clipping": float(merge_rewards[i]),
                           "shaping_reward_before_clipping": float(shaping_rewards[i]),
                           "positive_rewards_clipped": int(clipped_rewards[i])}
                active[i] = False
                env.board[i] = 0
    scores = np.array([row["score"] for row in rows])
    observed_returns = np.array([row["return"] for row in rows])
    discounted = np.array([row["discounted_return"] for row in rows])
    return {"games": games, "seed_start": seed, "mean_score": float(scores.mean()),
            "median_score": float(np.median(scores)),
            "mean_return": float(observed_returns.mean()),
            "mean_discounted_return": float(discounted.mean()),
            "score_return_pearson": float(np.corrcoef(scores, observed_returns)[0, 1]),
            "score_discounted_return_pearson": float(np.corrcoef(scores, discounted)[0, 1]),
            "mean_merge_reward_before_clipping": float(np.mean([row["merge_reward_before_clipping"] for row in rows])),
            "mean_shaping_reward_before_clipping": float(np.mean([row["shaping_reward_before_clipping"] for row in rows])),
            "positive_rewards_clipped": sum(row["positive_rewards_clipped"] for row in rows),
            "mean_valid_moves": float(np.mean([row["valid_moves"] for row in rows])),
            "mean_attempts": float(np.mean([row["attempts"] for row in rows])),
            "mean_invalid_attempts": float(np.mean([row["invalid_attempts"] for row in rows])),
            "timeouts": sum(row["timeout"] for row in rows), "per_game": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--natural-games", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = sorted((args.run / "models" / "checkpoints").glob("step_*/complete.json"))[-1].parent
    config = json.loads((args.run / "metadata" / "config.json").read_text(encoding="utf-8"))
    state = torch.load(checkpoint / "runtime.pt", map_location="cpu", weights_only=False)
    env = ReproductionVecEnv(config["ppo"]["n_envs"], config["training"]["seed"],
                             **config["environment"])
    try:
        model = TrackedRecurrentPPO.load(checkpoint / "model.zip", env=env,
                                         device=args.device, force_reset=False)
        assert state["steps"] == model.num_timesteps
        for name, array in state["native"].items():
            getattr(env.native, name)[:] = array
        env.counter = state["counter"]
        env.returns = state["returns"].copy()
        env.invalid_count = state["invalid_count"]
        env.action_count = state["action_count"]
        recurrent = state["last_lstm_states"]
        model._last_lstm_states = RNNStates(
            tuple(x.to(model.device) for x in recurrent.pi),
            tuple(x.to(model.device) for x in recurrent.vf))
        model._last_obs = state["last_obs"].copy()
        model._last_episode_starts = state["last_episode_starts"].copy()
        _, callback = model._setup_learn(model.n_envs * model.n_steps,
                                         callback=AuditCallback(), reset_num_timesteps=False)
        before_invalid, before_actions = env.invalid_count, env.action_count
        with torch.no_grad():
            completed = model.collect_rollouts(env, callback, model.rollout_buffer, model.n_steps)
        assert completed
        buffer = model.rollout_buffer
        results = {"audited_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                   "run": str(args.run.resolve()), "checkpoint": str(checkpoint.resolve()),
                   "parent_steps": int(state["steps"]), "audit_rollout_steps": model.n_steps * model.n_envs,
                   "device": str(model.device), "source": "fresh rollout from saved in-memory state; no optimizer update",
                   "episodes": episode_summary(args.run / "data" / "episodes.csv"),
                   "sampled_invalid_rate": (env.invalid_count - before_invalid) / (env.action_count - before_actions)}
        counts = {"real": 0, "padding": 0, "resets": 0, "invalid_mass_sum": 0.0,
                  "invalid_sampled": 0, "one_legal": 0, "max_logprob_error": 0.0,
                  "sum_logprob_error": 0.0, "max_value_error": 0.0, "sum_ratio": 0.0,
                  "sum_approx_kl": 0.0, "negative_logprob_nonfinite": 0,
                  "error_above_1e_5": 0, "error_above_1e_4": 0,
                  "error_above_1e_3": 0}
        samples = []
        model.policy.set_training_mode(True)
        for sample in buffer.get(batch_size=model.batch_size):
            samples.append(sample)
            valid = sample.mask > 0
            with torch.no_grad():
                values, log_probs, _ = model.policy.evaluate_actions(
                    sample.observations, sample.actions.long().flatten(),
                    sample.lstm_states, sample.episode_starts)
                delta = log_probs[valid] - sample.old_log_prob[valid]
                ratio = torch.exp(delta)
                counts["real"] += int(valid.sum())
                counts["padding"] += int((~valid).sum())
                counts["resets"] += int((sample.episode_starts[valid] > 0).sum())
                counts["max_logprob_error"] = max(counts["max_logprob_error"], float(delta.abs().max()))
                counts["sum_logprob_error"] += float(delta.abs().sum())
                counts["error_above_1e_5"] += int((delta.abs() > 1e-5).sum())
                counts["error_above_1e_4"] += int((delta.abs() > 1e-4).sum())
                counts["error_above_1e_3"] += int((delta.abs() > 1e-3).sum())
                counts["max_value_error"] = max(counts["max_value_error"],
                                                 float((values.flatten()[valid] - sample.old_values[valid]).abs().max()))
                counts["sum_ratio"] += float(ratio.sum())
                counts["sum_approx_kl"] += float(((ratio - 1) - delta).sum())
                counts["negative_logprob_nonfinite"] += int((~torch.isfinite(log_probs)).sum())
                latent, _ = model.policy._shared_latent(sample.observations,
                                                         sample.lstm_states.pi, sample.episode_starts)
                pi, _ = model.policy.mlp_extractor(latent)
                logits = model.policy.action_net(pi)[valid]
                probabilities = torch.softmax(logits, dim=-1)
                boards = sample.observations[valid, :16].cpu().numpy().astype(np.uint8)
                legal = torch.as_tensor(legal_masks(boards), device=model.device)
                counts["one_legal"] += int((legal.sum(dim=-1) == 1).sum())
                counts["invalid_mass_sum"] += float((probabilities * ~legal).sum())
                counts["invalid_sampled"] += int((~legal.gather(1, sample.actions.long()[valid])).sum())
        n = counts["real"]
        results["reconstruction"] = {
            "real_samples": n, "padding_samples": counts["padding"],
            "episode_resets": counts["resets"], "one_legal_states": counts["one_legal"],
            "max_abs_logprob_error": counts["max_logprob_error"],
            "mean_abs_logprob_error": counts["sum_logprob_error"] / n,
            "logprob_errors_above_1e_5": counts["error_above_1e_5"],
            "logprob_errors_above_1e_4": counts["error_above_1e_4"],
            "logprob_errors_above_1e_3": counts["error_above_1e_3"],
            "max_abs_value_error": counts["max_value_error"],
            "mean_probability_ratio": counts["sum_ratio"] / n,
            "approx_kl_before_update": counts["sum_approx_kl"] / n,
            "nonfinite_logprobs_including_padding": counts["negative_logprob_nonfinite"]}
        results["action_distribution"] = {
            "mean_invalid_probability_mass": counts["invalid_mass_sum"] / n,
            "sampled_invalid_rate_from_minibatches": counts["invalid_sampled"] / n}
        all_parameters = [parameter for parameter in model.policy.parameters() if parameter.requires_grad]
        gradient_rows = []
        for sample in samples:
            gradients, losses = gradient_audit(model.policy, sample, model.vf_coef)
            policy_loss, value_loss, entropy_loss = losses
            combined = torch.autograd.grad(policy_loss + value_loss + model.ent_coef * entropy_loss,
                                           all_parameters, allow_unused=True)
            combined_norm = norm(combined)
            gradients["full_preclip_norm"] = combined_norm
            gradients["clip_limit"] = model.max_grad_norm
            gradients["implied_global_clip_multiplier"] = min(1.0, model.max_grad_norm / (combined_norm + 1e-6))
            gradient_rows.append(gradients)
        results["gradients_first_epoch_minibatches"] = gradient_rows
        results["gradient_summary"] = {
            "minibatches": len(gradient_rows),
            "median_critic_to_actor_norm_encoder": float(np.median([
                row["shared"]["features_extractor"]["weighted_critic_norm"] /
                row["shared"]["features_extractor"]["actor_norm"] for row in gradient_rows])),
            "median_critic_to_actor_norm_lstm": float(np.median([
                row["shared"]["lstm_actor"]["weighted_critic_norm"] /
                row["shared"]["lstm_actor"]["actor_norm"] for row in gradient_rows])),
            "median_global_clip_multiplier": float(np.median([
                row["implied_global_clip_multiplier"] for row in gradient_rows]))}
        if args.natural_games:
            seed = config["evaluation"]["seed"] + config["evaluation"]["games"]
            unmasked = greedy_natural_games(model.policy, seed, args.natural_games,
                                            config["environment"], model.gamma, False)
            masked = greedy_natural_games(model.policy, seed, args.natural_games,
                                          config["environment"], model.gamma, True)
            scores_unmasked = np.array([row["score"] for row in unmasked["per_game"]])
            scores_masked = np.array([row["score"] for row in masked["per_game"]])
            results["greedy_natural_development_pair"] = {
                "unmasked": unmasked, "masked": masked,
                "mean_paired_score_difference_masked_minus_unmasked":
                    float((scores_masked - scores_unmasked).mean())}
        output = json.dumps(results, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output + "\n", encoding="utf-8")
        print(output)
    finally:
        env.close()


if __name__ == "__main__":
    main()
