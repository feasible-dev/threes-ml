"""CPU-only in-memory attempt and reward probe from a fixed checkpoint."""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sb3_contrib.common.recurrent.type_aliases import RNNStates

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from reproduction.environment import ReproductionVecEnv, legal_masks, moved_board, shaped_reward
from reproduction.training import TrackedRecurrentPPO


def bucket(length):
    if length < 4:
        return str(length)
    if length < 8:
        return "4-7"
    if length < 16:
        return "8-15"
    return "16+"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--checkpoint-steps", type=int, default=74_235_904)
    parser.add_argument("--environments", type=int, default=16)
    parser.add_argument("--time-steps", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = args.run / "models" / "checkpoints" / f"step_{args.checkpoint_steps:012d}"
    config = json.loads((args.run / "metadata" / "config.json").read_text(encoding="utf-8"))
    state = torch.load(checkpoint / "runtime.pt", map_location="cpu", weights_only=False)
    assert state["steps"] == args.checkpoint_steps
    assert 0 < args.environments <= len(state["native"]["board"])
    model = TrackedRecurrentPPO.load(checkpoint / "model.zip", device="cpu")
    model.policy.set_training_mode(False)
    env = ReproductionVecEnv(args.environments, config["training"]["seed"],
                             **config["environment"])
    try:
        for name, array in state["native"].items():
            getattr(env.native, name)[:] = array[:args.environments]
        env.counter = state["counter"]
        obs = state["last_obs"][:args.environments].copy()
        starts = state["last_episode_starts"][:args.environments].copy()
        old_states = state["last_lstm_states"]
        states = RNNStates(
            tuple(x[:, :args.environments].contiguous() for x in old_states.pi),
            tuple(x[:, :args.environments].contiguous() for x in old_states.vf))
        torch.set_rng_state(state["torch_rng"])
        current_streak = np.zeros(args.environments, dtype=np.int64)
        current_kind = np.zeros(args.environments, dtype=np.int64)
        streak_start = np.full(args.environments, -1, dtype=np.int64)
        streak_legal = np.zeros((args.environments, 4), dtype=bool)
        streak_actions = np.zeros((args.environments, 4), dtype=np.int64)
        streak_first_probs = np.zeros((args.environments, 4))
        streak_last_probs = np.zeros((args.environments, 4))
        streaks = {"natural": [], "curriculum": []}
        completed_streaks = {"natural": [], "curriculum": []}
        right_censored = {"natural": [], "curriculum": []}
        longest_segment = {"natural": None, "curriculum": None}
        totals = {kind: Counter() for kind in streaks}
        scale = config["environment"]["reward_scaler"]
        heuristic = config["environment"]["heuristic"]
        snake = config["environment"]["snake_weight"]
        def finish_streak(slot, last_step, ending):
            if not current_streak[slot]:
                return
            label = "curriculum" if current_kind[slot] else "natural"
            length = int(current_streak[slot])
            streaks[label].append(length)
            if ending == "probe_end":
                right_censored[label].append(length)
            else:
                completed_streaks[label].append(length)
            detail = {"observed_length": length, "slot": int(slot),
                      "first_observed_time_step": int(streak_start[slot]),
                      "last_observed_time_step": int(last_step),
                      "left_censored_by_checkpoint": bool(streak_start[slot] == 0),
                      "right_censored_by_probe_end": ending == "probe_end",
                      "ending": ending,
                      "legal_actions": streak_legal[slot].tolist(),
                      "invalid_action_counts": streak_actions[slot].tolist(),
                      "first_action_probabilities": streak_first_probs[slot].tolist(),
                      "last_action_probabilities": streak_last_probs[slot].tolist()}
            if longest_segment[label] is None or length > longest_segment[label]["observed_length"]:
                longest_segment[label] = detail
            current_streak[slot] = 0
            streak_actions[slot] = 0

        for time_step in range(args.time_steps):
            board = env.native.board.copy()
            kinds = env.native.meta[:, 7].copy()
            legal = legal_masks(board)
            with torch.no_grad():
                latent, next_state = model.policy._shared_latent(
                    torch.as_tensor(obs), states.pi, torch.as_tensor(starts, dtype=torch.float32))
                actor_latent, _ = model.policy.mlp_extractor(latent)
                distribution = model.policy._get_action_dist_from_latent(actor_latent)
                probabilities = torch.softmax(model.policy.action_net(actor_latent), dim=-1).numpy()
                actions = distribution.get_actions().numpy()
                states = RNNStates(next_state, next_state)
            selected_legal = legal[np.arange(args.environments), actions]
            merges = np.zeros(args.environments)
            for i in np.flatnonzero(selected_legal):
                _, _, merges[i] = moved_board(board[i], int(actions[i]))
            new_obs, rewards, done, stats = env.native.step(actions)
            np.testing.assert_array_equal(stats[:, 4].astype(bool), ~selected_legal)
            post_legal = legal_masks(env.native.board)
            for i in range(args.environments):
                kind = "curriculum" if kinds[i] else "natural"
                count = totals[kind]
                count["attempts"] += 1
                count["invalid_probability_mass_sum"] += float(probabilities[i, ~legal[i]].sum())
                count["one_legal_states"] += int(legal[i].sum() == 1)
                count["realized_reward"] += float(rewards[i])
                count["invalid"] += int(not selected_legal[i])
                count["valid"] += int(selected_legal[i])
                count["completed"] += int(done[i])
                count["timeouts"] += int(stats[i, 5])
                if selected_legal[i]:
                    shaping = shaped_reward(env.native.board[i], snake) if heuristic else 0.0
                    count["merge_reward_before_clipping"] += float(merges[i] * scale)
                    count["shaping_reward_before_clipping"] += float(shaping * scale)
                    count["positive_reward_clipped"] += int((merges[i] + shaping) * scale > 1.0
                                                          and post_legal[i].any())
                if not selected_legal[i]:
                    if current_streak[i] == 0:
                        current_kind[i] = kinds[i]
                        streak_start[i] = time_step
                        streak_legal[i] = legal[i]
                        streak_first_probs[i] = probabilities[i]
                    current_streak[i] += 1
                    streak_actions[i, actions[i]] += 1
                    streak_last_probs[i] = probabilities[i]
                if selected_legal[i] or done[i]:
                    finish_streak(i, time_step if not selected_legal[i] else time_step - 1,
                                  "episode_end" if done[i] else "valid_action")
            ended = np.flatnonzero(done)
            if len(ended):
                new_obs = env.native.reset(ended, env.next_seeds(len(ended)))
            obs, starts = new_obs, done
        for i, length in enumerate(current_streak):
            if length:
                finish_streak(i, args.time_steps - 1, "probe_end")
        output = {"probed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                  "checkpoint": str(checkpoint.resolve()), "parent_steps": args.checkpoint_steps,
                  "device": "cpu", "environments": args.environments,
                  "time_steps_per_environment": args.time_steps,
                  "transitions": args.environments * args.time_steps,
                  "sampling": "fresh stochastic actions from the first checkpointed environments; in-memory branch",
                  "groups": {}}
        for kind in streaks:
            count = totals[kind]
            lengths = streaks[kind]
            output["groups"][kind] = {
                **dict(count), "invalid_fraction": count["invalid"] / count["attempts"],
                "mean_invalid_probability_mass": count["invalid_probability_mass_sum"] / count["attempts"],
                "mean_reward_per_attempt": count["realized_reward"] / count["attempts"],
                "invalid_streaks": len(lengths),
                "max_invalid_streak": max(lengths, default=0),
                "mean_invalid_streak": float(np.mean(lengths)) if lengths else 0.0,
                "completed_streaks": len(completed_streaks[kind]),
                "max_completed_streak": max(completed_streaks[kind], default=0),
                "right_censored_streaks": len(right_censored[kind]),
                "longest_observed_segment": longest_segment[kind],
                "streak_histogram": dict(Counter(bucket(length) for length in lengths))}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Probed {output['transitions']:,} CPU transitions from {checkpoint.name}.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
