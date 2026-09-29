"""Batched recurrent evaluation on explicit seeds, with no curriculum."""

import csv
from pathlib import Path

import numpy as np
import torch

from .environment import NativeBatch


def evaluate(policy, games, seed, *, batch_size=64, terminal_rank=16, output=None):
    if seed < 0 or seed + games >= 2**63:
        raise ValueError("Evaluation uses seeds below 2**63; training occupies the upper half")
    count = min(batch_size, games)
    env = NativeBatch(count, terminal_rank=terminal_rank)
    env.reset(np.arange(count), np.arange(seed, seed + count, dtype=np.uint64))
    next_seed = seed + count
    active = np.ones(count, np.bool_)
    device = next(policy.parameters()).device
    hidden = torch.zeros((count, policy.lstm_actor.hidden_size), device=device)
    cell = torch.zeros_like(hidden)
    results = []
    was_training = policy.training
    policy.set_training_mode(False)
    try:
        with torch.no_grad():
            while active.any():
                obs = torch.as_tensor(env.observe(), device=device)
                logits, _, hidden, cell = policy.raw_forward(obs, hidden, cell)
                masks = env_masks(env)
                actions = logits.masked_fill(~torch.as_tensor(masks, device=device), -1e9).argmax(1)
                _, _, done, stats = env.step(actions.cpu().numpy())
                done |= stats[:, 3] >= 20000
                for i in np.flatnonzero(done & active):
                    score, rank, moves, ticks, _, timeout = stats[i]
                    results.append({"seed": int(env.seeds[i]), "score": int(score),
                                    "max_tile": 3 * 2 ** (int(rank) - 3) if rank >= 3 else int(rank),
                                    "moves": int(moves), "actions": int(ticks),
                                    "truncated": bool(timeout or ticks >= 20000)})
                    hidden[i] = 0
                    cell[i] = 0
                    if next_seed < seed + games:
                        env.reset([i], [next_seed])
                        next_seed += 1
                    else:
                        active[i] = False
                        env.board[i] = 0
    finally:
        policy.set_training_mode(was_training)
    results.sort(key=lambda row: row["seed"])
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    scores = np.asarray([row["score"] for row in results])
    tiles = np.asarray([row["max_tile"] for row in results])
    return {"eval_mean": float(scores.mean()), "eval_median": float(np.median(scores)),
            "eval_max": int(scores.max()), "eval_games": games,
            "eval_384": float((tiles >= 384).mean()), "eval_3072": float((tiles >= 3072).mean()),
            "eval_6144": float((tiles >= 6144).mean()),
            "eval_truncated": sum(row["truncated"] for row in results)}


def env_masks(env):
    from .environment import legal_masks
    return legal_masks(env.board)
