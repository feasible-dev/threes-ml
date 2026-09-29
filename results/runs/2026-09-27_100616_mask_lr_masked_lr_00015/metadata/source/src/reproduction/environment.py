"""Numba-compiled batched Threes rules and published reward components.

Independent implementation from the documented rules and upstream threes.h audit.
Uses ranks 0,1,2,3,4,... for values 0,1,2,3,6,... . No GUI or global RNG.
"""

import csv
from pathlib import Path

import gymnasium as gym
import numpy as np
from numba import njit
from stable_baselines3.common.vec_env import VecEnv


SCORES = np.asarray([0, 0, 0] + [3 ** (r - 2) for r in range(3, 19)], dtype=np.int64)
ENGINE_REVISION = 2
POW15 = np.asarray([0, 1, 2, 5, 8, 11, 14, 18, 22, 27, 31, 36, 41, 46, 52, 57, 64, 69, 75, 81])
# Metadata: next actual, preview, moves, ticks, bonus phase, lifetime rank,
# current rank, curriculum flag, cached episode time limit.


@njit(cache=True)
def random_below(rng, i, limit):
    x = rng[i]
    x ^= x >> np.uint64(12)
    x ^= x << np.uint64(25)
    x ^= x >> np.uint64(27)
    rng[i] = x
    return int((x * np.uint64(2685821657736338717)) % np.uint64(limit))


@njit(cache=True)
def draw(bag, meta, rng, i):
    if meta[i, 2] % 21 == meta[i, 4] and meta[i, 6] >= 7:
        count = meta[i, 6] - 6
        # Match C's random-call order, including rand()%1 for a three-tile pool.
        first = 4 if count <= 2 else 4 + random_below(rng, i, count - 2)
        meta[i, 0] = first if count == 1 else first + random_below(rng, i, min(count, 3))
        meta[i, 1] = first
    else:
        pick = random_below(rng, i, int(bag[i].sum()))
        tile = 0
        while pick >= bag[i, tile]:
            pick -= bag[i, tile]
            tile += 1
        bag[i, tile] -= 1
        meta[i, 0] = tile + 1
        meta[i, 1] = tile + 1
        if bag[i].sum() == 0:
            bag[i, :] = 4


@njit(cache=True)
def reset_rows(board, bag, meta, rng, ids, seeds, scaffold):
    for k in range(len(ids)):
        i = ids[k]
        # SplitMix avalanche avoids weak initial states for consecutive seeds.
        z = seeds[k] + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        rng[i] = (z ^ (z >> np.uint64(31))) | np.uint64(1)
        lifetime = meta[i, 5]
        meta[i, :] = 0
        meta[i, 5] = lifetime
        meta[i, 8] = 1000
        bag[i, :] = 4
        board[i, :] = 0
        use_curriculum = lifetime >= 6 and random_below(rng, i, 1000000) < int(scaffold * 1000000)
        count = 3 + random_below(rng, i, 4) if use_curriculum else 9
        for _ in range(count):
            free = np.flatnonzero(board[i] == 0)
            position = free[random_below(rng, i, len(free))]
            if use_curriculum:
                low = max(3, lifetime - 2)
                tile = low + random_below(rng, i, min(16, lifetime) - low + 1)
            else:
                draw(bag, meta, rng, i)
                tile = meta[i, 0]
            board[i, position] = tile
        meta[i, 6] = int(board[i].max())
        meta[i, 7] = int(use_curriculum)
        draw(bag, meta, rng, i)


@njit(cache=True)
def moved_board(board, action):
    result = board.copy()
    edges = np.empty(4, np.int64)
    changed = 0
    reward = 0.0
    for line in range(4):
        positions = np.empty(4, np.int64)
        for j in range(4):
            if action == 0:
                positions[j] = j * 4 + line
            elif action == 1:
                positions[j] = (3 - j) * 4 + line
            elif action == 2:
                positions[j] = line * 4 + j
            else:
                positions[j] = line * 4 + 3 - j
        values = board[positions].copy()
        did_move = False
        for j in range(1, 4):
            left, right = values[j - 1], values[j]
            if right >= 3 and left == right:
                reward += float(right) * 0.0625
                values[j - 1], values[j] = right + 1, 0
                did_move = True
            elif (left == 1 and right == 2) or (left == 2 and right == 1):
                reward += 3 * 0.0625
                values[j - 1], values[j] = 3, 0
                did_move = True
            elif left == 0 and right != 0:
                values[j - 1], values[j] = right, 0
                did_move = True
        if did_move:
            result[positions] = values
            edges[changed] = positions[3]
            changed += 1
    return result, edges[:changed], reward


@njit(cache=True)
def legal_masks(board):
    masks = np.zeros((len(board), 4), np.bool_)
    for i in range(len(board)):
        for action in range(4):
            _, edges, _ = moved_board(board[i], action)
            masks[i, action] = len(edges) > 0
    return masks


@njit(cache=True)
def shaped_reward(board, snake_weight):
    maximum = int(board.max())
    # The C source allows the two highest tiles to have the same rank.
    # A distinct-second-largest calculation omits its corner bonus.
    second = int(np.sort(board)[-2])
    state = 0.01 if np.all(board[:4] != 0) else 0.0
    if board[0] != maximum:
        return state
    if board[1] == second and maximum > 4:
        state += 0.01
    monotonicity = float(POW15[maximum])
    evidence = 0
    for row in range(2):
        row_min = 32
        next_max = 0
        for col in range(4):
            value = int(board[row * 4 + col])
            below = int(board[(row + 1) * 4 + col])
            if col < 3:
                right = int(board[row * 4 + col + 1])
                if value and right:
                    if row == 0 and value > right:
                        monotonicity += POW15[right]
                        evidence += 1
                    elif row == 1 and value < right:
                        monotonicity += POW15[value]
            if value:
                row_min = min(row_min, value)
            next_max = max(next_max, below)
            if value and below and value > below:
                monotonicity += below
        if row_min < 20 and next_max > 0 and row_min > next_max:
            monotonicity += 4 * POW15[row_min]
            if row == 0:
                evidence += 1
    snake = float(board[7]) ** 2 if evidence >= 4 and board[7] == board[4:].max() else 0.0
    return state + 0.00003 * monotonicity + snake_weight * snake


@njit(cache=True)
def step_rows(board, bag, meta, rng, actions, heuristic, snake_weight, terminal_rank, reward_scaler):
    rewards = np.zeros(len(board), np.float32)
    done = np.zeros(len(board), np.bool_)
    stats = np.zeros((len(board), 6), np.int64)
    for i in range(len(board)):
        moved, edges, reward = moved_board(board[i], actions[i])
        meta[i, 3] += 1
        if len(edges):
            board[i] = moved
            if meta[i, 2] % 21 == 0:
                meta[i, 4] = random_below(rng, i, 21)
            meta[i, 2] += 1
            # C updates the merge frontier before drawing, then scans the board
            # after spawning. Keep that ordering even for injected test states.
            meta[i, 6] = max(meta[i, 6], int(board[i].max()))
            board[i, edges[random_below(rng, i, len(edges))]] = meta[i, 0]
            draw(bag, meta, rng, i)
            meta[i, 6] = int(board[i].max())
            if heuristic:
                reward += shaped_reward(board[i], snake_weight)
            reward *= reward_scaler
            score = int(SCORES[board[i]].sum())
            meta[i, 8] = max(1000 * max(1, meta[i, 5] - 8), score // 4)
        else:
            reward = -0.05
        score = int(SCORES[board[i]].sum())
        blocked = True
        for a in range(4):
            _, possible, _ = moved_board(board[i], a)
            if len(possible):
                blocked = False
                break
        reached = meta[i, 6] >= terminal_rank
        timeout = meta[i, 3] >= meta[i, 8]
        done[i] = blocked or reached or timeout
        if blocked:
            reward = -1.0
        if done[i] and not meta[i, 7]:
            meta[i, 5] = max(meta[i, 5], meta[i, 6])
        rewards[i] = min(1.0, max(-1.0, reward))
        stats[i] = np.array([score, meta[i, 6], meta[i, 2], meta[i, 3],
                             int(not len(edges)), int(timeout and not blocked and not reached)])
    return rewards, done, stats


class NativeBatch:
    def __init__(self, count, *, heuristic=False, snake_weight=0.0, scaffold=0.0,
                 terminal_rank=16, reward_scaler=1.0):
        self.board = np.zeros((count, 16), np.uint8)
        self.bag = np.zeros((count, 3), np.int64)
        self.meta = np.zeros((count, 9), np.int64)
        self.rng = np.zeros(count, np.uint64)
        self.seeds = np.zeros(count, np.uint64)
        self.heuristic, self.snake_weight = heuristic, snake_weight
        self.scaffold, self.terminal_rank = scaffold, terminal_rank
        self.reward_scaler = reward_scaler

    def reset(self, ids, seeds):
        ids = np.asarray(ids, dtype=np.int64)
        seeds = np.asarray(seeds, dtype=np.uint64)
        self.seeds[ids] = seeds
        reset_rows(self.board, self.bag, self.meta, self.rng, ids, seeds, self.scaffold)
        return self.observe()

    def observe(self):
        return np.concatenate((self.board, self.meta[:, 1:2], self.bag,
                               self.bag.sum(1, keepdims=True)), axis=1).astype(np.float32)

    def step(self, actions):
        rewards, done, stats = step_rows(self.board, self.bag, self.meta, self.rng,
                                        np.asarray(actions, dtype=np.int64), self.heuristic,
                                        self.snake_weight, self.terminal_rank, self.reward_scaler)
        return self.observe(), rewards, done, stats


class ReproductionVecEnv(VecEnv):
    """Native batched simulation with disjoint training seed namespace and episode log."""
    def __init__(self, count, run_seed, episode_path=None, *, bootstrap_timeouts=False, **rules):
        self.native = NativeBatch(count, **rules)
        self.bootstrap_timeouts = bootstrap_timeouts
        self.run_seed = run_seed
        if not 0 <= run_seed < 2**31:
            raise ValueError("Run seed must fit in 31 bits")
        self.counter = 0
        self.returns = np.zeros(count)
        self.invalid_count = 0
        self.action_count = 0
        self.episode_path = Path(episode_path) if episode_path else None
        self.log = None
        self.writer = None
        if self.episode_path:
            exists = self.episode_path.exists()
            self.log = self.episode_path.open("a", newline="", encoding="utf-8")
            self.writer = csv.writer(self.log)
            if not exists:
                self.writer.writerow(("episode_seed", "score", "max_tile", "moves", "actions",
                                      "return", "curriculum", "truncated"))
        super().__init__(count, gym.spaces.Box(0, 17, (21,), dtype=np.float32), gym.spaces.Discrete(4))

    def next_seeds(self, count):
        if self.counter + count >= 2**32:
            raise RuntimeError("Training episode seed namespace exhausted")
        seeds = np.arange(self.counter, self.counter + count, dtype=np.uint64)
        seeds |= np.uint64(2**63 | (self.run_seed << 32))
        self.counter += count
        return seeds

    def reset(self):
        self.returns[:] = 0
        self._reset_seeds()
        return self.native.reset(np.arange(self.num_envs), self.next_seeds(self.num_envs))

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        obs, rewards, dones, stats = self.native.step(self.actions)
        self.returns += rewards
        self.invalid_count += int(stats[:, 4].sum())
        self.action_count += self.num_envs
        infos = [{} for _ in range(self.num_envs)]
        ended = np.flatnonzero(dones)
        for i in ended:
            score, rank, moves, ticks, _, timeout = stats[i]
            tile = 3 * 2 ** (int(rank) - 3) if rank >= 3 else int(rank)
            infos[i] = {"terminal_observation": obs[i].copy(),
                        "TimeLimit.truncated": bool(timeout and self.bootstrap_timeouts),
                        "score": int(score), "max_tile": tile,
                        "episode": {"r": float(self.returns[i]), "l": int(ticks), "t": 0.0},
                        "curriculum": bool(self.native.meta[i, 7])}
            if self.writer:
                self.writer.writerow((int(self.native.seeds[i]), int(score), tile, int(moves),
                                      int(ticks), float(self.returns[i]),
                                      int(self.native.meta[i, 7]), int(timeout)))
            self.returns[i] = 0
        if len(ended):
            obs = self.native.reset(ended, self.next_seeds(len(ended)))
            if self.log:
                self.log.flush()
        return obs, rewards, dones, infos

    def close(self):
        if self.log:
            self.log.close()

    def get_attr(self, attr_name, indices=None):
        # VecEnv asks for render_mode at construction.
        value = None if attr_name == "render_mode" else getattr(self, attr_name)
        return [value for _ in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError("Set batched parameters at construction")

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError("Use the batched environment directly")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]
