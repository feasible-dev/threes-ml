"""Optional largest-tile edge/corner reward for an isolated recurrent fork."""

import numpy as np

from reproduction.environment import NativeBatch, ReproductionVecEnv


CORNER_WEIGHT = 0.10
SIDE_FRACTION = 0.25
MIN_RANK = 5  # 12.
DISCOUNT = 0.999
CORNER_INDEX = (0, 3, 12, 15)
SIDE_INDEX = (1, 2, 4, 7, 8, 11, 13, 14)


def corner_potential(boards):
    """Largest tile >=12: corner 1, side 0.25, interior 0."""
    boards = np.asarray(boards)
    maxima = boards.max(axis=1)
    eligible = maxima >= MIN_RANK
    in_corner = np.any(boards[:, CORNER_INDEX] == maxima[:, None], axis=1)
    on_side = np.any(boards[:, SIDE_INDEX] == maxima[:, None], axis=1)
    return np.where(eligible & in_corner, 1.0,
                    np.where(eligible & on_side, SIDE_FRACTION, 0.0)).astype(np.float32)


def corner_delta(before, after, done, weight=CORNER_WEIGHT):
    """Potential difference; terminal states have zero continuation potential."""
    previous = corner_potential(before)
    following = corner_potential(after)
    following[np.asarray(done, dtype=bool)] = 0
    return (weight * (DISCOUNT * following - previous)).astype(np.float32)


class CornerNativeBatch(NativeBatch):
    def step(self, actions):
        before = self.board.copy()
        observations, rewards, done, stats = super().step(actions)
        rewards = np.clip(rewards + corner_delta(before, self.board, done), -1, 1).astype(np.float32)
        return observations, rewards, done, stats


class CornerRewardVecEnv(ReproductionVecEnv):
    def __init__(self, count, run_seed, episode_path=None, *, bootstrap_timeouts=False, **rules):
        super().__init__(count, run_seed, episode_path,
                         bootstrap_timeouts=bootstrap_timeouts, **rules)
        # ReproductionVecEnv has not reset or stepped the native batch yet.
        self.native = CornerNativeBatch(count, **rules)
