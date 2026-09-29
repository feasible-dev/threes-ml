"""Adaptateur Gymnasium du moteur Threes pour des politiques à coups masqués."""

import gymnasium as gym
import numpy as np

from observation import ACTIONS, Observation, observe
from rewards import REWARD_DESCRIPTIONS, move_reward, rank_weighted_merges
from threes import Game, move


class ThreesEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(self, reward_id: str = "log_score_gain_v1") -> None:
        super().__init__()
        if reward_id not in REWARD_DESCRIPTIONS:
            raise ValueError(f"Unknown reward_id: {reward_id}")
        self.reward_id = reward_id
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(35,), dtype=np.float32)
        self.game: Game | None = None
        self._state: Observation | None = None

    def _observation(self) -> np.ndarray:
        assert self._state is not None
        state = self._state
        return np.asarray(
            (*[rank / 15 for rank in state.board_ranks], *state.hint_mask,
             *[count / 4 for count in state.bag_remaining]),
            dtype=np.float32,
        )

    def action_masks(self) -> np.ndarray:
        assert self._state is not None
        return np.asarray(self._state.legal_actions, dtype=np.bool_)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        game_seed = seed if seed is not None else int(self.np_random.integers(0, 2**31))
        self.game = Game(seed=game_seed)
        self._state = observe(self.game)
        return self._observation(), {"score": self.game.current_score}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.game is None:
            raise RuntimeError("Appeler reset() avant step()")
        if not self.action_space.contains(action):
            raise ValueError(f"Action inconnue : {action}")
        before = self.game.current_score
        board_before = self.game.board
        moved_board = (move(board_before, ACTIONS[action])
                       if self.reward_id == "rank_weighted_merges_v1" else None)
        moved = self.game.step(ACTIONS[action])
        if moved:
            self._state = observe(self.game)
        score_gain = self.game.current_score - before
        if moved_board is not None:
            reward = float(rank_weighted_merges(board_before, moved_board)) if moved else -1.0
        else:
            reward = move_reward(self.reward_id, score_gain, moved)
        assert self._state is not None
        terminated = not any(self._state.legal_actions)
        info = {"score": self.game.current_score, "invalid_action": not moved}
        return self._observation(), reward, terminated, False, info
