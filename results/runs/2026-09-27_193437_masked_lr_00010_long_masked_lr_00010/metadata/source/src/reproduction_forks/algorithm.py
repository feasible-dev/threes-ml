"""Fork-only PPO plumbing that records pre-action legality for audit."""

import csv
from pathlib import Path
from time import perf_counter

import numpy as np
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer

from reproduction.training import TrackedRecurrentPPO
from .masking import MaskedPublishedPolicy, observation_action_masks


class MaskRecordingBuffer(RecurrentRolloutBuffer):
    def reset(self):
        super().reset()
        self.action_masks = np.zeros((self.buffer_size, self.n_envs, 4), dtype=np.bool_)

    def add(self, obs, *args, **kwargs):
        self.action_masks[self.pos] = observation_action_masks(obs)
        super().add(obs, *args, **kwargs)


class ForkRecurrentPPO(TrackedRecurrentPPO):
    """The parent optimizer and method, with a fork-local mask audit buffer."""

    def _setup_model(self):
        super()._setup_model()
        lstm = self.policy.lstm_actor
        shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)
        self.rollout_buffer = MaskRecordingBuffer(
            self.n_steps, self.observation_space, self.action_space, shape,
            self.device, gamma=self.gamma, gae_lambda=self.gae_lambda,
            n_envs=self.n_envs)

    def train(self):
        if not self.rollout_buffer.full:
            raise RuntimeError("Fork PPO requires a complete fresh rollout")
        observations = self.rollout_buffer.observations.reshape(-1, 21)
        recorded = self.rollout_buffer.action_masks.reshape(-1, 4)
        if not np.array_equal(recorded, observation_action_masks(observations)):
            raise RuntimeError("Stored pre-action masks disagree with PPO observations")
        if getattr(self, "fork_mask_training", False) != isinstance(self.policy, MaskedPublishedPolicy):
            raise RuntimeError("Fork mask setting and policy class disagree")
        selected_legal = np.take_along_axis(self.rollout_buffer.action_masks,
                                             self.rollout_buffer.actions.astype(np.int64), axis=2)
        actual_epochs_before = self._n_updates
        started = perf_counter()
        super().train()
        diagnostics_path = getattr(self, "fork_diagnostics_path", None)
        if diagnostics_path is not None:
            diagnostics_path = Path(diagnostics_path)
            row = {"steps": self.num_timesteps,
                   "attempted_actions": int(selected_legal.size),
                   "valid_moves": int(selected_legal.sum()),
                   "invalid_attempts": int(selected_legal.size - selected_legal.sum()),
                   "one_legal_states": int((self.rollout_buffer.action_masks.sum(axis=2) == 1).sum()),
                   "actual_ppo_epochs": self._n_updates - actual_epochs_before,
                   "configured_ppo_epochs": self.n_epochs,
                   "approx_kl": self.logger.name_to_value.get("train/approx_kl"),
                   "clip_fraction": self.logger.name_to_value.get("train/clip_fraction"),
                   "update_seconds": round(perf_counter() - started, 6)}
            exists = diagnostics_path.exists()
            with diagnostics_path.open("a", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
