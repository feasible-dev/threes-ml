"""Legal-action distribution for the recurrent masking experiment.

Legality is recomputed from each stored pre-action observation, so collection
and PPO replay use the same deterministic board rule. Padded zero observations
receive a finite dummy distribution; the recurrent buffer masks their losses.
"""

import numpy as np
import torch
from sb3_contrib.common.recurrent.type_aliases import RNNStates

from reproduction.environment import legal_masks
from reproduction.policy import PublishedPolicy


def observation_action_masks(observations, *, allow_padding=False):
    """Return pre-action legality, with all-true dummy masks on padding only."""
    if isinstance(observations, torch.Tensor):
        array = observations.detach().to("cpu").numpy()
    else:
        array = np.asarray(observations)
    if array.ndim != 2 or array.shape[1] != 21:
        raise ValueError("Expected a batch of 21-feature observations")
    boards = array[:, :16]
    if not np.all(np.isfinite(boards)) or np.any(boards < 0) or np.any(boards > 17):
        raise ValueError("Invalid tile ranks in the action-mask observation")
    masks = legal_masks(boards.astype(np.uint8))
    without_moves = ~masks.any(axis=1)
    if np.any(without_moves):
        padded = np.all(array[without_moves] == 0, axis=1)
        if not allow_padding or not np.all(padded):
            raise ValueError("Encountered a real state with no legal actions; reset terminal states first")
        masks[without_moves] = True
    return masks


class MaskedPublishedPolicy(PublishedPolicy):
    """PublishedPolicy with a legal categorical distribution during PPO."""

    def _masked_distribution(self, actor_latent, observations, *, allow_padding):
        masks = torch.as_tensor(observation_action_masks(observations, allow_padding=allow_padding),
                                device=actor_latent.device)
        logits = self.action_net(actor_latent).masked_fill(~masks, -1e9)
        if allow_padding:
            padded = torch.all(observations == 0, dim=1)
            logits = torch.where(padded[:, None], torch.zeros_like(logits), logits)
        return self.action_dist.proba_distribution(action_logits=logits)

    def forward(self, obs, lstm_states, episode_starts, deterministic=False):
        latent, state = self._shared_latent(obs, lstm_states.pi, episode_starts)
        pi, vf = self.mlp_extractor(latent)
        distribution = self._masked_distribution(pi, obs, allow_padding=False)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions, self.value_net(vf), distribution.log_prob(actions), RNNStates(state, state)

    def evaluate_actions(self, obs, actions, lstm_states, episode_starts):
        latent, _ = self._shared_latent(obs, lstm_states.pi, episode_starts)
        pi, vf = self.mlp_extractor(latent)
        distribution = self._masked_distribution(pi, obs, allow_padding=True)
        return self.value_net(vf), distribution.log_prob(actions), distribution.entropy()

    def get_distribution(self, obs, lstm_states, episode_starts):
        latent, state = self._shared_latent(obs, lstm_states, episode_starts)
        pi = self.mlp_extractor.forward_actor(latent)
        return self._masked_distribution(pi, obs, allow_padding=False), state
