"""Trainable reconstruction of the published ONNX computation graph.

Both actor and critic backpropagate through the shared encoder and LSTM.
SB3's default shared-LSTM policy detaches the critic, so we override that path.
"""

import numpy as np
import torch
from torch import nn
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PublishedEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space, hidden_size=1024):
        super().__init__(observation_space, features_dim=hidden_size)
        self.value_embed = nn.Embedding(18, 8)
        self.position_embed = nn.Parameter(torch.empty(17, 8))
        nn.init.normal_(self.position_embed, std=0.02)
        self.extra_encoder = nn.Sequential(nn.Linear(4, 32), nn.GELU())
        self.encoder = nn.Sequential(
            nn.Flatten(), nn.Linear(168, 2 * hidden_size), nn.GELU(),
            nn.Linear(2 * hidden_size, hidden_size), nn.GELU(),
            nn.Linear(hidden_size, hidden_size), nn.GELU(),
        )

    def forward(self, observations):
        tiles = self.value_embed(observations[:, :17].long()) + self.position_embed
        extra = self.extra_encoder(observations[:, 17:].float())
        return self.encoder(torch.cat((tiles.flatten(1), extra), dim=1))


class PublishedPolicy(RecurrentActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, hidden_size=1024, **kwargs):
        kwargs.update(
            features_extractor_class=PublishedEncoder,
            features_extractor_kwargs={"hidden_size": hidden_size},
            net_arch={"pi": [hidden_size], "vf": [hidden_size]},
            activation_fn=nn.GELU, ortho_init=False,
            lstm_hidden_size=hidden_size, shared_lstm=True, enable_critic_lstm=False,
        )
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _shared_latent(self, obs, state, starts):
        features = self.extract_features(obs)
        return self._process_sequence(features, state, starts, self.lstm_actor)

    def forward(self, obs, lstm_states, episode_starts, deterministic=False):
        latent, state = self._shared_latent(obs, lstm_states.pi, episode_starts)
        pi, vf = self.mlp_extractor(latent)
        distribution = self._get_action_dist_from_latent(pi)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions, self.value_net(vf), distribution.log_prob(actions), RNNStates(state, state)

    def evaluate_actions(self, obs, actions, lstm_states, episode_starts):
        latent, _ = self._shared_latent(obs, lstm_states.pi, episode_starts)
        pi, vf = self.mlp_extractor(latent)
        distribution = self._get_action_dist_from_latent(pi)
        return self.value_net(vf), distribution.log_prob(actions), distribution.entropy()

    def predict_values(self, obs, lstm_states, episode_starts):
        latent, _ = self._shared_latent(obs, lstm_states, episode_starts)
        return self.value_net(self.mlp_extractor.forward_critic(latent))

    def raw_forward(self, observations, hidden, cell):
        """One step with the ONNX interface, for equivalence checks and batched eval."""
        encoded = self.features_extractor(observations)
        latent, (next_h, next_c) = self.lstm_actor(encoded.unsqueeze(0),
                                                (hidden.unsqueeze(0), cell.unsqueeze(0)))
        pi, vf = self.mlp_extractor(latent[0])
        return self.action_net(pi), self.value_net(vf), next_h[0], next_c[0]

    def import_onnx(self, path):
        """Import published weights only for an explicitly requested validation run."""
        import onnx
        from onnx import numpy_helper

        arrays = {x.name: numpy_helper.to_array(x) for x in onnx.load(str(path)).graph.initializer}
        mapping = {
            "value_embed": self.features_extractor.value_embed,
            "extra_encoder.0": self.features_extractor.extra_encoder[0],
            "encoder.1": self.features_extractor.encoder[1],
            "encoder.3": self.features_extractor.encoder[3],
            "encoder.5": self.features_extractor.encoder[5],
            "decoder.0": self.mlp_extractor.policy_net[0],
            "decoder.2": self.action_net,
            "value.0": self.mlp_extractor.value_net[0],
            "value.2": self.value_net,
        }
        with torch.no_grad():
            for name, layer in mapping.items():
                for key, parameter in layer.named_parameters(recurse=False):
                    parameter.copy_(torch.from_numpy(arrays[f"model.policy.{name}.{key}"].copy()))
            for key, parameter in self.lstm_actor.named_parameters():
                parameter.copy_(torch.from_numpy(arrays[f"model.lstm.{key}"].copy()))
            positions = [array for key, array in arrays.items()
                         if key.startswith("onnx::Add_") and array.shape == (17, 8)]
            if len(positions) != 1:
                raise ValueError("Could not identify the published positional embedding")
            self.features_extractor.position_embed.copy_(torch.from_numpy(positions[0].copy()))


def verify_onnx(path, device="cpu", steps=12):
    """Compare all four outputs on a recurrent sequence, including a memory reset."""
    import gymnasium as gym
    import onnxruntime as ort

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    policy = PublishedPolicy(gym.spaces.Box(0, 17, (21,), dtype=np.float32),
                             gym.spaces.Discrete(4), lambda _: 0.0).to(device).eval()
    policy.import_onnx(path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(9381)
    hidden = np.zeros((3, 1024), np.float32)
    cell = hidden.copy()
    th = torch.from_numpy(hidden).to(device)
    tc = torch.from_numpy(cell).to(device)
    errors = np.zeros(4)
    for step in range(steps):
        obs = rng.integers(0, 15, (3, 21), dtype=np.int64)
        obs[:, 17:20] = rng.integers(0, 5, (3, 3))
        obs[:, 20] = obs[:, 17:20].sum(1)
        if step == steps // 2:
            hidden[1] = 0
            cell[1] = 0
            th[1] = 0
            tc[1] = 0
        expected = session.run(None, {"observations": obs, "lstm_h": hidden, "lstm_c": cell})
        with torch.no_grad():
            actual = policy.raw_forward(torch.from_numpy(obs).to(device), th, tc)
        for index, (reference, output) in enumerate(zip(expected, actual)):
            output = output.cpu().numpy()
            errors[index] = max(errors[index], float(np.abs(reference - output).max()))
            np.testing.assert_allclose(output, reference, atol=3e-4, rtol=3e-4)
        hidden, cell = expected[2:]
        th, tc = actual[2:]
    return {"device": device, "sequence_steps": steps, "batch": 3,
            "max_absolute_errors": dict(zip(("logits", "value", "hidden", "cell"), errors.tolist())),
            "parameters": sum(p.numel() for p in policy.parameters())}
