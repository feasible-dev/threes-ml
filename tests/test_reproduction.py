import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.callbacks import BaseCallback

from reproduction.environment import NativeBatch, ReproductionVecEnv, moved_board, draw, shaped_reward, legal_masks
from reproduction.policy import PublishedPolicy
from reproduction.training import TrackedRecurrentPPO, save_boundary, restore_boundary, scheduled_learning_rate
from threes import move


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_moves_and_merge_reward(self):
        ranks = np.asarray([1, 2, 0, 0, 3, 3, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
        result, edges, reward = moved_board(ranks, 2)
        self.assertEqual(result.tolist(), [3, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(edges.tolist(), [3, 7, 11])
        self.assertAlmostEqual(reward, (3 + 3 + 4) * 0.0625)
        rng = np.random.default_rng(13)
        for _ in range(50):
            ranks = rng.integers(0, 10, 16, dtype=np.uint8)
            values = [int(r) if r < 3 else 3 * 2 ** (int(r) - 3) for r in ranks]
            board = tuple(tuple(values[i:i + 4]) for i in range(0, 16, 4))
            for action, direction in enumerate(("up", "down", "left", "right")):
                expected = move(board, direction)
                result, _, _ = moved_board(ranks, action)
                actual = [int(r) if r < 3 else 3 * 2 ** (int(r) - 3) for r in result]
                self.assertEqual(actual, [x for row in expected for x in row])

    def test_single_legal_move_fixture(self):
        board = np.array([[0, 0, 0, 0, 3, 4, 5, 6,
                           7, 8, 9, 10, 11, 12, 13, 14]], dtype=np.uint8)
        np.testing.assert_array_equal(legal_masks(board), [[True, False, False, False]])

    def test_seed_namespaces_and_bag(self):
        first = ReproductionVecEnv(4, 71)
        second = ReproductionVecEnv(4, 71)
        np.testing.assert_array_equal(first.reset(), second.reset())
        self.assertTrue(np.all(first.native.seeds >= np.uint64(2**63)))
        self.assertEqual(len(set(first.next_seeds(100).tolist())), 100)
        np.testing.assert_array_equal(first.native.bag.sum(1), [2] * 4)
        observation, _, _, _ = first.step(np.asarray([0, 1, 2, 3]))
        again, _, _, _ = second.step(np.asarray([0, 1, 2, 3]))
        np.testing.assert_array_equal(observation, again)
        first.close()
        second.close()

    def test_blocked_move_penalty_and_time_limit(self):
        env = NativeBatch(1)
        env.reset([0], [123])
        env.board[0] = [1] + [0] * 15
        _, rewards, done, stats = env.step([0])
        self.assertAlmostEqual(float(rewards[0]), -0.05)
        self.assertFalse(done[0])
        env.meta[0, 3] = 999
        _, _, done, stats = env.step([0])
        self.assertTrue(done[0])
        self.assertTrue(stats[0, 5])

    def test_bonus_preview_and_bag_consumption(self):
        env = NativeBatch(1)
        env.reset([0], [789])
        env.meta[0, 6] = 10  # 384: bonus pool 6,12,24,48.
        env.meta[0, 2] = env.meta[0, 4] = 5
        bag = env.bag.copy()
        for _ in range(30):
            draw(env.bag, env.meta, env.rng, 0)
            self.assertIn(env.meta[0, 1], (4, 5))
            self.assertTrue(env.meta[0, 1] <= env.meta[0, 0] <= env.meta[0, 1] + 2)
        np.testing.assert_array_equal(env.bag, bag)
        env.meta[0, 2] = 6
        env.bag[0] = [0, 0, 1]
        draw(env.bag, env.meta, env.rng, 0)
        self.assertEqual(env.meta[0, 0], 3)
        np.testing.assert_array_equal(env.bag[0], [4, 4, 4])

    def test_game_over_overrides_other_rewards(self):
        env = NativeBatch(1, heuristic=True)
        env.reset([0], [123])
        env.board[0] = [3, 4, 3, 4, 4, 3, 4, 3, 3, 4, 3, 4, 4, 3, 4, 3]
        env.meta[0, 6] = 4
        _, rewards, done, stats = env.step([0])
        self.assertTrue(done[0])
        self.assertEqual(rewards[0], -1)
        self.assertFalse(stats[0, 5])

    def test_corner_reward_with_two_equal_largest_tiles(self):
        # The upstream C rule treats the duplicate maximum as the second maximum.
        board = np.asarray([10, 10] + [0] * 14, dtype=np.uint8)
        self.assertAlmostEqual(shaped_reward(board, 0.0), 0.01 + 31 * 0.00003)

    def test_curriculum_reset_and_natural_only_progression(self):
        env = NativeBatch(32, scaffold=1.0)
        env.meta[:, 5] = 10
        obs = env.reset(np.arange(32), np.arange(32))
        count = (env.board != 0).sum(1)
        self.assertTrue(np.all((count >= 3) & (count <= 6)))
        self.assertTrue(np.all((env.board == 0) | ((env.board >= 8) & (env.board <= 10))))
        self.assertTrue(np.all(env.meta[:, 7] == 1))
        np.testing.assert_array_equal(obs[:, 16], env.meta[:, 1])
        np.testing.assert_array_equal(env.meta[:, 6], env.board.max(1))
        # Ending an artificial game must not increase the natural-game frontier.
        env.board[:] = [11, 12, 11, 12, 12, 11, 12, 11, 11, 12, 11, 12, 12, 11, 12, 11]
        env.meta[:, 6] = 12
        _, _, done, _ = env.step(np.zeros(32, dtype=int))
        self.assertTrue(done.all())
        np.testing.assert_array_equal(env.meta[:, 5], np.full(32, 10))
        env.meta[:, 7] = 0
        env.step(np.zeros(32, dtype=int))
        np.testing.assert_array_equal(env.meta[:, 5], np.full(32, 12))

    def test_timeout_bootstrap_is_explicit_and_cached_limit_is_preserved(self):
        for bootstrap in (False, True):
            env = ReproductionVecEnv(1, 75, bootstrap_timeouts=bootstrap)
            env.reset()
            env.native.board[0] = [1] + [0] * 15
            env.native.meta[0, 3] = 999
            env.native.meta[0, 5] = 14  # Still 1000 until a valid move updates the limit.
            _, _, done, infos = env.step(np.array([0]))
            self.assertTrue(done[0])
            self.assertEqual(infos[0]["TimeLimit.truncated"], bootstrap)
            env.close()

    def test_reward_scaler_leaves_invalid_penalty_unchanged(self):
        env = NativeBatch(1, reward_scaler=2.0)
        env.reset([0], [123])
        env.board[0] = [3, 3] + [0] * 14
        _, reward, _, _ = env.step([2])
        self.assertAlmostEqual(reward[0], 2 * 3 * .0625)
        env.board[0] = [1] + [0] * 15
        _, reward, _, _ = env.step([0])
        self.assertAlmostEqual(reward[0], -.05)

    def test_learning_rate_uses_absolute_steps(self):
        schedule = {"milestones": [8, 16], "factor": .5}
        self.assertAlmostEqual(scheduled_learning_rate(.001, schedule, 7), .001)
        self.assertAlmostEqual(scheduled_learning_rate(.001, schedule, 8), .0005)
        self.assertAlmostEqual(scheduled_learning_rate(.001, schedule, 160), .00025)

    def test_fused_sequence_matches_general_reset_path_and_gradients(self):
        torch.manual_seed(3)
        lstm = torch.nn.LSTM(5, 7)
        features = torch.randn(15, 5, requires_grad=True)
        states = (torch.randn(1, 3, 7), torch.randn(1, 3, 7))
        for starts in (torch.tensor([1, 0, 0, 0, 0] * 3),
                       torch.tensor([1, 0, 1, 0, 0] * 3)):
            starts = starts.float()
            expected, state1 = RecurrentActorCriticPolicy._process_sequence(features, states, starts, lstm)
            actual, state2 = PublishedPolicy._process_sequence(features, states, starts, lstm)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(state1[0], state2[0])
            grad1 = torch.autograd.grad(expected.sum(), features, retain_graph=True)[0]
            grad2 = torch.autograd.grad(actual.sum(), features, retain_graph=True)[0]
            torch.testing.assert_close(grad1, grad2)

    def test_critic_trains_shared_lstm(self):
        policy = PublishedPolicy(gym.spaces.Box(0, 17, (21,), np.float32), gym.spaces.Discrete(4),
                                 lambda _: .0003, hidden_size=16)
        states = (torch.zeros(1, 2, 16), torch.zeros(1, 2, 16))
        values, _, _ = policy.evaluate_actions(torch.ones(2, 21), torch.zeros(2, dtype=torch.long),
                                              RNNStates(states, states), torch.ones(2))
        values.sum().backward()
        self.assertGreater(float(policy.lstm_actor.weight_ih_l0.grad.abs().sum()), 0)
        self.assertGreater(float(policy.features_extractor.value_embed.weight.grad.abs().sum()), 0)

    def test_rollout_log_probs_reconstruct_with_reset_and_padding(self):
        class Continue(BaseCallback):
            def _on_step(self):
                return True

        env = ReproductionVecEnv(4, 74)
        try:
            model = TrackedRecurrentPPO(PublishedPolicy, env, seed=74, device="cpu",
                                        n_steps=8, batch_size=10, n_epochs=1,
                                        policy_kwargs={"hidden_size": 16})
            _, callback = model._setup_learn(32, callback=Continue(), reset_num_timesteps=True)
            env.native.meta[0, 3] = 999  # Force one mid-rollout timeout/reset.
            env.native.meta[0, 8] = 1000
            self.assertTrue(model.collect_rollouts(env, callback, model.rollout_buffer, 8))
            real = padding = resets = 0
            largest_error = 0.0
            with torch.no_grad():
                for batch in model.rollout_buffer.get(batch_size=10):
                    valid = batch.mask > 0
                    _, log_probs, _ = model.policy.evaluate_actions(
                        batch.observations, batch.actions.long().flatten(),
                        batch.lstm_states, batch.episode_starts)
                    real += int(valid.sum())
                    padding += int((~valid).sum())
                    resets += int((batch.episode_starts[valid] > 0).sum())
                    largest_error = max(largest_error,
                                        float((log_probs[valid] - batch.old_log_prob[valid]).abs().max()))
            self.assertEqual(real, 32)
            self.assertGreater(padding, 0)
            self.assertGreater(resets, 4)
            self.assertLess(largest_error, 2e-5)
        finally:
            env.close()

    def test_resume_matches_uninterrupted_next_update(self):
        with tempfile.TemporaryDirectory() as directory:
            env = ReproductionVecEnv(2, 74)
            model = TrackedRecurrentPPO(PublishedPolicy, env, seed=74, device="cpu",
                                       n_steps=4, batch_size=8, n_epochs=1,
                                       policy_kwargs={"hidden_size": 16})
            model.absolute_lr_schedule = {"milestones": [16], "factor": .5}
            model.learn(8)
            save_boundary(model, env, Path(directory))
            model.learn(8, reset_num_timesteps=False)
            expected = {key: value.clone() for key, value in model.policy.state_dict().items()}
            env2 = ReproductionVecEnv(2, 74)
            restored = restore_boundary(env2, Path(directory), "cpu")
            restored.learn(8, reset_num_timesteps=False)
            self.assertEqual(restored.absolute_lr_schedule, model.absolute_lr_schedule)
            self.assertAlmostEqual(restored.policy.optimizer.param_groups[0]["lr"], .00015)
            for key, value in restored.policy.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
            np.testing.assert_array_equal(env.native.board, env2.native.board)
            env.close()
            env2.close()


if __name__ == "__main__":
    unittest.main()
