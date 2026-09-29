import unittest

from stable_baselines3.common.env_checker import check_env

from rl_env import ThreesEnv


class ThreesEnvTests(unittest.TestCase):
    def test_score_survival_reward_uses_official_score_gain(self):
        import math

        for reward_id in ("log_score_gain_v1", "scaled_score_survival_v1"):
            env = ThreesEnv(reward_id)
            env.reset(seed=42)
            action = int(env.action_masks().nonzero()[0][0])
            before = env.game.current_score
            _, reward, _, _, info = env.step(action)
            gain = info["score"] - before
            expected = math.log1p(gain) if reward_id == "log_score_gain_v1" else gain / 27 + 0.05
            self.assertAlmostEqual(reward, expected)
            self.assertFalse(info["invalid_action"])

    def test_gym_contract_and_seeded_reset(self):
        env = ThreesEnv()
        check_env(env, warn=True)
        first, _ = env.reset(seed=42)
        second, _ = env.reset(seed=42)
        self.assertTrue((first == second).all())
        self.assertEqual(first.shape, (35,))
        self.assertEqual(env.action_masks().shape, (4,))

        action = int(env.action_masks().nonzero()[0][0])
        _, reward, terminated, truncated, info = env.step(action)
        self.assertIsInstance(reward, float)
        self.assertFalse(truncated)
        self.assertFalse(info["invalid_action"])
        self.assertFalse(terminated)

    def test_run_seed_starts_reproducible_but_distinct_episodes(self):
        env = ThreesEnv()

        def episode_starts():
            env.reset(seed=52)
            boards = [env.game.board]
            for _ in range(3):
                env.reset()
                boards.append(env.game.board)
            return boards

        first = episode_starts()
        self.assertEqual(len(set(first)), 4)
        self.assertEqual(first, episode_starts())


if __name__ == "__main__":
    unittest.main()
