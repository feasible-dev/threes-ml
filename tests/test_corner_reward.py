"""Focused checks for the optional, isolated corner-reward experiment."""

import unittest
from unittest.mock import patch

import numpy as np

from reproduction.environment import NativeBatch, legal_masks
from reproduction_forks.corner_reward import (CORNER_INDEX, CORNER_WEIGHT, DISCOUNT,
                                              SIDE_FRACTION, SIDE_INDEX,
                                              CornerNativeBatch, CornerRewardVecEnv,
                                              corner_delta, corner_potential)
from reproduction_forks.corner_probe import (CONTROL, TREATMENT,
                                             configured_runner, register_arm)
from reproduction_forks import runner


class CornerRewardTests(unittest.TestCase):
    def board(self, first=0, second=0):
        board = np.zeros((1, 16), np.uint8)
        board[0, 0] = first
        board[0, 1] = second
        return board

    def test_side_corner_transitions_and_terminal(self):
        corner = self.board(5, 3)
        side = self.board(3, 5)
        interior = self.board(3)
        interior[0, 5] = 5
        self.assertAlmostEqual(float(corner_delta(interior, side, [False])[0]),
                               CORNER_WEIGHT * SIDE_FRACTION * DISCOUNT)
        self.assertAlmostEqual(float(corner_delta(side, corner, [False])[0]),
                               CORNER_WEIGHT * (DISCOUNT - SIDE_FRACTION))
        self.assertAlmostEqual(float(corner_delta(corner, side, [False])[0]),
                               CORNER_WEIGHT * (DISCOUNT * SIDE_FRACTION - 1))
        self.assertAlmostEqual(float(corner_delta(side, interior, [False])[0]),
                               -CORNER_WEIGHT * SIDE_FRACTION)
        self.assertAlmostEqual(float(corner_delta(corner, corner, [False])[0]),
                               CORNER_WEIGHT * (DISCOUNT - 1), places=7)
        self.assertAlmostEqual(float(corner_delta(side, side, [False])[0]),
                               CORNER_WEIGHT * SIDE_FRACTION * (DISCOUNT - 1), places=7)
        self.assertAlmostEqual(float(corner_delta(corner, corner, [True])[0]),
                               -CORNER_WEIGHT)

    def test_all_edges_threshold_and_tied_maximum(self):
        small = self.board(4, 3)
        corner_tie = self.board(5, 5)
        self.assertEqual(float(corner_potential(small)[0]), 0)
        self.assertEqual(float(corner_potential(corner_tie)[0]), 1)
        for index in CORNER_INDEX:
            board = np.zeros((1, 16), np.uint8)
            board[0, index] = 5
            self.assertEqual(float(corner_potential(board)[0]), 1)
        for index in SIDE_INDEX:
            board = np.zeros((1, 16), np.uint8)
            board[0, index] = 5
            self.assertEqual(float(corner_potential(board)[0]), SIDE_FRACTION)
        for index in (5, 6, 9, 10):
            board = np.zeros((1, 16), np.uint8)
            board[0, index] = 5
            self.assertEqual(float(corner_potential(board)[0]), 0)

    def test_native_step_keeps_game_transition_and_opening_reward(self):
        base = NativeBatch(1, heuristic=True, scaffold=0)
        treatment = CornerNativeBatch(1, heuristic=True, scaffold=0)
        seed = np.array([730100], np.uint64)
        base.reset([0], seed)
        treatment.reset([0], seed)
        action = int(np.flatnonzero(legal_masks(base.board)[0])[0])
        base_obs, base_rewards, base_done, base_stats = base.step([action])
        corner_obs, corner_rewards, corner_done, corner_stats = treatment.step([action])
        np.testing.assert_array_equal(corner_obs, base_obs)
        np.testing.assert_array_equal(corner_done, base_done)
        np.testing.assert_array_equal(corner_stats, base_stats)
        np.testing.assert_array_equal(treatment.board, base.board)
        np.testing.assert_array_equal(treatment.bag, base.bag)
        np.testing.assert_array_equal(treatment.rng, base.rng)
        np.testing.assert_array_equal(corner_rewards, base_rewards)

    def test_high_tile_step_changes_only_reward(self):
        base = NativeBatch(1, heuristic=True, scaffold=0)
        treatment = CornerNativeBatch(1, heuristic=True, scaffold=0)
        for env in (base, treatment):
            env.reset([0], np.array([730100], np.uint64))
            env.board[:] = 0
            env.board[0, 0] = 5
            env.board[0, 5] = 1
            env.meta[0, 6] = 5
            env.meta[0, 8] = 1000
        base_obs, base_reward, base_done, base_stats = base.step([3])
        corner_obs, corner_reward, corner_done, corner_stats = treatment.step([3])
        np.testing.assert_array_equal(corner_obs, base_obs)
        np.testing.assert_array_equal(corner_done, base_done)
        np.testing.assert_array_equal(corner_stats, base_stats)
        self.assertAlmostEqual(float(corner_reward[0] - base_reward[0]),
                               CORNER_WEIGHT * (DISCOUNT * SIDE_FRACTION - 1), places=6)

    def test_fork_dispatch_and_recorded_treatment(self):
        register_arm()
        selected = []

        def capture_arm(specification, arm, group):
            selected.append((arm, runner.ReproductionVecEnv))

        with patch.object(runner, "run_arm", capture_arm):
            with configured_runner():
                runner.run_arm({}, CONTROL, "test")
                runner.run_arm({}, TREATMENT, "test")
                parent = {"ppo": {"learning_rate": 0.00005},
                          "training": {"steps": 1, "lr_schedule": {}}, "fork": {}}
                specification = {"rounded_additional_steps": 8192, "parent_steps": 340041728}
                control = runner.arm_config(parent, specification, CONTROL)
                treatment = runner.arm_config(parent, specification, TREATMENT)
        self.assertIs(selected[0][1], runner.ReproductionVecEnv)
        self.assertIs(selected[1][1], CornerRewardVecEnv)
        self.assertEqual(control["fork"]["corner_reward"]["corner_weight"], 0)
        self.assertEqual(control["fork"]["corner_reward"]["side_weight"], 0)
        self.assertEqual(treatment["fork"]["corner_reward"]["corner_weight"], CORNER_WEIGHT)
        self.assertEqual(treatment["fork"]["corner_reward"]["side_weight"],
                         CORNER_WEIGHT * SIDE_FRACTION)


if __name__ == "__main__":
    unittest.main()
