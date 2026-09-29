import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.callbacks import BaseCallback

from reproduction.environment import ReproductionVecEnv
from reproduction.policy import PublishedPolicy
from reproduction.training import Tracking, TrackedRecurrentPPO, save_boundary
from reproduction_forks.algorithm import ForkRecurrentPPO
from reproduction_forks.compare import evaluate_model, paired_result
from reproduction_forks.masking import MaskedPublishedPolicy, observation_action_masks
from reproduction_forks.lr_probe import (CONTROL, TREATMENT, compare_probe,
                                         probe_plan, register_probe_arm)
from reproduction_forks.runner import (Tee, arm_config, restore_parent_for_fork,
                                       restore_runtime_state, truncate_fork_diagnostics)
from extend_reproduction_fork import extension_plan
from extend_masked_lr_probe import long_plan


class Continue(BaseCallback):
    def _on_step(self):
        return True


class ForkTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_extension_plan_uses_completed_fork_endpoint_and_same_treatment(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "metadata").mkdir()
            (run / "metadata/config.json").write_text(json.dumps({
                "ppo": {"n_envs": 2, "n_steps": 4},
                "training": {"seed": 72}}), encoding="utf-8")
            inspected = {"run": str(run), "checkpoint": str(run / "models/checkpoints/step_000000000016"),
                         "saved_steps": 16, "status": "complete", "remaining_steps": 0,
                         "compatibility_issues": [], "arm": "masked_lr_00005",
                         "mask_training": True, "effective_lr": 0.00005}
            with patch("extend_reproduction_fork.inspect_fork", return_value=inspected):
                planned = extension_plan(run, 9)
                self.assertEqual(planned["parent_steps"], 16)
                self.assertEqual(planned["rounded_additional_steps"], 16)
                self.assertEqual(planned["endpoint_steps"], 32)
                self.assertEqual(list(planned["arms"]), ["masked_lr_00005"])
                with self.assertRaisesRegex(ValueError, "positive"):
                    extension_plan(run, 0)
                inspected["status"] = "interrupted"
                with self.assertRaisesRegex(ValueError, "completed"):
                    extension_plan(run, 9)

    def test_lr_probe_plans_matched_masked_branches(self):
        parent = {"parent_run": "parent", "parent_steps": 225026048,
                  "rounded_additional_steps": 5005312,
                  "arms": {CONTROL: {"mask_training": True, "effective_lr": 0.00005}}}
        with patch("reproduction_forks.lr_probe.extension_plan", return_value=parent):
            planned = probe_plan("parent", 5000000)
            self.assertEqual(list(planned["arms"]), [CONTROL, TREATMENT])
            self.assertEqual(planned["arms"][TREATMENT]["effective_lr"], 0.00010)
            self.assertTrue(planned["arms"][TREATMENT]["mask_training"])
            self.assertEqual(planned["development_comparison"],
                             {"games": 300, "seed_start": 840000})
            rejected = {**parent, "arms": {"unmasked_lr_00005": {"mask_training": False,
                                                                    "effective_lr": 0.00005}}}
            with patch("reproduction_forks.lr_probe.extension_plan", return_value=rejected):
                with self.assertRaisesRegex(ValueError, "masked LR 0.00005"):
                    probe_plan("parent", 5000000)

    def test_long_lr_00010_continuation_uses_only_completed_treatment(self):
        parent = {"parent_run": "probe", "parent_steps": 230031360,
                  "rounded_additional_steps": 100007936,
                  "arms": {TREATMENT: {"mask_training": True, "effective_lr": 0.00010}}}
        with patch("extend_masked_lr_probe.extension_plan", return_value=parent):
            planned = long_plan("probe", 100000000)
            self.assertEqual(list(planned["arms"]), [TREATMENT])
            self.assertEqual(planned["development_comparison"]["seed_start"], 850000)
            rejected = {**parent, "arms": {CONTROL: {"mask_training": True,
                                                     "effective_lr": 0.00005}}}
            with patch("extend_masked_lr_probe.extension_plan", return_value=rejected):
                with self.assertRaisesRegex(ValueError, "masked_lr_00010"):
                    long_plan("probe", 100000000)

    def test_lr_probe_restores_masked_optimizer_at_new_rate(self):
        register_probe_arm()
        with tempfile.TemporaryDirectory() as directory:
            env = ReproductionVecEnv(2, 72)
            try:
                parent = ForkRecurrentPPO(MaskedPublishedPolicy, env, seed=72,
                                          device="cpu", n_steps=4, batch_size=8,
                                          n_epochs=1, learning_rate=0.00005,
                                          policy_kwargs={"hidden_size": 16})
                parent.fork_mask_training = True
                parent.learn(8)
                checkpoint = save_boundary(parent, env, Path(directory))
                child_env = ReproductionVecEnv(2, 72)
                try:
                    child = restore_parent_for_fork(checkpoint, child_env, True, 0.00010, "cpu")
                    self.assertEqual(child.num_timesteps, 8)
                    self.assertIsInstance(child.policy, MaskedPublishedPolicy)
                    self.assertEqual(child.policy.optimizer.param_groups[0]["lr"], 0.00010)
                    before = parent.policy.optimizer.state_dict()["state"]
                    after = child.policy.optimizer.state_dict()["state"]
                    self.assertEqual(before.keys(), after.keys())
                    for index in before:
                        for name, value in before[index].items():
                            restored = after[index][name]
                            if isinstance(value, torch.Tensor):
                                torch.testing.assert_close(value, restored, rtol=0, atol=0)
                            else:
                                self.assertEqual(value, restored)
                    child.learn(8, reset_num_timesteps=False)
                    self.assertEqual(child.num_timesteps, 16)
                    self.assertEqual(child_env.invalid_count, 0)
                finally:
                    child_env.close()
            finally:
                env.close()

    def test_lr_probe_compares_paired_fixed_endpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            groups, runs = root / "fork_batches", root / "runs"
            group = "probe_test"
            group_dir = groups / group
            group_dir.mkdir(parents=True)
            parent = runs / "parent"
            checkpoint = parent / "models/checkpoints/step_000000000008"
            checkpoint.mkdir(parents=True)
            (checkpoint / "model.zip").write_bytes(b"parent model")
            (checkpoint / "runtime.pt").write_bytes(b"parent runtime")
            (parent / "metadata").mkdir()
            (parent / "metadata/config.json").write_text(json.dumps({
                "training": {"device": "cpu"}, "environment": {"terminal_rank": 16}}),
                encoding="utf-8")
            from reproduction_forks.runner import sha256
            model_hash = sha256(checkpoint / "model.zip")
            runtime_hash = sha256(checkpoint / "runtime.pt")
            (group_dir / "plan.json").write_text(json.dumps({
                "probe": "masked_lr_00005_vs_00010", "group": group,
                "parent_run": str(parent), "parent_checkpoint": str(checkpoint),
                "parent_steps": 8, "endpoint_steps": 16,
                "rounded_additional_steps": 8,
                "parent_model_sha256": model_hash,
                "parent_runtime_sha256": runtime_hash,
                "arms": {CONTROL: {}, TREATMENT: {}},
                "development_comparison": {"games": 3, "seed_start": 840000}}),
                encoding="utf-8")
            for arm, rate in ((CONTROL, 0.00005), (TREATMENT, 0.00010)):
                run = runs / f"{group}_{arm}"
                (run / "metadata").mkdir(parents=True)
                (run / "metadata/fork.json").write_text(json.dumps({
                    "arm": arm, "mask_training": True, "effective_lr": rate,
                    "parent_model_sha256": model_hash,
                    "parent_runtime_sha256": runtime_hash}), encoding="utf-8")
                (run / "metadata/status.json").write_text('{"status":"complete"}', encoding="utf-8")
                endpoint = run / "models/checkpoints/step_000000000016"
                endpoint.mkdir(parents=True)
                (endpoint / "complete.json").write_text('{"steps":16}', encoding="utf-8")
                (endpoint / "model.zip").write_bytes(arm.encode())

            def fake_evaluate(model_path, output, games, seed, terminal_rank, device):
                base = {"parent_endpoint": 100, CONTROL: 200, TREATMENT: 300}[output.stem]
                return {seed + i: {"score": base + i, "max_tile": 384,
                                   "moves": 10, "attempts": 10, "truncated": False}
                        for i in range(games)}

            with patch("reproduction_forks.lr_probe.runner.GROUP_ROOT", groups), \
                 patch("reproduction_forks.lr_probe.runner.RUNS_ROOT", runs), \
                 patch("reproduction_forks.lr_probe.evaluate_model", side_effect=fake_evaluate):
                result = compare_probe(group)
            self.assertEqual(result["paired_comparisons"]["lr_00010_minus_00005"][
                "mean_score_difference"], 100)
            self.assertTrue((group_dir / "comparison.md").is_file())
            self.assertTrue((group_dir / "comparison.json").is_file())

    def test_terminal_display_keeps_archived_trainer_output(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "evaluation").mkdir()
            (run / "evaluation/step_000000000008.csv").write_text(
                "seed,max_tile\n1,192\n2,768\n", encoding="utf-8")
            visible, archived = io.StringIO(), io.StringIO()
            tee = Tee(visible, archived, run=run)
            original = ("           8 steps | 2,500 steps/s | validation 500.0 "
                        "| >=384 50.0% | >=3072 0.0% >=6144 0.0%")
            tee.write(original)
            tee.write("\n")
            self.assertEqual(archived.getvalue(), original + "\n")
            self.assertIn("| >=192 100.0% | >=384 50.0% | >=768 50.0%",
                          visible.getvalue())

    def test_one_legal_move_and_finite_padding(self):
        env = ReproductionVecEnv(1, 72)
        try:
            policy = MaskedPublishedPolicy(env.observation_space, env.action_space,
                                           lambda _: 0.00015, hidden_size=16)
            obs = np.zeros((1, 21), dtype=np.float32)
            obs[0, :16] = [0, 0, 0, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
            obs[0, 16] = 1
            np.testing.assert_array_equal(observation_action_masks(obs), [[True, False, False, False]])
            memory = (torch.zeros(1, 1, 16), torch.zeros(1, 1, 16))
            states = RNNStates(memory, memory)
            tensor = torch.from_numpy(obs)
            action, _, log_prob, _ = policy(tensor, states, torch.ones(1))
            self.assertEqual(int(action), 0)
            self.assertEqual(float(log_prob.detach()), 0.0)
            predicted, _ = policy.predict(obs, deterministic=False)
            self.assertEqual(int(predicted[0]), 0)
            _, replay_log_prob, entropy = policy.evaluate_actions(
                tensor, action, states, torch.ones(1))
            self.assertEqual(float(replay_log_prob.detach()), 0.0)
            self.assertEqual(float(entropy.detach()), 0.0)
            gradients = torch.autograd.grad(replay_log_prob.sum(), policy.action_net.parameters())
            self.assertTrue(all(torch.count_nonzero(gradient) == 0 for gradient in gradients))
            zero = torch.zeros((1, 21))
            with self.assertRaisesRegex(ValueError, "no legal actions"):
                policy(zero, states, torch.ones(1))
            _, padded_log_prob, padded_entropy = policy.evaluate_actions(
                zero, torch.zeros(1, dtype=torch.long), states, torch.ones(1))
            self.assertTrue(torch.isfinite(padded_log_prob).all())
            self.assertTrue(torch.isfinite(padded_entropy).all())
            self.assertAlmostEqual(float(padded_entropy.detach()), np.log(4), places=6)
        finally:
            env.close()

    def test_masked_rollout_reconstructs_with_reset_and_padding(self):
        env = ReproductionVecEnv(4, 72)
        try:
            model = ForkRecurrentPPO(MaskedPublishedPolicy, env, seed=72, device="cpu",
                                     n_steps=8, batch_size=10, n_epochs=1,
                                     policy_kwargs={"hidden_size": 16})
            model.fork_mask_training = True
            _, callback = model._setup_learn(32, callback=Continue(), reset_num_timesteps=True)
            env.native.meta[0, 3] = 999
            env.native.meta[0, 8] = 1000
            self.assertTrue(model.collect_rollouts(env, callback, model.rollout_buffer, 8))
            buffer = model.rollout_buffer
            self.assertTrue(np.array_equal(buffer.action_masks.reshape(-1, 4),
                                           observation_action_masks(buffer.observations.reshape(-1, 21))))
            selected = np.take_along_axis(buffer.action_masks, buffer.actions.astype(int), axis=2)
            self.assertTrue(selected.all())
            real = padding = resets = 0
            largest_error = 0.0
            with torch.no_grad():
                for batch in buffer.get(batch_size=10):
                    valid = batch.mask > 0
                    _, log_probs, entropy = model.policy.evaluate_actions(
                        batch.observations, batch.actions.long().flatten(),
                        batch.lstm_states, batch.episode_starts)
                    real += int(valid.sum())
                    padding += int((~valid).sum())
                    resets += int((batch.episode_starts[valid] > 0).sum())
                    largest_error = max(largest_error,
                                        float((log_probs[valid] - batch.old_log_prob[valid]).abs().max()))
                    self.assertTrue(torch.isfinite(log_probs).all())
                    self.assertTrue(torch.isfinite(entropy).all())
            self.assertEqual(real, 32)
            self.assertGreater(padding, 0)
            self.assertGreater(resets, 4)
            self.assertLess(largest_error, 2e-5)
        finally:
            env.close()

    def test_parent_optimizer_restores_into_masked_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            env = ReproductionVecEnv(2, 72)
            try:
                parent = TrackedRecurrentPPO(PublishedPolicy, env, seed=72, device="cpu",
                                             n_steps=4, batch_size=8, n_epochs=1,
                                             policy_kwargs={"hidden_size": 16})
                parent.learn(8)
                checkpoint = save_boundary(parent, env, Path(directory))
                env2 = ReproductionVecEnv(2, 72)
                try:
                    fork = ForkRecurrentPPO.load(
                        checkpoint / "model.zip", env=env2, device="cpu", force_reset=False,
                        custom_objects={"policy_class": MaskedPublishedPolicy})
                    self.assertIsInstance(fork.policy, MaskedPublishedPolicy)
                    before = parent.policy.optimizer.state_dict()
                    after = fork.policy.optimizer.state_dict()
                    self.assertEqual(before["param_groups"], after["param_groups"])
                    self.assertEqual(before["state"].keys(), after["state"].keys())
                    for index in before["state"]:
                        for name, original in before["state"][index].items():
                            restored = after["state"][index][name]
                            if isinstance(original, torch.Tensor):
                                torch.testing.assert_close(original, restored, rtol=0, atol=0)
                            else:
                                self.assertEqual(original, restored)
                finally:
                    env2.close()
            finally:
                env.close()

    def test_restored_fork_completes_fresh_update_and_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent_dir = root / "parent"
            env = ReproductionVecEnv(2, 72)
            try:
                parent = TrackedRecurrentPPO(PublishedPolicy, env, seed=72, device="cpu",
                                             n_steps=4, batch_size=8, n_epochs=1,
                                             policy_kwargs={"hidden_size": 16})
                parent.learn(8)
                checkpoint = save_boundary(parent, env, parent_dir)
                run_dir = root / "fork"
                for name in ("metadata", "data", "evaluation", "models", "report"):
                    (run_dir / name).mkdir(parents=True)
                config = {
                    "schema": "recurrent_reproduction_v2",
                    "model": {"hidden_size": 16},
                    "environment": {"heuristic": True, "snake_weight": 0.0,
                                    "scaffold": 0.5, "terminal_rank": 16,
                                    "reward_scaler": 1.0, "bootstrap_timeouts": False},
                    "ppo": {"n_envs": 2, "n_steps": 4, "batch_size": 8,
                            "n_epochs": 1, "learning_rate": 0.0003,
                            "gamma": 0.999, "gae_lambda": 0.95, "ent_coef": 0.01,
                            "clip_range": 0.2, "vf_coef": 0.5,
                            "max_grad_norm": 0.5, "target_kl": 0.03},
                    "training": {"seed": 72, "steps": 8, "device": "cpu",
                                 "torch_threads": 1, "checkpoint_interval": 8,
                                 "lr_schedule": {"milestones": [5], "factor": 0.5}},
                    "evaluation": {"games": 2, "seed": 710000,
                                   "interval": 8, "batch_size": 2}}
                specification = {"parent_steps": 8, "rounded_additional_steps": 8}
                config = arm_config(config, specification, "masked_lr_00005")
                (run_dir / "metadata/config.json").write_text(json.dumps(config), encoding="utf-8")
                (run_dir / "metadata/method.json").write_text(json.dumps({
                    "parameters": sum(p.numel() for p in parent.policy.parameters()),
                    "device": "cpu", "engine_revision": 2,
                    "initialization": "local checkpoint", "fork": {
                        "parent_run": "parent", "parent_steps": 8,
                        "mask_training": True, "effective_lr": 0.00005}}), encoding="utf-8")
                (run_dir / "metadata/provenance.json").write_text(json.dumps({
                    "started_at_local": "2026-09-26T00:00:00-04:00",
                    "source_status": "snapshot_saved", "git_commit": None}), encoding="utf-8")
                fork_env = ReproductionVecEnv(2, 72, run_dir / "data/episodes.csv",
                                               **config["environment"])
                try:
                    fork = restore_parent_for_fork(checkpoint, fork_env, True, 0.00005, "cpu")
                    self.assertEqual(fork.num_timesteps, 8)
                    self.assertEqual(fork.policy.optimizer.param_groups[0]["lr"], 0.00005)
                    np.testing.assert_array_equal(fork_env.native.board, env.native.board)
                    tracking = Tracking(fork_env, run_dir, config, fork.num_timesteps)
                    fork.tracking = tracking
                    fork.fork_diagnostics_path = run_dir / "data/fork_updates.csv"
                    fork.learn(8, callback=tracking, reset_num_timesteps=False)
                    self.assertEqual(fork.num_timesteps, 16)
                    self.assertTrue((run_dir / "models/checkpoints/step_000000000016/complete.json").is_file())
                    self.assertTrue((run_dir / "report/report.html").is_file())
                    self.assertTrue((run_dir / "FICHE.md").is_file())
                    with (run_dir / "data/fork_updates.csv").open(encoding="utf-8", newline="") as stream:
                        import csv
                        diagnostics = list(csv.DictReader(stream))
                    self.assertEqual(len(diagnostics), 1)
                    self.assertEqual(int(diagnostics[0]["invalid_attempts"]), 0)
                    self.assertEqual(int(diagnostics[0]["actual_ppo_epochs"]), 1)
                    compared = evaluate_model(
                        run_dir / "models/checkpoints/step_000000000016/model.zip",
                        run_dir / "evaluation/endpoint_check.csv", 2, 710100, 16, "cpu")
                    self.assertEqual(len(compared), 2)
                finally:
                    fork_env.close()
            finally:
                env.close()

    def test_paired_comparison_uses_matching_seeds(self):
        earlier = {710000 + i: {"score": i} for i in range(4)}
        later = {710000 + i: {"score": i + 10} for i in range(4)}
        result = paired_result(earlier, later)
        self.assertEqual(result["mean_score_difference"], 10)
        self.assertEqual(result["wins"], 4)
        self.assertEqual(result["paired_bootstrap_95_percentile_interval"], [10, 10])
        with self.assertRaisesRegex(ValueError, "matching game seeds"):
            paired_result(earlier, {710001: {"score": 0}})

    def test_resume_discards_uncheckpointed_diagnostic_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fork_updates.csv"
            path.write_text("steps,valid_moves\n16,8\n24,7\n32,8\n", encoding="utf-8")
            truncate_fork_diagnostics(path, 16)
            self.assertEqual(path.read_text(encoding="utf-8"),
                             "steps,valid_moves\n16,8\n")

    def test_masked_fork_resume_matches_uninterrupted_update(self):
        with tempfile.TemporaryDirectory() as directory:
            env = ReproductionVecEnv(2, 72)
            try:
                fork = ForkRecurrentPPO(MaskedPublishedPolicy, env, seed=72, device="cpu",
                                        n_steps=4, batch_size=8, n_epochs=1,
                                        learning_rate=0.00005,
                                        policy_kwargs={"hidden_size": 16})
                fork.absolute_lr_schedule = {"milestones": [], "factor": 1.0}
                fork.fork_mask_training = True
                fork.learn(8)
                checkpoint = save_boundary(fork, env, Path(directory))
                fork.learn(8, reset_num_timesteps=False)
                expected = {name: value.clone() for name, value in fork.policy.state_dict().items()}
                env2 = ReproductionVecEnv(2, 72)
                try:
                    restored = ForkRecurrentPPO.load(checkpoint / "model.zip", env=env2,
                                                     device="cpu", force_reset=False)
                    state = torch.load(checkpoint / "runtime.pt", map_location="cpu", weights_only=False)
                    restore_runtime_state(state, restored, env2, True, 0.00005, "cpu")
                    restored.learn(8, reset_num_timesteps=False)
                    for name, value in restored.policy.state_dict().items():
                        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
                    np.testing.assert_array_equal(env.native.board, env2.native.board)
                finally:
                    env2.close()
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
