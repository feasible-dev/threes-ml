import json
import tempfile
import unittest
from pathlib import Path

from experiment_config import config_document, load_config, normalize_config, write_config


class ExperimentConfigTests(unittest.TestCase):
    def setUp(self):
        self.flat = {
            "net_arch": [64, 64], "n_steps": 128, "batch_size": 64,
            "n_epochs": 2, "learning_rate": 0.0003, "gamma": 0.99,
            "seed": 42, "train_steps": 20000, "eval_interval": 2048,
            "eval_games": 20, "eval_seed": 100042,
        }

    def test_old_and_new_formats_have_the_same_values(self):
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory) / "old.json"
            new = Path(directory) / "new.json"
            old.write_text(json.dumps(self.flat), encoding="utf-8")
            write_config(new, self.flat)
            self.assertEqual(load_config(old), load_config(new))
            self.assertEqual(list(config_document(self.flat)),
                             ["model", "ppo", "training", "evaluation", "reward"])
            self.assertEqual(load_config(old)["reward_id"], "log_score_gain_v1")

    def test_reward_variant_is_validated_and_round_trips(self):
        changed = self.flat | {"reward_id": "scaled_score_survival_v1"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "variant.json"
            write_config(path, changed)
            self.assertEqual(load_config(path)["reward_id"], changed["reward_id"])
        with self.assertRaisesRegex(ValueError, "reward_id"):
            normalize_config(self.flat | {"reward_id": "unknown"})

    def test_rejects_unknown_or_invalid_values_before_training(self):
        unknown = self.flat | {"unused": True}
        with self.assertRaises(ValueError):
            normalize_config(unknown)
        invalid = self.flat | {"batch_size": 65}
        with self.assertRaisesRegex(ValueError, "divisible"):
            normalize_config(invalid)


if __name__ == "__main__":
    unittest.main()
