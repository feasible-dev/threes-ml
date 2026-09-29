import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import attach_comparison


class AttachComparisonTests(unittest.TestCase):
    def test_custom_csv_names_attach_results_to_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            comparison = root / "comparison"
            comparison.mkdir()
            run = root / "runs" / "sample_run"
            (run / "models").mkdir(parents=True)
            (run / "metadata").mkdir()
            (run / "models" / "model.zip").touch()
            with (comparison / "models.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("agent", "seed", "score", "largest_tile"))
                writer.writerow(("sample_run", 500000, 123, 12))
            with (comparison / "heuristics.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("agent", "seed", "score", "moves", "max_tile"))
                writer.writerow(("apparition", 500000, 456, 20, 24))

            arguments = ["attach_comparison.py", str(comparison), "--models-csv", "models.csv",
                         "--heuristics-csv", "heuristics.csv"]
            with patch.object(sys, "argv", arguments), patch.object(attach_comparison, "RUNS_ROOT", root / "runs"), \
                    patch.object(attach_comparison, "update_catalog"):
                attach_comparison.main()

            review = json.loads((run / "metadata" / "review.json").read_text(encoding="utf-8"))
            self.assertEqual(review["holdout"]["agent"], "sample_run")
            self.assertEqual(review["reference"]["agent"], "apparition")
            self.assertEqual(len((run / "evaluation" / "holdout.csv").read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
