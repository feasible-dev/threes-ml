import tempfile
import unittest
from pathlib import Path

from experiment_catalog import metric_summary


class CatalogMetricsTests(unittest.TestCase):
    def test_best_evaluation_can_precede_final_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            path.write_text(
                "steps,eval_mean\n128,100\n256,\n384,300\n512,200\n",
                encoding="utf-8",
            )
            last, final, best = metric_summary(path)
            self.assertEqual(last["steps"], "512")
            self.assertEqual(final["eval_mean"], "200")
            self.assertEqual(best["eval_mean"], "300")


if __name__ == "__main__":
    unittest.main()
