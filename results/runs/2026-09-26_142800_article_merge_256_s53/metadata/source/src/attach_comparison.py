"""Attach a completed series comparison to each participating run."""

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

from experiment_catalog import update_catalog
from run_layout import RUNS_ROOT, model_path, review_path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison_dir", type=Path)
    args = parser.parse_args()
    comparison_dir = args.comparison_dir
    models = defaultdict(list)
    for row in read_rows(comparison_dir / "holdout_models.csv"):
        models[row["agent"]].append(row)
    references = read_rows(comparison_dir / "holdout_heuristics.csv")
    expected_seeds = {row["seed"] for row in references if row["agent"] == "apparition"}
    if not models or not expected_seeds:
        raise ValueError("Comparison is missing model or expected-spawn results")

    for name, rows in models.items():
        run_dir = RUNS_ROOT / name
        if not model_path(run_dir).is_file():
            raise ValueError(f"Unknown run or missing model: {name}")
        if len(rows) != len(expected_seeds) or {row["seed"] for row in rows} != expected_seeds:
            raise ValueError(f"Held-out seeds differ for {name}")

    for name, rows in models.items():
        run_dir = RUNS_ROOT / name
        evaluation_dir = run_dir / "evaluation"
        evaluation_dir.mkdir(exist_ok=True)
        with (evaluation_dir / "holdout.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("agent", "seed", "score", "largest_tile"))
            writer.writeheader()
            writer.writerows(rows)
        shutil.copyfile(comparison_dir / "holdout_heuristics.csv",
                        evaluation_dir / "reference.csv")
        review_file = review_path(run_dir)
        review = json.loads(review_file.read_text(encoding="utf-8-sig")) if review_file.is_file() else {
            "question": "How does this condition compare with the controls on held-out games?",
            "insight": "Pending interpretation of the completed series.",
            "relevance": "pending review",
            "retention": "undecided",
            "retention_reason": "Review the comparison before trimming artifacts.",
        }
        review["holdout"] = {"file": "evaluation/holdout.csv", "agent": name}
        review["reference"] = {"file": "evaluation/reference.csv", "agent": "apparition"}
        review_file.write_text(json.dumps(review, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")
        print(f"Attached held-out results: {name}")
    update_catalog(RUNS_ROOT)


if __name__ == "__main__":
    main()
