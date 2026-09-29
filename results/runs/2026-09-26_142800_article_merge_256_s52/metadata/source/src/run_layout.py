"""Chemins stables des artefacts d'une expérience."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = PROJECT_ROOT / "results" / "runs"


def config_path(run_dir: Path) -> Path:
    return run_dir / "metadata" / "config.json"


def method_path(run_dir: Path) -> Path:
    return run_dir / "metadata" / "method.json"


def review_path(run_dir: Path) -> Path:
    return run_dir / "metadata" / "review.json"


def provenance_path(run_dir: Path) -> Path:
    return run_dir / "metadata" / "provenance.json"


def metrics_path(run_dir: Path) -> Path:
    return run_dir / "data" / "metrics.csv"


def report_path(run_dir: Path) -> Path:
    return run_dir / "report" / "report.html"


def model_path(run_dir: Path) -> Path:
    return run_dir / "models" / "model.zip"


def checkpoints_path(run_dir: Path) -> Path:
    return run_dir / "models" / "checkpoints"
