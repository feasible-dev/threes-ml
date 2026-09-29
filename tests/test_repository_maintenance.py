"""Maintenance must preserve run evidence and source-snapshot completeness."""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import experiment_catalog
import run_provenance
from check_repository import check_document


class RepositoryMaintenanceTests(unittest.TestCase):
    def test_catalog_only_does_not_write_run_cards(self):
        with tempfile.TemporaryDirectory() as directory:
            runs = Path(directory) / "runs"
            run = runs / "example"
            (run / "metadata").mkdir(parents=True)
            (run / "data").mkdir()
            (run / "metadata/config.json").write_text("{}", encoding="utf-8")
            (run / "data/metrics.csv").write_text("steps,eval_mean\n8,12\n", encoding="utf-8")
            (run / "FICHE.md").write_text("Frozen existing card", encoding="utf-8")
            before = {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()}
            row = dict.fromkeys(experiment_catalog.CATALOG_FIELDS, "")
            row.update(run="example", steps=8, source_status="snapshot_saved")
            with patch.object(experiment_catalog, "summarize_run", return_value=(row, "new card")), \
                    patch.object(experiment_catalog, "write_run_card") as write:
                experiment_catalog.update_catalog(runs, write_cards=False)
                write.assert_not_called()
            after = {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            catalog = (runs.parent / "EXPERIENCES.md").read_text(encoding="utf-8")
            self.assertIn("Generated:", catalog)
            with (runs.parent / "experiences.csv").open(encoding="utf-8", newline="") as stream:
                self.assertEqual(next(csv.DictReader(stream))["steps"], "8")

    def test_source_snapshot_keeps_current_and_relocated_launchers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ("src/example.py", "watch_policy.ps1", "scripts/legacy/run_experiments.ps1")
            for name in names:
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(f"# {name}\n", encoding="utf-8")
            with patch.object(run_provenance, "PROJECT_ROOT", root):
                hashes = run_provenance.copy_source(root / "snapshot")
            self.assertEqual(set(hashes), set(names))
            for name in names:
                self.assertEqual((root / name).read_bytes(), (root / "snapshot" / name).read_bytes())

    def test_document_check_finds_broken_links_and_requires_timezone(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "README.md"
            p.write_text("---\nstatus: current\nupdated: 2026-09-26T12:00:00\nscope: test\n---\n"
                         "[missing](missing.md)\n```text\n[example](ignored.md)\n```\n", encoding="utf-8")
            issues = check_document(p)
            self.assertEqual(len(issues), 2)
            self.assertTrue(any("UTC offset" in issue for issue in issues))
            self.assertIn("missing local link: missing.md", issues)


if __name__ == "__main__":
    unittest.main()
