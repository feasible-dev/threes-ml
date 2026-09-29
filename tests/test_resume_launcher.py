"""Continuation bookkeeping tests use isolated files and a tiny fake trainer."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import resume_reproduction as resume


class ResumeLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "results/runs/example"
        self.run.mkdir(parents=True)
        self.patches = [patch.object(resume, "ROOT", self.root),
                        patch.object(resume, "RUNS", self.root / "results/runs")]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        for name in resume.CRITICAL_SOURCE:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# unchanged training source\n", encoding="utf-8")
        (self.root / "resume_reproduction.ps1").write_text("# test launcher\n", encoding="utf-8")
        self.write("metadata/config.json", {"schema":"recurrent_reproduction_v2", "ppo":{"n_envs":2,"n_steps":4},
                   "training":{"seed":72,"device":"cpu","lr_schedule":{"milestones":[16],"factor":.5}}})
        versions = {name:importlib.metadata.version(package) for name,package in
                    (("torch","torch"),("sb3_contrib","sb3-contrib"),("numba","numba"))}
        self.write("metadata/method.json", {"engine_revision":2,"versions":versions})
        self.write("metadata/provenance.json", {"sha256": {n:hashlib.sha256((self.root/n).read_bytes()).hexdigest() for n in resume.CRITICAL_SOURCE}})
        self.write("metadata/status.json", {"status":"complete"})
        self.write("models/best.json", {"steps":8,"eval_mean":100})
        self.write("metadata/review.json", {"insight":"old budget"})
        self.write("evaluation/reproduction_validation.json", {"metrics":{"eval_mean":100}})
        (self.run / "models/best.zip").write_bytes(b"old best")
        (self.run / "data").mkdir()
        (self.run / "data/metrics.csv").write_text("steps,eval_mean\n8,100\n", encoding="utf-8")
        (self.run / "data/episodes.csv").write_text("seed,score\n1,100\n", encoding="utf-8")
        self.write("models/checkpoints/step_000000000008/complete.json", {"steps":8})
        for name in ("model.zip", "runtime.pt"):
            (self.run / "models/checkpoints/step_000000000008" / name).write_bytes(b"checkpoint")

    def write(self, name, document):
        p = self.run / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(document), encoding="utf-8")

    def hashes(self):
        return {str(p.relative_to(self.run)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.run.rglob("*") if p.is_file()}

    def test_validation_display_replaces_only_large_tile_rates(self):
        (self.run / "evaluation/step_000000000008.csv").write_text(
            "seed,max_tile\n1,192\n2,768\n3,1536\n4,6144\n", encoding="utf-8")
        original = ("           8 steps | 2,500 steps/s | Vloss 0.123 | validation 1,200.0 "
                    "| >=384 75.0% | >=3072 25.0% >=6144 25.0%\n")
        displayed = resume.format_validation_line(self.run, original)
        self.assertEqual(displayed,
                         "           8 steps | 2,500 steps/s | Vloss 0.123 | validation 1,200.0 "
                         "| >=192 100.0% | >=384 75.0% | >=768 75.0%\n")
        routine = "          16 steps | 2,600 steps/s | Vloss 0.120\n"
        self.assertEqual(resume.format_validation_line(self.run, routine), routine)

    def test_inspection_is_read_only_and_rounds_additional_budget(self):
        before = self.hashes()
        plan = resume.inspect_run(self.run, 9)
        self.assertEqual(plan["from_steps"], 8)
        self.assertEqual(plan["planned_end_steps"], 24)
        self.assertEqual(plan["compatibility_issues"], [])
        self.assertEqual(self.hashes(), before)
        # An unfinished later save never becomes the continuation starting point.
        (self.run / "models/checkpoints/step_000000000016").mkdir()
        self.assertEqual(resume.inspect_run(self.run, 8)["from_steps"], 8)

    def test_changed_source_blocks_execution_before_writes(self):
        (self.root / resume.CRITICAL_SOURCE[0]).write_text("# changed\n", encoding="utf-8")
        plan = resume.inspect_run(self.run, 8)
        before = self.hashes()
        with self.assertRaisesRegex(ValueError, "changed"):
            resume.execute(plan, "")
        self.assertEqual(self.hashes(), before)

    def test_active_training_blocks_even_for_a_completed_other_run(self):
        plan = resume.inspect_run(self.run, 8)
        before = self.hashes()
        with patch.object(resume, "active_workers", return_value=[{"ProcessId":123}]), \
                self.assertRaisesRegex(RuntimeError, "already running"):
            resume.execute(plan, "")
        self.assertEqual(self.hashes(), before)

    def test_cim_detection_and_permission_failure(self):
        ok = subprocess.CompletedProcess([], 0, json.dumps([{"ProcessId":123,"CommandLine":"python src/train_reproduction.py --run-dir x"}]), "")
        with patch.object(resume.sys, "platform", "win32"), patch.object(resume.subprocess, "run", return_value=ok):
            self.assertEqual(resume.active_workers()[0]["ProcessId"], 123)
        bad = subprocess.CompletedProcess([], 1, "", "Access denied")
        with patch.object(resume.sys, "platform", "win32"), patch.object(resume.subprocess, "run", return_value=bad), \
                self.assertRaisesRegex(RuntimeError, "Cannot inspect"):
            resume.active_workers()

    def test_actual_subprocess_archives_and_tracks_continuation(self):
        # Exercises the launcher subprocess/log path without invoking ML or a GPU.
        trainer = self.root / "src/train_reproduction.py"
        trainer.write_text('''import argparse, json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--resume',type=Path)
p.add_argument('--steps',type=int)
a=p.parse_args()
d=a.resume/'models/checkpoints/step_000000000016'
d.mkdir()
(d/'model.zip').write_bytes(b'new model')
(d/'runtime.pt').write_bytes(b'new state')
(d/'complete.json').write_text(json.dumps({'steps':16}))
print('continued to 16 steps',flush=True)
''', encoding="utf-8")
        plan = resume.inspect_run(self.run, 8)
        with patch.object(resume, "active_workers", return_value=[]):
            self.assertEqual(resume.execute(plan, "Test continuation"), 0)
        session_path = next((self.run / "metadata/continuations").glob("*/session.json"))
        session = resume.read_json(session_path)
        self.assertEqual(session["last_saved_steps"], 16)
        self.assertEqual(session["status"], "complete")
        self.assertEqual(session["note"], "Test continuation")
        before = session_path.parent / "before"
        self.assertTrue((before / "evaluation/reproduction_validation.json").exists())
        self.assertTrue((before / "metadata/review.json").exists())
        self.assertEqual((before / "models/best.zip").read_bytes(), b"old best")
        self.assertFalse((self.run / "evaluation/reproduction_validation.json").exists())
        self.assertFalse((self.run / "metadata/review.json").exists())
        self.assertIn("continued to 16 steps", (session_path.parent / "terminal.log").read_text())
        self.assertIn("complete", (self.run / "CONTINUATIONS.md").read_text())
        self.assertEqual(resume.inspect_run(self.run, 8)["from_steps"], 16)

    def test_venv_redirector_is_own_process_but_other_trainers_still_block(self):
        venv = r'C:\project\.venv\Scripts\python.exe'
        base = r'C:\Python311\python.exe'
        args = ' -u src/resume_reproduction.py --run "results/runs/example"'
        rows = [
            {"ProcessId": 10, "ParentProcessId": 9, "ExecutablePath": venv,
             "CommandLine": f'"{venv}"{args}'},
            {"ProcessId": 11, "ParentProcessId": 10, "ExecutablePath": base,
             "CommandLine": f'"{base}"{args}'},
            {"ProcessId": 20, "ParentProcessId": 19, "ExecutablePath": venv,
             "CommandLine": f'"{venv}" src/train_reproduction.py --run-dir other'},
        ]
        with patch.object(resume.os, "getpid", return_value=11), \
                patch.object(resume.os, "getppid", return_value=10), \
                patch.object(resume.sys, "executable", venv), \
                patch.object(resume.sys, "platform", "win32"):
            self.assertEqual(resume.own_python_processes(rows), {10, 11})
            result = subprocess.CompletedProcess([], 0, json.dumps(rows), "")
            with patch.object(resume.subprocess, "run", return_value=result):
                self.assertEqual([r["ProcessId"] for r in resume.active_workers()], [20])
            # A Python parent orchestrating training is not our redirector.
            rows[0]["CommandLine"] = f'"{venv}" src/train_reproduction.py'
            self.assertEqual(resume.own_python_processes(rows), {11})
            # Matching arguments alone do not identify the venv executable.
            rows[0]["CommandLine"] = f'"{venv}"{args}'
            rows[0]["ExecutablePath"] = base
            self.assertEqual(resume.own_python_processes(rows), {11})


if __name__ == "__main__":
    unittest.main()
