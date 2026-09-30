"""Resume the frozen fresh corner run with the current four-threshold console trace."""

import argparse
import builtins
import io
import json
import shutil
import sys
from contextlib import contextmanager, redirect_stdout
from datetime import datetime
from pathlib import Path

import fresh_corner_initial
from resume_reproduction import format_validation_line
from run_layout import RUNS_ROOT


@contextmanager
def corrected_trace(run):
    original_print = builtins.print

    def print_with_trace(*values, **options):
        if (len(values) == 1 and isinstance(values[0], str) and
                options.get("file") in (None, sys.stdout)):
            values = (format_validation_line(run, values[0]),)
        return original_print(*values, **options)

    builtins.print = print_with_trace
    try:
        yield
    finally:
        builtins.print = original_print


def preserve_before_resume(run, plan):
    """Keep validation and selected-best artifacts from after the saved boundary."""
    label = datetime.now().astimezone().strftime("%Y-%m-%d_%H%M%S_%f")
    session = run / "metadata" / "trace_sessions" / f"{label}_from_{plan['saved_steps']}"
    before = session / "before"
    before.mkdir(parents=True)
    for relative in ("data/metrics.csv", "data/episodes.csv", "models/best.json",
                     "models/best.zip", "metadata/status.json", "metadata/config.json",
                     "metadata/method.json", "metadata/provenance.json",
                     "metadata/experiment.json", "metadata/manual_stop.json"):
        source = run / relative
        if source.is_file():
            target = before / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    evaluations = run / "evaluation"
    if evaluations.is_dir():
        shutil.copytree(evaluations, before / "evaluation")
    record = {"created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
              "purpose": "Preserve pre-resume evidence and restore four-threshold terminal trace",
              "parent_checkpoint": plan["checkpoint"], "saved_steps": plan["saved_steps"],
              "target_steps": plan["target_steps"],
              "method_change": False, "trace_thresholds": [192, 384, 768, 1536]}
    (session / "metadata.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to(RUNS_ROOT.resolve()):
        parser.error("Run must be inside results/runs")
    previous_argv = sys.argv
    try:
        sys.argv = [str(Path(fresh_corner_initial.__file__)), "--resume-run", str(run), "--preview"]
        output = io.StringIO()
        with redirect_stdout(output):
            fresh_corner_initial.main()
        plan = json.loads(output.getvalue())
        if args.preview:
            print(json.dumps({**plan, "trace_thresholds": [192, 384, 768, 1536],
                              "pre_resume_artifacts": "preserved in metadata/trace_sessions"}, indent=2))
            return
        session = preserve_before_resume(run, plan)
        print(f"Preserved pre-resume evidence: {session}", flush=True)
        sys.argv = [str(Path(fresh_corner_initial.__file__)), "--resume-run", str(run)]
        with corrected_trace(run):
            fresh_corner_initial.main()
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
