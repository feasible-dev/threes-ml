"""Inspect and continue recurrent runs without changing their training implementation.

Preview/list are read-only and use only the standard library. Execution delegates
to the existing trainer and records a separate, auditable continuation session.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from reproduction_progress import format_trainer_line


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "runs"
CRITICAL_SOURCE = ("src/reproduction/environment.py", "src/reproduction/policy.py",
                   "src/reproduction/training.py", "src/reproduction/evaluation.py")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def inspect_run(run, additional_steps):
    run = run.resolve()
    if not run.is_relative_to(RUNS.resolve()):
        raise ValueError(f"Run must be inside {RUNS}")
    config = read_json(run / "metadata/config.json")
    method = read_json(run / "metadata/method.json")
    if config.get("schema") != "recurrent_reproduction_v2" or method.get("engine_revision") != 2:
        raise ValueError("Only engine-v2 runs can continue with this launcher")
    markers = sorted((run / "models/checkpoints").glob("step_*/complete.json"))
    if not markers:
        raise ValueError("No complete checkpoint yet")
    checkpoint = markers[-1].parent
    for name in ("model.zip", "runtime.pt"):
        if not (checkpoint / name).is_file():
            raise ValueError(f"Latest marked checkpoint is incomplete: {checkpoint / name}")
    start = int(read_json(checkpoint / "complete.json")["steps"])
    if checkpoint.name != f"step_{start:012d}":
        raise ValueError("Checkpoint name and transition count disagree")
    rollout = config["ppo"]["n_envs"] * config["ppo"]["n_steps"]
    if additional_steps < 1:
        raise ValueError("Additional transitions must be positive")
    planned = start + math.ceil(additional_steps / rollout) * rollout
    provenance = read_json(run / "metadata/provenance.json")
    latest_source = provenance.get("resumes", [])[-1] if provenance.get("resumes") else provenance
    mismatches = [name for name in CRITICAL_SOURCE
                  if latest_source.get("sha256", {}).get(name) != hashlib.sha256((ROOT / name).read_bytes()).hexdigest()]
    versions = method.get("versions", {})
    for name, package in (("torch", "torch"), ("sb3_contrib", "sb3-contrib"), ("numba", "numba")):
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = "missing"
        if versions.get(name) != actual:
            mismatches.append(f"{package}: recorded {versions.get(name)}, installed {actual}")
    best_path = run / "models/best.json"
    best = read_json(best_path) if best_path.exists() else {}
    return {"run": str(run), "checkpoint": str(checkpoint), "from_steps": start,
            "additional_steps_requested": additional_steps, "planned_end_steps": planned,
            "rollout_size": rollout, "seed": config["training"]["seed"],
            "device": config["training"]["device"], "lr_schedule": config["training"]["lr_schedule"],
            "status": read_json(run / "metadata/status.json").get("status", "unknown"),
            "best_validation_mean": best.get("eval_mean"), "compatibility_issues": mismatches}


def own_python_processes(rows):
    """Include the Windows venv redirector only when its identity is verified.

    The venv python.exe can stay alive while its base-interpreter child runs
    this script. That parent is part of this invocation, not another worker.
    Other Python parents (e.g. batch orchestrators) must still block training.
    """
    own = {os.getpid()}
    current = next((row for row in rows if row["ProcessId"] == os.getpid()), None)
    parent = next((row for row in rows if row["ProcessId"] == os.getppid()), None)
    if not current or not parent or current.get("ParentProcessId") != parent["ProcessId"]:
        return own
    parent_exe = parent.get("ExecutablePath")
    current_exe = current.get("ExecutablePath")
    if not parent_exe or not current_exe:
        return own
    if (os.path.normcase(parent_exe) != os.path.normcase(sys.executable)
            or os.path.normcase(current_exe) == os.path.normcase(parent_exe)):
        return own
    # Only the executable differs in the redirector/child command lines.
    # Preserve the argument text exactly; uncertain cases remain blockers.
    pattern = r'\s*(?:"[^"]*"|\S+)\s+(.+)'
    parent_args = re.fullmatch(pattern, parent.get("CommandLine") or "", re.DOTALL)
    current_args = re.fullmatch(pattern, current.get("CommandLine") or "", re.DOTALL)
    if parent_args and current_args and parent_args[1] == current_args[1]:
        own.add(parent["ProcessId"])
    return own


def active_workers():
    """Fail closed if Windows cannot inspect Python workers; never stop one."""
    if sys.platform != "win32":
        raise RuntimeError("This launcher currently checks running workers on Windows only")
    command = ("$ErrorActionPreference='Stop'; "
               "@(Get-CimInstance Win32_Process -Filter \"Name = 'python.exe' OR Name = 'pythonw.exe'\" | "
               "Select-Object ProcessId,ParentProcessId,ExecutablePath,CommandLine) | ConvertTo-Json -Compress")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                            capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise RuntimeError("Cannot inspect active Python workers. Run this launcher in your normal PowerShell terminal. "
                           "No training was started and no run files were changed. " + result.stderr.strip())
    rows = json.loads(result.stdout) if result.stdout.strip() else []
    rows = rows if isinstance(rows, list) else [rows]
    own = own_python_processes(rows)
    found = []
    for row in rows:
        if row["ProcessId"] in own:
            continue
        command = row.get("CommandLine")
        if command is None:
            raise RuntimeError(f"Cannot inspect Python PID {row['ProcessId']}; cannot safely start a continuation")
        if any(name in command.lower() for name in
               ("train_reproduction.py", "evaluate_reproduction.py", "resume_reproduction.py")):
            found.append(row)
    return found


@contextmanager
def continuation_lock():
    # An OS lock releases on process exit, including a crash. The file can remain.
    import msvcrt
    path = ROOT / ".external" / "continuation.lock"
    path.parent.mkdir(exist_ok=True)
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError("Another continuation launcher is already running") from error
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def prepare_session(plan, note):
    run = Path(plan["run"])
    label = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f") + f"_from_{plan['from_steps']}"
    directory = run / "metadata/continuations" / label
    directory.mkdir(parents=True, exist_ok=False)
    # The trainer may roll back rows after its last complete checkpoint. Preserve
    # pre-resume evidence, including a later best model, before that happens.
    for relative in ("data/metrics.csv", "data/episodes.csv", "models/best.json", "models/best.zip",
                     "metadata/status.json", "metadata/review.json", "metadata/config.json",
                     "metadata/provenance.json", "metadata/method.json"):
        source = run / relative
        if source.exists():
            target = directory / "before" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    if (run / "evaluation").exists():
        shutil.copytree(run / "evaluation", directory / "before/evaluation")
    shutil.copy2(Path(__file__), directory / "resume_reproduction.py")
    shutil.copy2(ROOT / "resume_reproduction.ps1", directory / "resume_reproduction.ps1")
    session = {**plan, "started_at": now(), "finished_at": None, "status": "prepared", "note": note,
               "command": [sys.executable, "-u", str(ROOT / "src/train_reproduction.py"),
                           "--resume", str(run), "--steps", str(plan["additional_steps_requested"])]}
    write_json(directory / "session.json", session)
    render_history(run)
    return directory, session


def render_history(run):
    lines = ["# Training continuations", "", "Budgets below are additional transitions. Each row links to its session record.", "",
             "| Started | From | Requested | Last saved | Status |", "| --- | ---: | ---: | ---: | --- |"]
    for path in sorted((run / "metadata/continuations").glob("*/session.json")):
        session = read_json(path)
        relative = path.parent.relative_to(run).as_posix()
        lines.append(f"| [{session['started_at']}]({relative}/SUMMARY.md) | {session['from_steps']:,} | "
                     f"{session['additional_steps_requested']:,} | {session.get('last_saved_steps', session['from_steps']):,} | {session['status']} |")
        summary = (f"# Continuation from {session['from_steps']:,} transitions\n\n"
                   f"Status: **{session['status']}**. Started: {session['started_at']}. Finished: {session.get('finished_at')}.\n\n"
                   f"Requested: {session['additional_steps_requested']:,} additional transitions. "
                   f"Planned boundary: {session['planned_end_steps']:,}. Latest saved boundary: {session.get('last_saved_steps', session['from_steps']):,}.\n\n"
                   f"Research note: {session['note'] or 'No note supplied.'}\n\n"
                   "[Session details](session.json) · [Terminal log](terminal.log) · [State before continuation](before/)\n\n"
                   "Weights, optimizer, LSTM state, live games, RNG and LR schedule resume from the last complete checkpoint. "
                   "The best model is used for evaluation; it is not the continuation starting point. "
                   "Old independent comparisons and reviews are archived before extending the run.\n")
        (path.parent / "SUMMARY.md").write_text(summary, encoding="utf-8")
    (run / "CONTINUATIONS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute(plan, note):
    if plan["compatibility_issues"]:
        raise ValueError("Training code/dependencies changed: " + "; ".join(plan["compatibility_issues"]))
    workers = active_workers()
    if workers:
        raise RuntimeError("Training/evaluation is already running (PIDs " +
                           ", ".join(str(p["ProcessId"]) for p in workers) + "). Let the current batch finish first.")
    with continuation_lock():
        # Re-read immediately before archiving, after activity checks.
        current = inspect_run(Path(plan["run"]), plan["additional_steps_requested"])
        if current != plan:
            raise RuntimeError("Run changed during preparation; inspect it again")
        directory, session = prepare_session(plan, note)
        run = Path(plan["run"])
        child = None
        try:
            with (directory / "terminal.log").open("w", encoding="utf-8") as log:
                child = subprocess.Popen(session["command"], cwd=ROOT, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                         env=dict(os.environ, PYTHONIOENCODING="utf-8"))
                session.update(status="running", pid=child.pid)
                write_json(directory / "session.json", session)
                # A previous comparison/review applies to the old training budget.
                # Its original files are safely archived in this session's before/.
                for path in (run / "evaluation/reproduction_validation.json", run / "metadata/review.json"):
                    if path.exists():
                        path.unlink()
                render_history(run)
                for line in child.stdout:
                    display = format_trainer_line(run, line, resuming=True)
                    if display is not None:
                        print(display, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = child.wait()
                session.update(status="complete" if code == 0 else "failed", exit_code=code)
        except KeyboardInterrupt:
            # Console Ctrl+C also reaches the trainer, which records interruption.
            session["status"] = "interrupted"
            if child is not None:
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    child.wait()
        except BaseException as error:
            session.update(status="failed", error=str(error))
            if child is not None and child.poll() is None:
                child.terminate()
                child.wait()
            raise
        finally:
            if child is not None and child.stdout is not None:
                child.stdout.close()
            session["finished_at"] = now()
            markers = sorted((run / "models/checkpoints").glob("step_*/complete.json"))
            session["last_saved_steps"] = int(read_json(markers[-1])["steps"])
            if session["status"] == "complete" and session["last_saved_steps"] < session["planned_end_steps"]:
                session["status"] = "incomplete"
            write_json(directory / "session.json", session)
            render_history(run)
        print(f"Continuation: {session['status']} | history: {run / 'CONTINUATIONS.md'}", flush=True)
        return 0 if session["status"] == "complete" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--additional-steps", type=int, default=10000000)
    parser.add_argument("--note", default="")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    if args.list:
        for run in sorted(RUNS.iterdir()):
            if not (run / "metadata/method.json").exists():
                continue
            if read_json(run / "metadata/method.json").get("engine_revision") != 2:
                continue
            try:
                plan = inspect_run(run, args.additional_steps)
                print(f"{run.name}\n  {plan['status']} | saved {plan['from_steps']:,} | best validation {plan['best_validation_mean']} | "
                      f"compatible: {not plan['compatibility_issues']}")
            except ValueError as error:
                print(f"{run.name}\n  {error}")
        return 0
    if args.run is None:
        parser.error("Use --list or --run RUN_DIRECTORY")
    try:
        plan = inspect_run(args.run, args.additional_steps)
        print(json.dumps(plan, indent=2), flush=True)
        if args.preview:
            print("Preview only: no run files changed. Execution also checks for active training/evaluation workers.")
            return 0
        return execute(plan, args.note)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"Cannot continue: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
