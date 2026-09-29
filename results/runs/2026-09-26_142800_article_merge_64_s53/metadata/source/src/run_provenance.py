"""Copie le code utilisé par une nouvelle run et enregistre ses empreintes."""

import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from run_layout import PROJECT_ROOT, provenance_path


def git_commit() -> str | None:
    if not (PROJECT_ROOT / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
            text=True, check=False,
        )
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def copy_source(snapshot: Path) -> dict[str, str]:
    """Copie les fichiers et retourne leur SHA-256, chemins relatifs au projet."""
    files = sorted(path for path in (PROJECT_ROOT / "src").rglob("*")
                   if path.is_file() and path.suffix in {".py", ".html"})
    files += [PROJECT_ROOT / name for name in
              ("run_experiments.ps1", "run_next_experiments.ps1",
               "requirements.txt", "requirements-ml.txt")]
    hashes = {}
    for source in files:
        if not source.is_file():
            continue
        relative = source.relative_to(PROJECT_ROOT)
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[relative.as_posix()] = hashlib.sha256(destination.read_bytes()).hexdigest()
    return hashes


def snapshot_source(run_dir: Path) -> None:
    """Garde les fichiers exacts, car un commit seul ne couvre pas les modifications locales."""
    hashes = copy_source(run_dir / "metadata" / "source")
    provenance = {
        "started_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "source_status": "snapshot_saved",
        "source_directory": "metadata/source",
        "sha256": hashes,
    }
    provenance_path(run_dir).write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def snapshot_resume(run_dir: Path, from_steps: int) -> None:
    """Une reprise peut utiliser du code différent : en archiver une nouvelle copie."""
    provenance_file = provenance_path(run_dir)
    provenance = json.loads(provenance_file.read_text(encoding="utf-8-sig"))
    resumed_at = datetime.now().astimezone()
    label = resumed_at.strftime("%Y-%m-%d_%H%M%S_%f") + f"_from_{from_steps}"
    relative = Path("metadata") / "resumes" / label / "source"
    hashes = copy_source(run_dir / relative)
    provenance.setdefault("resumes", []).append({
        "resumed_at_local": resumed_at.isoformat(timespec="seconds"),
        "from_steps": from_steps,
        "git_commit": git_commit(),
        "source_directory": relative.as_posix(),
        "sha256": hashes,
    })
    provenance_file.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
