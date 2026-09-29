"""Check maintained Markdown metadata and local link targets without changing files.

Frozen run/source snapshots and generated results deliberately keep their original
layout and dates. They are evidence, not maintained project documentation.
"""

from datetime import datetime
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parent.parent
STATUSES = {"current", "reference", "decision", "proposal", "historical", "snapshot"}


def maintained_documents(root: Path) -> list[Path]:
    documents = list(root.glob("*.md"))
    for directory in ("docs", "configs", "scripts", "src", "tests", "external_models"):
        documents.extend((root / directory).rglob("*.md"))
    documents.append(root / "results/README.md")
    return sorted(set(documents))


def check_document(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8-sig")
    problems = []
    header = re.match(r"\A---\n(.*?)\n---\n", text, re.DOTALL)
    metadata = {}
    if header:
        metadata = dict(line.split(": ", 1) for line in header[1].splitlines() if ": " in line)
    if metadata.get("status") not in STATUSES:
        problems.append("missing/invalid document status")
    if not metadata.get("scope"):
        problems.append("missing document scope")
    try:
        timestamp = datetime.fromisoformat(metadata.get("updated", ""))
        if timestamp.utcoffset() is None:
            raise ValueError("timezone required")
    except ValueError:
        problems.append("updated must be an ISO timestamp with UTC offset")

    # Ignore examples in fenced blocks. Only check inline Markdown file links;
    # URL availability and section anchors are outside this local check's scope.
    prose = re.sub(r"(?m)^(```|~~~).*?^\1[^\n]*$", "", text, flags=re.DOTALL)
    for link in re.findall(r"\]\(([^)]+)\)", prose):
        target = link.strip().strip("<>")
        url = urlsplit(target)
        if url.scheme or url.netloc or not url.path:
            continue
        if not (path.parent / unquote(url.path)).exists():
            problems.append(f"missing local link: {target}")
    return problems


def main() -> int:
    documents = maintained_documents(ROOT)
    errors = []
    for path in documents:
        if not path.is_file():
            errors.append(f"{path.relative_to(ROOT)}: missing document")
            continue
        errors.extend(f"{path.relative_to(ROOT)}: {message}" for message in check_document(path))
    for error in errors:
        print(error)
    print(f"Checked {len(documents)} maintained documents: {len(errors)} issue(s).")
    print("Frozen/generated evidence excluded; URL availability and anchors not checked.")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
