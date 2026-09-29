"""Tableau de bord local (sans dépendance) et rapport HTML autonome."""

import argparse
import csv
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from experiment_config import load_config
from run_layout import config_path, metrics_path, report_path


TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "dashboard.html"
HTML = TEMPLATE_PATH.read_text(encoding="utf-8")


def read_run(run_dir: Path) -> dict:
    if not metrics_path(run_dir).is_file() or not config_path(run_dir).is_file():
        return {"config": {}, "metrics": []}
    with metrics_path(run_dir).open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("steps")]
    document = json.loads(config_path(run_dir).read_text(encoding="utf-8-sig"))
    if document.get("schema") in ("recurrent_reproduction_v1", "recurrent_reproduction_v2",
                                  "recurrent_mask_lr_fork_v1"):
        from reproduction.artifacts import dashboard_config
        config = dashboard_config(document)
    else:
        config = load_config(config_path(run_dir))
    return {"config": config,
            "metrics": rows}


def write_report(run_dir: Path) -> Path:
    output = report_path(run_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    content = HTML.replace("__DATA__", json.dumps(read_run(run_dir), ensure_ascii=False))
    content = content.replace("__LIVE__", "false")
    output.write_text(content, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Afficher les métriques PPO en direct")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/api/metrics":
                data = json.dumps(read_run(run_dir), ensure_ascii=False).encode("utf-8")
                kind = "application/json; charset=utf-8"
            elif self.path == "/":
                data = HTML.replace("__DATA__", "null").replace("__LIVE__", "true").encode("utf-8")
                kind = "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Tableau de bord : http://127.0.0.1:{args.port}/  (Ctrl+C pour fermer)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
