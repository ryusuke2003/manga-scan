"""Loopback-only web UI. No CDN, analytics, external fetches, or upload service."""

import secrets
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from urllib.parse import urlparse

from flask import Flask, abort, jsonify, request, send_file

from .config import Config
from .ingest import create_project
from .pipeline import edit, run
from .storage import read_manifest


def create_app(projects, config=None):
    root = Path(projects).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = config or Config()
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=64 * 1024, TRUSTED_HOSTS=["localhost", "127.0.0.1"])
    token = secrets.token_urlsafe(32)
    app.config["API_TOKEN"] = token
    guard = threading.Lock()
    job = {"busy": False, "project": None, "error": None}

    def project_path(name):
        path = (root / name).resolve()
        if path.parent != root or not (path / "manifest.json").is_file():
            abort(404)
        return path

    @app.before_request
    def protect():
        origin = request.headers.get("Origin")
        if origin and urlparse(origin).netloc != request.host:
            abort(403)
        if request.method != "GET":
            if not secrets.compare_digest(request.headers.get("X-Manga-Token", ""), token):
                abort(403)

    @app.after_request
    def headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        )
        return response

    @app.errorhandler(ValueError)
    @app.errorhandler(KeyError)
    @app.errorhandler(StopIteration)
    def invalid(exc):
        return jsonify(error=str(exc) or "Invalid selection"), 400

    @app.errorhandler(RuntimeError)
    @app.errorhandler(OSError)
    @app.errorhandler(subprocess.SubprocessError)
    def failure(exc):
        return jsonify(error=str(exc)), 500

    @app.get("/")
    def index():
        index_file = Path(app.static_folder) / "index.html"
        if not index_file.is_file():
            return (
                "Frontend build missing. Run: npm --prefix frontend install --no-package-lock && "
                "npm --prefix frontend run build",
                503,
                {"Content-Type": "text/plain; charset=utf-8"},
            )
        return send_file(index_file)

    @app.get("/api/state")
    def state():
        projects = []
        for p in sorted(root.iterdir()):
            if not p.is_dir() or not (p / "manifest.json").is_file():
                continue
            m = read_manifest(p)
            projects.append(
                {
                    "id": p.name,
                    "source_name": Path(m["source"]).name,
                    "status": m["status"],
                    "pages": len(m["pages"]),
                }
            )
        return jsonify(token=token, projects=projects, job=job)

    @app.post("/api/choose")
    def choose():
        if sys.platform != "darwin":
            raise ValueError("Enter an absolute video path on this platform")
        result = subprocess.run(
            [
                "osascript",
                "-e",
                'POSIX path of (choose file with prompt "漫画動画を選択 (.mov / .mp4)")',
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode:
            raise ValueError("ファイル選択がキャンセルされました")
        return jsonify(path=result.stdout.strip())

    @app.post("/api/projects")
    def create():
        if not guard.acquire(blocking=False):
            return jsonify(error="処理中です"), 409
        try:
            data = request.get_json()
            cfg = Config.from_dict({**config.to_dict(), **data.get("config", {})})
            name = "scan-" + uuid.uuid4().hex[:10]
            manifest = create_project(data["video"], root / name, cfg)
            return jsonify(id=name, manifest=manifest)
        finally:
            guard.release()

    @app.get("/api/projects/<name>")
    def get_project(name):
        return jsonify(read_manifest(project_path(name)))

    @app.post("/api/projects/<name>/delete")
    def delete_project(name):
        if not guard.acquire(blocking=False):
            return jsonify(error="処理中です。完了後に削除してください"), 409
        try:
            project = project_path(name)
            shutil.rmtree(project)
            if job["project"] == name:
                job.update(project=None, error=None)
            return jsonify(deleted=name)
        finally:
            guard.release()

    @app.get("/files/<name>/<path:filename>")
    def file(name, filename):
        project = project_path(name)
        path = (project / filename).resolve()
        if (
            not path.is_relative_to(project)
            or path.suffix.lower() not in (".png", ".jpg", ".pdf")
            or not path.is_file()
        ):
            abort(404)
        return send_file(path, conditional=True)

    def start_job(name, fn):
        if not guard.acquire(blocking=False):
            return jsonify(error="処理中です。完了後に操作してください"), 409
        job.update(busy=True, project=name, error=None)

        def work():
            try:
                fn()
            except Exception as exc:
                app.logger.exception("Job failed")
                job["error"] = str(exc)
            finally:
                job["busy"] = False
                guard.release()

        threading.Thread(target=work, daemon=True).start()
        return jsonify(started=True), 202

    @app.post("/api/projects/<name>/run")
    def process(name):
        project = project_path(name)
        roi = request.get_json()["roi"]
        from .perspective import validate_roi

        validate_roi(roi)
        return start_job(name, lambda: run(project, roi))

    @app.post("/api/projects/<name>/edit")
    def review(name):
        project = project_path(name)
        data = request.get_json()
        action = data.pop("action")
        return start_job(name, lambda: edit(project, action, **data))

    return app


def serve(projects, config, port):
    if not 1024 <= port <= 65535:
        raise ValueError("Port must be 1024..65535")
    app = create_app(projects, config)
    print(f"Manga Scan: http://127.0.0.1:{port} (Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True, use_reloader=False)
