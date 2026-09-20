import re
from contextlib import contextmanager

import manga_scan.ui as ui_module
from manga_scan.config import Config
from manga_scan.ui import create_app


def test_local_ui_token_origin_host_and_static_assets(tmp_path):
    app = create_app(tmp_path)
    client = app.test_client()

    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assets = re.findall(r'(?:src|href)="(/static/[^"]+)"', html)
    assert assets, "Vite build should reference static assets"
    assert all(client.get(asset).status_code == 200 for asset in assets)

    assert client.post("/api/projects", json={}).status_code == 403
    token = client.get("/api/state").json["token"]
    assert (
        client.post(
            "/api/projects",
            json={},
            headers={"X-Manga-Token": token, "Origin": "https://example.com"},
        ).status_code
        == 403
    )
    assert client.get("/api/state", headers={"Host": "evil.example"}).status_code == 400
    assert client.get("/files/missing/../../etc/passwd").status_code == 404
    assert (
        client.post("/api/projects", json={}, headers={"X-Manga-Token": token}).status_code == 400
    )


def test_state_exposes_effective_ui_defaults(tmp_path):
    config = Config(
        refine_quad=True,
        perspective_mode="per_page",
        page_contour_min_confidence=0.7,
        split_mode="auto",
        dewarp_mode="auto",
        illumination_correction=True,
        illumination_strength=0.45,
        white_normalization=True,
        white_target=250,
    )
    state = create_app(tmp_path, config).test_client().get("/api/state").json
    assert state["defaults"]["refine_quad"] is True
    assert state["defaults"]["perspective_mode"] == "per_page"
    assert state["defaults"]["page_contour_min_confidence"] == 0.7
    assert state["defaults"]["split_mode"] == "auto"
    assert state["defaults"]["dewarp_mode"] == "auto"
    assert state["defaults"]["illumination_correction"] is True
    assert state["defaults"]["illumination_strength"] == 0.45
    assert state["defaults"]["white_normalization"] is True
    assert state["defaults"]["white_target"] == 250


def test_missing_file_in_existing_project_returns_404(tmp_path):
    project = tmp_path / "scan-test"
    project.mkdir()
    (project / "manifest.json").write_text(
        '{"source": "/tmp/book.mp4", "status": "ready", "pages": []}\n'
    )

    client = create_app(tmp_path).test_client()
    assert client.get("/files/scan-test/pages/missing.png").status_code == 404


def test_delete_project_requires_token_and_removes_only_project(tmp_path):
    projects = tmp_path / "projects"
    project = projects / "scan-delete"
    source = tmp_path / "book.mp4"
    source.write_bytes(b"original video")
    (project / "pages").mkdir(parents=True)
    (project / "manifest.json").write_text(
        '{"source": "' + str(source) + '", "status": "complete", "pages": []}\n'
    )
    (project / "pages/page.png").write_bytes(b"page")

    client = create_app(projects).test_client()
    assert client.post("/api/projects/scan-delete/delete", json={}).status_code == 403

    token = client.get("/api/state").json["token"]
    response = client.post(
        "/api/projects/scan-delete/delete",
        json={},
        headers={"X-Manga-Token": token},
    )
    assert response.status_code == 200
    assert response.json == {"deleted": "scan-delete"}
    assert not project.exists()
    assert source.read_bytes() == b"original video"
    assert client.get("/api/projects/scan-delete").status_code == 404
    assert not any(item["id"] == "scan-delete" for item in client.get("/api/state").json["projects"])


def test_delete_project_returns_conflict_when_project_lock_is_busy(tmp_path, monkeypatch):
    project = tmp_path / "scan-busy"
    project.mkdir()
    (project / "manifest.json").write_text(
        '{"source": "/tmp/book.mp4", "status": "complete", "pages": []}\n'
    )

    @contextmanager
    def busy_project_lock(_project):
        raise ValueError("This project is busy in another process")
        yield

    monkeypatch.setattr(ui_module, "project_lock", busy_project_lock)
    client = create_app(tmp_path).test_client()
    token = client.get("/api/state").json["token"]
    response = client.post(
        "/api/projects/scan-busy/delete",
        json={},
        headers={"X-Manga-Token": token},
    )
    assert response.status_code == 409
    assert project.exists()


def test_delete_project_rejects_symlink_alias(tmp_path):
    target = tmp_path / "scan-real"
    target.mkdir()
    (target / "manifest.json").write_text(
        '{"source": "/tmp/book.mp4", "status": "complete", "pages": []}\n'
    )
    alias = tmp_path / "scan-alias"
    alias.symlink_to(target, target_is_directory=True)

    client = create_app(tmp_path).test_client()
    token = client.get("/api/state").json["token"]
    response = client.post(
        "/api/projects/scan-alias/delete",
        json={},
        headers={"X-Manga-Token": token},
    )
    assert response.status_code == 400
    assert target.exists()
    assert alias.is_symlink()
