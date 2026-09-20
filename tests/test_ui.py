import re

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


def test_missing_file_in_existing_project_returns_404(tmp_path):
    project = tmp_path / "scan-test"
    project.mkdir()
    (project / "manifest.json").write_text(
        '{"source": "/tmp/book.mp4", "status": "ready", "pages": []}\n'
    )

    client = create_app(tmp_path).test_client()
    assert client.get("/files/scan-test/pages/missing.png").status_code == 404


def test_delete_project_requires_token_and_removes_project(tmp_path):
    project = tmp_path / "scan-delete"
    (project / "pages").mkdir(parents=True)
    (project / "manifest.json").write_text(
        '{"source": "/tmp/book.mp4", "status": "complete", "pages": []}\n'
    )
    (project / "pages/page.png").write_bytes(b"page")

    client = create_app(tmp_path).test_client()
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
    assert client.get("/api/projects/scan-delete").status_code == 404
