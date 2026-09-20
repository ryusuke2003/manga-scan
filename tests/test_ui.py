from manga_scan.ui import create_app


def test_local_ui_token_origin_host_and_static_assets(tmp_path):
    app = create_app(tmp_path)
    client = app.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
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
