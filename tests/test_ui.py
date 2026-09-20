import re

from manga_scan.ui import create_app


def test_local_ui_token_origin_host_and_static_assets(tmp_path):
    app = create_app(tmp_path)
    client = app.test_client()

    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    script = re.search(r'<script[^>]+src="([^"]+)"', html)
    assert script, "Vite build should reference a JavaScript asset"
    assert client.get(script.group(1)).status_code == 200

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
