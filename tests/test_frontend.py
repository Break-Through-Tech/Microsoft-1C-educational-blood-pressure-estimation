"""FastAPI serves the built frontend and falls back to index.html for client-side routes."""
import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from src.paths import Paths

INDEX = "<!doctype html><title>CufflessAI</title><div id=root></div>"


@pytest.fixture
def client(tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(INDEX, encoding="utf-8")
    (dist / "assets" / "app-abc123.js").write_text("console.log(1)", encoding="utf-8")
    (dist / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("outside dist", encoding="utf-8")
    with TestClient(create_app(Paths(root=tmp_path / "empty"), frontend_dist=dist)) as c:
        yield c


@pytest.mark.parametrize("path", ["/", "/predict", "/compare", "/explain", "/try", "/about", "/no/such/page"])
def test_client_routes_get_index(client, path):
    r = client.get(path)
    assert r.status_code == 200
    assert r.text == INDEX
    assert r.headers["content-type"].startswith("text/html")
    assert r.headers["cache-control"] == "no-cache"


def test_static_files_are_served(client):
    assert client.get("/assets/app-abc123.js").text == "console.log(1)"
    assert client.get("/favicon.svg").text == "<svg/>"


def test_api_still_wins_and_unknown_api_is_json_404(client):
    assert client.get("/api/health").json()["mode"] == "demo"
    r = client.get("/api/nope")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/json")
    assert client.get("/docs").status_code == 200


def test_no_path_traversal(client):
    for p in ["/../secret.txt", "/%2e%2e/secret.txt", "/assets/../../secret.txt"]:
        r = client.get(p)
        assert "outside dist" not in r.text


def test_without_a_build_only_the_api_is_served(tmp_path):
    with TestClient(create_app(Paths(root=tmp_path), frontend_dist=tmp_path / "missing")) as c:
        assert c.app.state.serves_frontend is False
        assert c.get("/").status_code == 404
        assert c.get("/api/health").status_code == 200
