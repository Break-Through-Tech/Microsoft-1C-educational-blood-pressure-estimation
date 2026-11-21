"""Results bundle: pack/install round trip, integrity checks, and the startup fetch used on Render."""
import hashlib
import http.server
import io
import threading
import zipfile

import pytest

from src import results_bundle as rb
from src.paths import Paths


def _fake_results(root):
    for f in rb.RESULT_FILES:
        p = root / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(f"content of {f}".encode())


def _bundle(extra: dict | None = None, drop: str | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for f in rb.RESULT_FILES:
            if f != drop:
                z.writestr(f, f"content of {f}")
        for name, data in (extra or {}).items():
            z.writestr(name, data)
    return buf.getvalue()


def test_pack_then_install_round_trip(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _fake_results(src)
    out = tmp_path / "b.zip"
    digest = rb.pack(src, out)
    assert digest == hashlib.sha256(out.read_bytes()).hexdigest()
    assert rb.install(out.read_bytes(), dst, digest) == rb.RESULT_FILES
    assert rb.missing_files(dst) == []
    assert (dst / "results/metrics.csv").read_text() == "content of results/metrics.csv"


def test_pack_refuses_when_files_are_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        rb.pack(tmp_path, tmp_path / "b.zip")


def test_install_refuses_checksum_mismatch(tmp_path):
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        rb.install(_bundle(), tmp_path, "0" * 64)
    assert rb.missing_files(tmp_path) == rb.RESULT_FILES      # nothing written


def test_install_refuses_incomplete_bundle(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        rb.install(_bundle(drop="results/residuals.csv"), tmp_path)


def test_install_writes_only_known_paths(tmp_path):
    root = tmp_path / "root"
    rb.install(_bundle(extra={"../escape.txt": "x", "backend/main.py": "evil"}), root)
    assert not (tmp_path / "escape.txt").exists()
    assert not (root / "backend" / "main.py").exists()


class _Server:
    """Serves /bundle.zip (requires a bearer token) and /redirect -> other-host /plain.zip (no auth allowed)."""

    def __init__(self, payload: bytes, token: str):
        seen = self.seen = {}

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                auth = self.headers.get("Authorization")
                seen[self.path] = auth
                if self.path == "/bundle.zip" and auth == f"Bearer {token}":
                    self._send(payload)
                elif self.path == "/redirect":
                    self.send_response(302)
                    self.send_header("Location", f"http://localhost:{self.server.server_port}/plain.zip")
                    self.end_headers()
                elif self.path == "/plain.zip" and auth is None:
                    self._send(payload)
                else:
                    self.send_response(403)
                    self.end_headers()

            def _send(self, body):
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_port
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def server():
    s = _Server(_bundle(), token="secret")
    yield s
    s.close()


def test_fetch_with_token_and_checksum(tmp_path, monkeypatch, server):
    monkeypatch.setenv("CUFFLESS_RESULTS_URL", f"http://127.0.0.1:{server.port}/bundle.zip")
    monkeypatch.setenv("CUFFLESS_RESULTS_TOKEN", "secret")
    monkeypatch.setenv("CUFFLESS_RESULTS_SHA256", hashlib.sha256(_bundle()).hexdigest())
    assert rb.fetch_if_configured(Paths(root=tmp_path)) is True
    assert rb.missing_files(tmp_path) == []
    assert rb.fetch_if_configured(Paths(root=tmp_path)) is False   # already present: no second download


def test_token_is_not_forwarded_to_another_host(tmp_path, monkeypatch, server):
    monkeypatch.setenv("CUFFLESS_RESULTS_URL", f"http://127.0.0.1:{server.port}/redirect")
    monkeypatch.setenv("CUFFLESS_RESULTS_TOKEN", "secret")
    monkeypatch.delenv("CUFFLESS_RESULTS_SHA256", raising=False)
    assert rb.fetch_if_configured(Paths(root=tmp_path)) is True
    assert server.seen["/redirect"] == "Bearer secret"
    assert server.seen["/plain.zip"] is None


def test_fetch_failure_never_raises(tmp_path, monkeypatch, server):
    monkeypatch.setenv("CUFFLESS_RESULTS_URL", f"http://127.0.0.1:{server.port}/bundle.zip")
    monkeypatch.setenv("CUFFLESS_RESULTS_TOKEN", "wrong")
    assert rb.fetch_if_configured(Paths(root=tmp_path)) is False
    assert rb.missing_files(tmp_path) == rb.RESULT_FILES


def test_no_url_means_no_fetch(tmp_path, monkeypatch):
    monkeypatch.delenv("CUFFLESS_RESULTS_URL", raising=False)
    assert rb.fetch_if_configured(Paths(root=tmp_path)) is False
