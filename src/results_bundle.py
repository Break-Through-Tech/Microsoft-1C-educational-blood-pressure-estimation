"""The trained results as one zip ("results bundle"), so a host that builds only from git (Render) can fetch
them at startup instead of committing them.

    python -m src.results_bundle pack                 # writes results-bundle.zip and prints its SHA-256
    CUFFLESS_RESULTS_URL=... python scripts/start.py  # start.py fetches the bundle if files are missing

Environment variables read by fetch_if_configured():
    CUFFLESS_RESULTS_URL     where the zip lives (e.g. a release asset in a PRIVATE GitHub repo, via the API URL
                             https://api.github.com/repos/<owner>/<repo>/releases/assets/<id>)
    CUFFLESS_RESULTS_TOKEN   optional bearer token for a private URL (e.g. a fine-grained GitHub token with
                             read access to that one repo's contents)
    CUFFLESS_RESULTS_SHA256  optional but recommended: the zip's SHA-256. The .pt checkpoint is loaded with
                             weights_only=False, so a swapped file could run code; a mismatch is refused.

Any failure is reported and the app starts in demo mode; it never prevents startup.
"""
from __future__ import annotations

import hashlib
import io
import os
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from .paths import Paths

RESULT_FILES = [
    "artifacts/resnet/resnet_bp_v3.pt",
    "artifacts/baseline/mean_baseline.json",
    "results/best_model.json",
    "results/metrics.csv",
    "results/residuals.csv",
    "data/test_samples.npz",
]
MAX_BUNDLE_BYTES = 200 * 1024 * 1024
DEFAULT_BUNDLE = "results-bundle.zip"


def _log(msg: str) -> None:
    print(f"[results] {msg}", file=sys.stderr, flush=True)


def missing_files(root: Path) -> list[str]:
    return [f for f in RESULT_FILES if not (root / f).is_file()]


def pack(root: Path, out: Path) -> str:
    """Zip the result files (paths relative to the repo root) and return the zip's SHA-256."""
    missing = missing_files(root)
    if missing:
        raise FileNotFoundError("missing result files (run scripts/evaluate.py first): " + ", ".join(missing))
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in RESULT_FILES:
            z.write(root / f, arcname=f)
    return hashlib.sha256(out.read_bytes()).hexdigest()


class _DropAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """GitHub release downloads redirect to signed storage URLs that reject an Authorization header."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc:
            new.headers.pop("Authorization", None)
            new.unredirected_hdrs.pop("Authorization", None)
        return new


def download(url: str, token: str | None = None, timeout: float = 60) -> bytes:
    headers = {"User-Agent": "cufflessai-results-fetch"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if urllib.parse.urlsplit(url).netloc == "api.github.com":
        headers["Accept"] = "application/octet-stream"       # the asset bytes, not its JSON description
    opener = urllib.request.build_opener(_DropAuthOnRedirect)
    with opener.open(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
        data = r.read(MAX_BUNDLE_BYTES + 1)
    if len(data) > MAX_BUNDLE_BYTES:
        raise ValueError(f"bundle is larger than {MAX_BUNDLE_BYTES // 2**20} MB")
    return data


def install(data: bytes, root: Path, sha256: str | None = None) -> list[str]:
    """Verify (optionally) and extract exactly the known result files; nothing else is written."""
    if sha256:
        got = hashlib.sha256(data).hexdigest()
        if got.lower() != sha256.strip().lower():
            raise ValueError(f"SHA-256 mismatch: expected {sha256.strip()}, got {got}")
    written = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = set(z.namelist())
        absent = [f for f in RESULT_FILES if f not in names]
        if absent:
            raise ValueError("bundle is missing: " + ", ".join(absent))
        for f in RESULT_FILES:                       # fixed paths: no zip-slip, no surprise files
            dest = root / f
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".part")
            tmp.write_bytes(z.read(f))
            os.replace(tmp, dest)
            written.append(f)
    return written


def fetch_if_configured(paths: Paths | None = None) -> bool:
    """Fetch the bundle when CUFFLESS_RESULTS_URL is set and files are missing. Returns True if installed."""
    root = (paths or Paths()).root
    url = os.environ.get("CUFFLESS_RESULTS_URL", "").strip()
    if not url:
        return False
    if not missing_files(root):
        _log("result files already present; not downloading")
        return False
    sha = os.environ.get("CUFFLESS_RESULTS_SHA256") or None
    if not sha:
        _log("warning: CUFFLESS_RESULTS_SHA256 is not set, so the download can't be verified")
    try:
        data = download(url, os.environ.get("CUFFLESS_RESULTS_TOKEN") or None)
        written = install(data, root, sha)
    except Exception as e:  # noqa: BLE001 - never block startup; the app falls back to demo mode
        _log(f"could not fetch the results bundle ({type(e).__name__}: {e}); starting in demo mode")
        return False
    _log(f"installed {len(written)} result files ({len(data) / 2**20:.1f} MB)")
    return True


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Pack the trained results into one zip for deployment.")
    ap.add_argument("command", choices=["pack"])
    ap.add_argument("--out", default=DEFAULT_BUNDLE)
    args = ap.parse_args()
    root = Paths().root
    out = Path(args.out).resolve()
    digest = pack(root, out)
    print(f"wrote {out} ({out.stat().st_size / 2**20:.1f} MB)")
    print(f"SHA-256: {digest}")
    print("Upload it as a release asset in a PRIVATE repo; set CUFFLESS_RESULTS_SHA256 to the value above.")


if __name__ == "__main__":
    main()
