"""Deploy the app to a Hugging Face Space (Docker) as one clean snapshot, without git history.

    pip install -r requirements.txt          # includes huggingface_hub
    hf auth login                            # or set HF_TOKEN; needs a token with write access
    python scripts/deploy_hf.py --space <user>/cufflessai --dry-run          # list what would be sent
    python scripts/deploy_hf.py --space <user>/cufflessai --create           # first time: create + upload code
    python scripts/deploy_hf.py --space <user>/cufflessai --with-results     # also upload the trained results

Why a script instead of `git push`: the Space is its own git repo, and Hugging Face rejects pushes that
contain binary files outside its large-file storage. This repo's history has PNGs, .npz files and the
face-detector model, so a plain push fails. Uploading through the Hub API sends only what the app needs
and stores binaries correctly.

The Space rebuilds the Docker image after each upload (a few minutes). Code files that no longer exist
here are removed from the Space; uploaded result files are never deleted by a code-only deploy.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE_README = ROOT / "deploy" / "huggingface" / "README.md"

# Code and config the container needs (git-tracked or new-but-not-ignored files only).
CODE_FILES = ["Dockerfile", ".dockerignore", "requirements-app.txt", "scripts/start.py"]
CODE_DIRS = ["backend/", "src/", "frontend/"]
CODE_EXCLUDE = ("frontend/node_modules/", "frontend/dist/", "__pycache__/")

# Trained results (git-ignored). Only sent with --with-results.
RESULT_FILES = [
    "artifacts/resnet/resnet_bp_v3.pt",
    "artifacts/baseline/mean_baseline.json",
    "results/best_model.json",
    "results/metrics.csv",
    "results/residuals.csv",
    "data/test_samples.npz",
]
RESULT_PREFIXES = ("artifacts/", "results/", "data/")


def code_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                         cwd=ROOT, check=True, capture_output=True, text=True).stdout.splitlines()
    keep = []
    for f in out:
        f = f.replace("\\", "/")
        if not (ROOT / f).is_file():          # tracked but deleted locally
            continue
        if f in CODE_FILES or (f.startswith(tuple(CODE_DIRS)) and not any(x in f for x in CODE_EXCLUDE)):
            keep.append(f)
    missing = [f for f in CODE_FILES if f not in keep]
    if missing:
        sys.exit(f"missing required files: {missing}")
    return sorted(keep)


def result_files() -> list[str]:
    missing = [f for f in RESULT_FILES if not (ROOT / f).is_file()]
    if missing:
        sys.exit("--with-results needs these files (run scripts/evaluate.py or get them from a teammate):\n  "
                 + "\n  ".join(missing))
    return list(RESULT_FILES)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", required=True, help="Space id, e.g. your-username/cufflessai")
    ap.add_argument("--create", action="store_true", help="create the Space first if it doesn't exist")
    ap.add_argument("--private", action="store_true", help="with --create: make the Space private")
    ap.add_argument("--with-results", action="store_true", help="also upload artifacts/, results/ and the samples")
    ap.add_argument("--dry-run", action="store_true", help="list the files and exit without uploading")
    args = ap.parse_args()

    files = code_files()
    results = result_files() if args.with_results else []
    size = sum((ROOT / f).stat().st_size for f in files + results)
    summary = f"{len(files)} code files + README.md" + (f" + {len(results)} result files" if results else "")
    print(f"{summary} ({size / 1e6:.1f} MB) -> spaces/{args.space}")
    if args.dry_run:
        for f in files + results:
            print("  " + f)
        print("  README.md  (from deploy/huggingface/README.md)")
        return

    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    api = HfApi()
    if args.create:
        api.create_repo(args.space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
    try:
        existing = set(api.list_repo_files(args.space, repo_type="space"))
    except RepositoryNotFoundError:
        sys.exit(f"Space {args.space} not found. Create it with --create, or check the name and your token.")

    if results:
        public = not api.space_info(args.space).private
        if public:
            print("Note: this Space is public, so anyone can download the uploaded result files from its Files tab.")

    ops = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=str(ROOT / f)) for f in files + results]
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(SPACE_README)))
    # remove code files deleted here; never touch results or the Space's own .gitattributes
    sent = {op.path_in_repo for op in ops}
    stale = [f for f in existing if f not in sent and not f.startswith(RESULT_PREFIXES) and f != ".gitattributes"]
    ops += [CommitOperationDelete(path_in_repo=f) for f in stale]

    commit = api.create_commit(args.space, repo_type="space", operations=ops,
                               commit_message="Deploy from GitHub" + (" (with results)" if results else ""))
    print(f"Uploaded. Removed {len(stale)} stale file(s). The Space is rebuilding: "
          f"https://huggingface.co/spaces/{args.space}\nCommit: {commit.commit_url}")


if __name__ == "__main__":
    main()
