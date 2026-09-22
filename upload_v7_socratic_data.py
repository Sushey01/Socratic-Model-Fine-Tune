#!/usr/bin/env python3
"""Upload socratic_v7_final_v3.jsonl to dataset repo Susu11/v7_socratic_data."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, login

ROOT = Path(__file__).resolve().parent
DIR = ROOT / "dataset7b"
JSONL = DIR / "socratic_v7_final_v3.jsonl"
README = DIR / "v7_socratic_data_README.md"
DEFAULT_REPO = "Susu11/v7_socratic_data"


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
        return
    except ImportError:
        pass
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        s = line.strip().lstrip("\ufeff")
        if not s or s.startswith("#") or "=" not in s:
            continue
        if s.startswith("export "):
            s = s[7:].strip()
        key, _, val = s.partition("=")
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = val


def main() -> None:
    _load_dotenv()
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    repo = (os.environ.get("HF_V7_DATASET_REPO") or DEFAULT_REPO).strip()
    if not token:
        raise SystemExit("HF_TOKEN is empty; cannot upload dataset.")
    if not JSONL.is_file():
        raise SystemExit(f"Missing {JSONL}")
    login(token=token, add_to_git_credential=False)
    api = HfApi(token=token)
    api.create_repo(repo, exist_ok=True, repo_type="dataset", private=True)
    print(f"Uploading {JSONL.name} ({JSONL.stat().st_size} bytes) → {repo}/{JSONL.name} ...", flush=True)
    api.upload_file(
        path_or_fileobj=str(JSONL),
        path_in_repo=JSONL.name,
        repo_id=repo,
        repo_type="dataset",
    )
    if README.is_file():
        api.upload_file(
            path_or_fileobj=str(README),
            path_in_repo="README.md",
            repo_id=repo,
            repo_type="dataset",
        )
    print(f"Dataset: https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Dataset upload failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
