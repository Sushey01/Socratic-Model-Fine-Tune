#!/usr/bin/env python3
"""Upload dataset7b JSONL to Susu11/qwen-socratic-tutor (dataset repo, not the 7B model)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, login

ROOT = Path(__file__).resolve().parent


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


DIR = ROOT / "dataset7b"
DEFAULT_REPO = "Susu11/qwen-socratic-tutor"
FILES = (
    "socratic_dataset_v7_annotated.jsonl",
    "socratic_v7_train.jsonl",
    "socratic_v7_val.jsonl",
)


def main() -> None:
    _load_dotenv()
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    repo = (os.environ.get("HF_QWEN25_DATASET_REPO") or DEFAULT_REPO).strip()
    if not token:
        raise SystemExit("HF_TOKEN is empty; cannot upload dataset.")
    missing = [DIR / name for name in FILES if not (DIR / name).is_file()]
    if missing:
        raise SystemExit("Missing: " + ", ".join(str(p) for p in missing))
    login(token=token, add_to_git_credential=False)
    api = HfApi(token=token)
    api.create_repo(repo, exist_ok=True, repo_type="dataset", private=True)
    for name in FILES:
        path = DIR / name
        print(f"Uploading {path.name} ({path.stat().st_size} bytes) → {repo}/{name} ...", flush=True)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=name,
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
