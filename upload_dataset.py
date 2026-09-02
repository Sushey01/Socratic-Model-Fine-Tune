#!/usr/bin/env python3
"""Upload training JSONL to a Hugging Face dataset repo (no hf CLI)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, login

ROOT = Path(__file__).resolve().parent


def main() -> None:
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    repo = (os.environ.get("HF_DATASET_REPO") or "Susu11/socraticfinetune").strip()
    if not token:
        raise SystemExit("HF_TOKEN is empty; cannot upload dataset.")
    login(token=token, add_to_git_credential=False)
    api = HfApi(token=token)
    api.create_repo(repo, exist_ok=True, repo_type="dataset", private=True)
    jsonl = ROOT / "socratic_train.jsonl"
    if not jsonl.is_file():
        raise SystemExit(f"Missing {jsonl}")
    print(f"Uploading {jsonl.name} to {repo} ...")
    api.upload_file(
        path_or_fileobj=str(jsonl),
        path_in_repo=jsonl.name,
        repo_id=repo,
        repo_type="dataset",
    )
    extra = ROOT / "socratic_train_data.jsonl"
    if extra.is_file():
        api.upload_file(
            path_or_fileobj=str(extra),
            path_in_repo=extra.name,
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
