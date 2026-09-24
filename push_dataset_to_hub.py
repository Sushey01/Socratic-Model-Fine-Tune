"""
Script to upload the local socratic_dataset.jsonl directly to Hugging Face Hub
Target repository: Susu11/socratic_idea_expansion
"""

import os
import json
import argparse
from datasets import Dataset
from huggingface_hub import HfApi, login

try:
    from kaggle_secrets import UserSecretsClient
    IN_KAGGLE = True
except ImportError:
    IN_KAGGLE = False


def main():
    parser = argparse.ArgumentParser(description="Upload Socratic dataset to Hugging Face Hub")
    parser.add_argument("--dataset_file", type=str, default="socratic_dataset.jsonl",
                        help="Path to the local JSONL dataset file")
    parser.add_argument("--repo_id", type=str, default="Susu11/socratic_idea_expansion",
                        help="Target Hugging Face Dataset repository ID")
    parser.add_argument("--private", action="store_true",
                        help="Set dataset repo as private")
    parser.add_argument("--hf_token", type=str, default="",
                        help="Hugging Face API token (optional if set in env or Kaggle secrets)")

    args = parser.parse_args()

    # 1. Hugging Face Login
    token = args.hf_token or os.getenv("HF_TOKEN")
    if not token and IN_KAGGLE:
        try:
            token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            pass

    if token:
        login(token=token, add_to_git_credential=True)
        print("✅ Logged in to Hugging Face Hub.")
    else:
        print("⚠️ No explicit token provided; assuming already logged in via huggingface-cli.")

    # 2. Load Local JSONL
    if not os.path.exists(args.dataset_file):
        raise FileNotFoundError(f"Could not find dataset file at: {args.dataset_file}")

    print(f"📖 Reading {args.dataset_file}...")
    records = []
    with open(args.dataset_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "messages" in data and isinstance(data["messages"], list):
                    records.append({"messages": data["messages"]})
            except json.JSONDecodeError as e:
                print(f"⚠️ Skipping invalid JSON at line {line_num}: {e}")

    dataset = Dataset.from_list(records)
    print(f"✅ Prepared {len(dataset)} conversation records.")

    # 3. Push to Hugging Face Hub
    print(f"🚀 Pushing dataset to Hugging Face Hub: {args.repo_id}...")
    dataset.push_to_hub(
        repo_id=args.repo_id,
        private=args.private,
    )
    print(f"🎉 Dataset successfully uploaded! View it at: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
