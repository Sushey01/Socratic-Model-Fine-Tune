#!/usr/bin/env python3
"""Merge QLoRA adapters, convert to GGUF, upload to HF_HUB_REPO under gguf/."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi, login
from peft import PeftModel
from transformers import AutoTokenizer

from train import DEFAULT_MODEL, DEFAULT_OUTPUT, from_pretrained_phi3

ROOT = Path(__file__).resolve().parent
MERGED = ROOT / "merged_model"
LLAMA_CPP = ROOT / ".llama.cpp"
GGUF_DIR = ROOT / "gguf"


def _run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.check_call(args, cwd=str(ROOT))


def merge(adapter: Path, base: str) -> Path:
    print(f"Merging {adapter} onto {base} (needs RAM)...", flush=True)
    tok = AutoTokenizer.from_pretrained(
        str(adapter) if (adapter / "tokenizer_config.json").is_file() else base,
        trust_remote_code=True,
    )
    try:
        model = from_pretrained_phi3(
            base,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
    except TypeError:
        model = from_pretrained_phi3(
            base,
            dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
    model = PeftModel.from_pretrained(model, str(adapter))
    model = model.merge_and_unload()
    MERGED.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MERGED), safe_serialization=True)
    tok.save_pretrained(str(MERGED))
    print(f"Merged weights in {MERGED}", flush=True)
    return MERGED


def ensure_llama_cpp() -> Path:
    convert = LLAMA_CPP / "convert_hf_to_gguf.py"
    if convert.is_file():
        return convert
    print("Cloning llama.cpp (convert script only)...", flush=True)
    _run(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp", str(LLAMA_CPP)])
    if not convert.is_file():
        raise SystemExit(f"Missing {convert}")
    return convert


def convert_gguf(merged: Path) -> Path:
    convert = ensure_llama_cpp()
    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    out = GGUF_DIR / "socratic-phi3-q8_0.gguf"
    for outtype, name in (("q8_0", "socratic-phi3-q8_0.gguf"), ("f16", "socratic-phi3-f16.gguf")):
        dest = GGUF_DIR / name
        cmd = [sys.executable, str(convert), str(merged), "--outfile", str(dest), "--outtype", outtype]
        try:
            _run(cmd)
            print(f"Wrote {dest}", flush=True)
            return dest
        except subprocess.CalledProcessError:
            print(f"convert --outtype {outtype} failed; trying next.", flush=True)
    raise SystemExit("GGUF convert failed. Install git and retry; merge folder is kept.")


def upload(path: Path, repo_id: str) -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    if not token:
        raise SystemExit("HF_TOKEN required to upload GGUF.")
    login(token=token, add_to_git_credential=False)
    api = HfApi()
    api.create_repo(repo_id, exist_ok=True, repo_type="model", private=True)
    dest = f"gguf/{path.name}"
    print(f"Uploading {path} -> {repo_id}/{dest}", flush=True)
    api.upload_file(path_or_fileobj=str(path), path_in_repo=dest, repo_id=repo_id, repo_type="model")
    print(f"GGUF: https://huggingface.co/{repo_id}/blob/main/{dest}", flush=True)


def main() -> None:
    adapter = Path(os.environ.get("ADAPTER_DIR") or DEFAULT_OUTPUT)
    if not adapter.is_dir():
        raise SystemExit(f"No adapters at {adapter}.")
    base = os.environ.get("BASE_MODEL", DEFAULT_MODEL)
    repo = (os.environ.get("HF_HUB_REPO") or "").strip()
    if not repo:
        raise SystemExit("Set HF_HUB_REPO.")
    skip_upload = os.environ.get("GGUF_SKIP_UPLOAD", "").strip() in {"1", "true", "yes"}
    merge(adapter, base)
    gguf = convert_gguf(MERGED)
    if skip_upload:
        print("GGUF_SKIP_UPLOAD set; not pushing to Hub.")
        return
    upload(gguf, repo)
    shutil.rmtree(MERGED, ignore_errors=True)


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
