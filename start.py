#!/usr/bin/env python3
"""One-command pipeline: load .env, SFT, ScienceQA→W&B, push model to Hugging Face.

Does not use the `hf` CLI (broken on Git Bash). Run:

    python start.py
    uv run python start.py
"""

from __future__ import annotations

import argparse
import getpass
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"


def load_dotenv_file() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        _load_env_fallback(ENV_PATH)
        return
    load_dotenv(ENV_PATH, override=False)


def _load_env_fallback(path: Path) -> None:
    if not path.is_file():
        return
    raw = path.read_text(encoding="utf-8-sig")
    for line in raw.splitlines():
        line = line.strip().lstrip("\ufeff")
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = val


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def _need(name: str) -> bool:
    val = _env(name)
    return not val or val.startswith("YOUR_")


def _prompt_secret(name: str, message: str) -> str:
    if not sys.stdin.isatty():
        raise SystemExit(f"{name} missing in .env and stdin is not a terminal.")
    print(message, flush=True)
    value = getpass.getpass("").strip()
    if value:
        os.environ[name] = value
        _write_env_key(name, value)
        print(f"Wrote {name} into .env for next runs.")
    return value


def _write_env_key(key: str, value: str) -> None:
    try:
        from dotenv import set_key

        if not ENV_PATH.is_file():
            ENV_PATH.write_text("", encoding="utf-8")
        set_key(str(ENV_PATH), key, value, quote_mode="always")
        return
    except Exception:
        pass
    lines = []
    found = False
    if ENV_PATH.is_file():
        for line in ENV_PATH.read_text(encoding="utf-8-sig").splitlines():
            if line.startswith(f"{key}=") or line.startswith(f"export {key}="):
                lines.append(f'{key}="{value}"')
                found = True
            else:
                lines.append(line)
    if not found:
        lines.append(f'{key}="{value}"')
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_secrets() -> None:
    load_dotenv_file()
    os.environ.setdefault("HF_DATASET_REPO", "Susu11/socraticfinetune")
    os.environ.setdefault("WANDB_PROJECT", "socratic-phi3")
    if _need("HF_HUB_REPO"):
        os.environ["HF_HUB_REPO"] = "Susu11/socratic-phi3"

    if _env("WANDB_API_KEY"):
        print("Using WANDB_API_KEY from .env")
    else:
        if not _prompt_secret("WANDB_API_KEY", "WANDB_API_KEY not in .env. Paste W&B key (hidden), then Enter:"):
            raise SystemExit("WANDB_API_KEY is required. Put WANDB_API_KEY=... in .env with no # in front.")

    token = _env("HF_TOKEN") or _env("HUGGING_FACE_HUB_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        print("Using HF_TOKEN from .env")
    else:
        token = _prompt_secret(
            "HF_TOKEN",
            "HF_TOKEN not in .env. Paste Hugging Face write token (hidden), then Enter:",
        )
        if not token:
            raise SystemExit("HF_TOKEN is required. Put HF_TOKEN=hf_... in .env with no # in front.")
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token


def _python() -> list[str]:
    uv = shutil.which("uv")
    if uv:
        return [uv, "run", "python"]
    return [sys.executable]


def run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.check_call(args, cwd=str(ROOT))


def sync_deps() -> None:
    uv = shutil.which("uv")
    if uv:
        print("Installing Python dependencies with uv...")
        run([uv, "sync"])
        return
    print("uv not found; using current Python:", sys.executable)


def maybe_cuda_wheel() -> None:
    code = (
        "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
    )
    try:
        subprocess.check_call(_python() + ["-c", code], cwd=str(ROOT))
    except subprocess.CalledProcessError:
        print("PyTorch has no CUDA. Trying CUDA 12.4 wheel...")
        uv = shutil.which("uv")
        if uv:
            subprocess.call(
                [uv, "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cu124"],
                cwd=str(ROOT),
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT + ScienceQA/W&B + Hugging Face push")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--download-checkpoints", action="store_true")
    parser.add_argument("--gguf", action="store_true")
    ns = parser.parse_args()

    if ns.gguf:
        print(
            "GGUF is a follow-up after LoRA, not part of default train.\n"
            "1. Merge adapters in ./socratic_finetuned_model into microsoft/Phi-3-mini-4k-instruct\n"
            "2. llama.cpp convert_hf_to_gguf.py + quantize (e.g. Q4_K_M)\n"
            "3. Upload gguf/socratic-phi3-q4_k_m.gguf to HF_HUB_REPO"
        )
        return

    os.chdir(ROOT)
    ensure_secrets()
    sync_deps()

    print("Uploading JSONL via Hugging Face Python API (no hf CLI)...")
    run(_python() + [str(ROOT / "upload_dataset.py")])

    if shutil.which("nvidia-smi") is None:
        print(
            "nvidia-smi not found. Git Bash/Windows often cannot run bitsandbytes.\n"
            "Use WSL2 Ubuntu or a Linux GPU PC. Dataset upload already used Python, not hf CLI.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    maybe_cuda_wheel()

    if ns.download_checkpoints:
        repo = _env("HF_HUB_REPO")
        dest = ROOT / "socratic_finetuned_model"
        print(f"Downloading adapters from {repo} ...")
        run(
            _python()
            + [
                "-c",
                "from huggingface_hub import snapshot_download; "
                f"snapshot_download(repo_id={repo!r}, local_dir={str(dest)!r})",
            ]
        )

    train = [str(ROOT / "train.py"), "--push-to-hub", _env("HF_HUB_REPO")]
    if ns.fresh:
        train.append("--no-resume")
    print("Starting SFT + ScienceQA (W&B) + Hub push...")
    run(_python() + train)
    print(
        f"Done. Adapters: {ROOT / 'socratic_finetuned_model'} | "
        f"W&B: {_env('WANDB_PROJECT')} | Hub: {_env('HF_HUB_REPO')}"
    )


if __name__ == "__main__":
    main()
