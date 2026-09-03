#!/usr/bin/env python3
"""One-command pipeline: load .env, SFT, ScienceQA→W&B, push model to Hugging Face.

Does not use the `hf` CLI (broken on Git Bash). Run:

    python start.py
    uv run python start.py
    powershell -ExecutionPolicy Bypass -File .\\start.ps1
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
# Separate from the .venv that launched `uv run python start.py`. Windows will
# not let uv delete `.venv\Scripts` while this process is still using it.
TRAIN_VENV = ROOT / ".venv-train"


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


def _uv() -> str | None:
    found = shutil.which("uv")
    if found:
        return found
    for extra in (
        Path.home() / ".local" / "bin",
        Path.home() / ".cargo" / "bin",
        Path(os.environ.get("USERPROFILE") or "") / ".local" / "bin",
        Path(os.environ.get("LOCALAPPDATA") or "") / "uv",
    ):
        if not extra:
            continue
        os.environ["PATH"] = str(extra) + os.pathsep + os.environ.get("PATH", "")
        exe = extra / ("uv.exe" if os.name == "nt" else "uv")
        if exe.is_file():
            return str(exe)
    return shutil.which("uv")


def _venv_root() -> Path:
    if os.environ.get("UV_PROJECT_ENVIRONMENT"):
        return Path(os.environ["UV_PROJECT_ENVIRONMENT"])
    return ROOT / ".venv"


def _site_packages() -> Path:
    venv = _venv_root()
    win = venv / "Lib" / "site-packages"
    if win.is_dir():
        return win
    lib = venv / "lib"
    if lib.is_dir():
        matches = sorted(lib.glob("python3.*/site-packages"))
        if matches:
            return matches[-1]
    return win


def _use_train_venv() -> None:
    os.environ["UV_PROJECT_ENVIRONMENT"] = str(TRAIN_VENV)


def _managed_python() -> str | None:
    uv = _uv()
    if not uv:
        return None
    try:
        out = subprocess.check_output(
            [uv, "python", "find", "3.12"],
            cwd=str(ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    path = out.strip().splitlines()[-1].strip() if out.strip() else ""
    return path or None


def _python() -> list[str]:
    managed = os.environ.get("SOCRATIC_PYTHON") or _managed_python()
    site = _site_packages()
    if managed and site.is_dir() and Path(managed).is_file():
        os.environ["SOCRATIC_PYTHON"] = managed
        os.environ["VIRTUAL_ENV"] = str(_venv_root())
        pp = str(site)
        cur = os.environ.get("PYTHONPATH", "")
        if pp not in cur.split(os.pathsep):
            os.environ["PYTHONPATH"] = pp + (os.pathsep + cur if cur else "")
        return [managed]
    uv = _uv()
    if uv:
        if sys.version_info[:2] != (3, 12):
            _use_train_venv()
            return [uv, "run", "--python", "3.12", "python"]
        return [uv, "run", "python"]
    return [sys.executable]


def _looks_like_python_cmd(args: list[str]) -> bool:
    head = Path(args[0]).name.lower()
    if head in {"python", "python.exe", "python3", "python3.exe"}:
        return True
    return "uv" in head and any(a == "run" for a in args[1:4])


def run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    proc = subprocess.run(args, cwd=str(ROOT))
    if proc.returncode == 0:
        return
    err = proc.returncode
    managed = _managed_python()
    if os.name == "nt" and managed and _looks_like_python_cmd(args) and args[:1] != [managed]:
        print("Retrying with uv-managed Python (Application Control may block .venv\\Scripts\\python.exe)...")
        rest = args[1:]
        if rest[:1] == ["python"]:
            rest = rest[1:]
        elif len(args) >= 3 and args[1] == "run":
            i = 2
            if i < len(args) and args[i] == "--python":
                i += 2
            if i < len(args) and args[i] == "python":
                i += 1
            rest = args[i:]
        retry = [managed, *rest]
        print("+", " ".join(retry), flush=True)
        proc = subprocess.run(retry, cwd=str(ROOT))
        if proc.returncode == 0:
            return
        err = proc.returncode
    raise SystemExit(err) from None


def _install_uv() -> str | None:
    existing = _uv()
    if existing:
        return existing
    print("uv not found; installing...")
    try:
        if os.name == "nt":
            subprocess.check_call(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    "irm https://astral.sh/uv/install.ps1 | iex",
                ]
            )
        else:
            subprocess.check_call(["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"])
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(
            "Could not install uv automatically. See https://docs.astral.sh/uv/getting-started/installation/",
            file=sys.stderr,
        )
        return None
    return _uv()


def bootstrap_runtime() -> None:
    uv = _install_uv()
    if not uv:
        print("Continuing with", sys.executable)
        return
    print("Ensuring Python 3.12 is installed...")
    subprocess.call([uv, "python", "install", "3.12"], cwd=str(ROOT))
    if sys.version_info[:2] != (3, 12):
        print(
            f"Launcher is Python {sys.version_info.major}.{sys.version_info.minor}; "
            f"train env will be {TRAIN_VENV.name}"
        )
        _use_train_venv()


def sync_deps() -> None:
    uv = _uv()
    if not uv:
        print("uv not found; using current Python:", sys.executable)
        return
    # Never pin+recreate the in-use .venv (Windows Access is denied / App Control).
    py = ["--python", "3.12"] if sys.version_info[:2] != (3, 12) else []
    if sys.version_info[:2] != (3, 12):
        _use_train_venv()
        print(f"Syncing Python 3.12 env at {TRAIN_VENV.name} (CUDA torch from PyTorch index)...")
    else:
        print("Python 3.12 already; syncing current environment (CUDA torch from PyTorch index)...")
    run([uv, "sync", *py])


def nvidia_smi() -> str | None:
    found = shutil.which("nvidia-smi")
    if found:
        return found
    for p in (
        Path(r"C:\Windows\System32\nvidia-smi.exe"),
        Path(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"),
    ):
        if p.is_file():
            return str(p)
    return None


def _torch_probe() -> str:
    r = subprocess.run(
        _python()
        + [
            "-c",
            "import torch; print(torch.__version__); print(torch.version.cuda); "
            "print('yes' if torch.cuda.is_available() else 'no')",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    return (r.stdout or r.stderr or "").strip()


def cuda_ok() -> bool:
    probe = _torch_probe()
    print("torch:", " | ".join(probe.splitlines()) or "(could not import torch)")
    return probe.splitlines()[-1:] == ["yes"]


def maybe_cuda_wheel() -> None:
    if cuda_ok():
        print("PyTorch CUDA is available.")
        return
    uv = _uv()
    if not uv:
        print("Install uv, then re-run so CUDA torch can be installed.", file=sys.stderr)
        return
    print("PyTorch has no CUDA. Reinstalling a CUDA wheel (PyPI torch on Windows is CPU-only)...")
    extra = ["--python", "3.12"] if sys.version_info[:2] != (3, 12) else []
    subprocess.call([uv, "sync", *extra, "--reinstall-package", "torch"], cwd=str(ROOT))
    if cuda_ok():
        print("PyTorch CUDA is available after uv sync.")
        return
    backends = ["auto", "cu124", "cu126", "cu118"]
    for backend in backends:
        print(f"Trying torch backend {backend}...")
        subprocess.call(
            [uv, "pip", "install", "--reinstall", "torch", "--torch-backend", backend],
            cwd=str(ROOT),
        )
        if cuda_ok():
            print("PyTorch CUDA is available after wheel install.")
            return
    print(
        "Still no CUDA. nvidia-smi can work while torch stays CPU if the wheel is +cpu.\n"
        "After git pull, in this folder:\n"
        "  .\\start.ps1\n"
        "If Application Control blocks python.exe (error 4551), allow this folder or use WSL2.\n"
        "If nvidia-smi works but torch.cuda is still False, use WSL2 Ubuntu or a Linux GPU machine.",
        file=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT + ScienceQA/W&B + Hugging Face push")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--download-checkpoints", action="store_true")
    parser.add_argument("--eval", action="store_true", help="ScienceQA only on saved adapters (no SFT)")
    parser.add_argument("--gguf", action="store_true")
    ns = parser.parse_args()

    os.chdir(ROOT)
    if ns.gguf:
        ensure_secrets()
        bootstrap_runtime()
        sync_deps()
        print("Merging adapters, converting GGUF, uploading to Hub...")
        run(_python() + [str(ROOT / "export_gguf.py")])
        return
    if ns.eval:
        ensure_secrets()
        bootstrap_runtime()
        sync_deps()
        maybe_cuda_wheel()
        if nvidia_smi() is None and not cuda_ok():
            raise SystemExit("Need NVIDIA GPU for eval.")
        print("ScienceQA eval only (no training)...")
        run(_python() + [str(ROOT / "run_eval.py")])
        return
    ensure_secrets()
    bootstrap_runtime()
    sync_deps()

    print("Uploading JSONL via Hugging Face Python API (no hf CLI)...")
    run(_python() + [str(ROOT / "upload_dataset.py")])

    maybe_cuda_wheel()
    if nvidia_smi() is None and not cuda_ok():
        print(
            "nvidia-smi not found and PyTorch has no CUDA. Install NVIDIA drivers, or train in WSL2/Linux.",
            file=sys.stderr,
        )
        raise SystemExit(1)

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
