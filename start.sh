#!/usr/bin/env bash
# All-in-one: read .env, install tools, upload dataset, train, push adapters.
# Usage after clone:  bash start.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

FRESH=0
DOWNLOAD=0
GGUF=0
for arg in "$@"; do
  case "$arg" in
    --fresh) FRESH=1 ;;
    --download-checkpoints) DOWNLOAD=1 ;;
    --gguf) GGUF=1 ;;
    -h|--help)
      echo "Usage: bash start.sh [--fresh] [--download-checkpoints] [--gguf]"
      echo "  (default)  read .env (no hf auth login), upload JSONL, train, upload adapters + latest checkpoint"
      echo "  --fresh    ignore existing checkpoints and start a new run"
      echo "  --download-checkpoints  pull HF_HUB_REPO (adapters + latest checkpoint) then train"
      echo "  --gguf     after fine-tune: print merge/quantize/upload steps (not run during train)"
      exit 0
      ;;
  esac
done

if [[ "$GGUF" -eq 1 ]]; then
  cat <<'EOF'
GGUF is a follow-up after LoRA training, not part of the default train command.

1. Merge LoRA adapters in ./socratic_finetuned_model into microsoft/Phi-3-mini-4k-instruct (needs extra RAM/VRAM).
2. Convert with llama.cpp convert_hf_to_gguf.py and quantize (e.g. Q4_K_M).
3. Upload gguf/socratic-phi3-q4_k_m.gguf to the same HF_HUB_REPO.

Keep using bash start.sh for train/resume. Add GGUF conversion here later when a college run has finished.
EOF
  exit 0
fi

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
else
  echo "Missing .env. Create it next to start.sh with HF_TOKEN, HF_DATASET_REPO, and HF_HUB_REPO (no # in front of HF_TOKEN)." >&2
  exit 1
fi

export HF_DATASET_REPO="${HF_DATASET_REPO:-Susu11/socraticfinetune}"
if [[ -z "${HF_HUB_REPO:-}" || "${HF_HUB_REPO}" == "YOUR_HF_USER/socratic-phi3" ]]; then
  export HF_HUB_REPO="Susu11/socratic-phi3"
fi

hf_token() {
  echo "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
}

if [[ -z "$(hf_token)" ]]; then
  if grep -qE '^[[:space:]]*#[[:space:]]*HF_TOKEN=' "$ROOT/.env"; then
    echo "HF_TOKEN in .env is commented out (starts with #). Remove the # so start.sh can read it." >&2
  else
    echo "HF_TOKEN is not set in .env. Add: HF_TOKEN=hf_... (no quotes needed unless the value has spaces)." >&2
  fi
  exit 1
fi

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv installed but not on PATH. Open a new terminal or add ~/.local/bin to PATH." >&2
    exit 1
  fi
}

ensure_hf_cli() {
  export PATH="${HOME}/.local/bin:${PATH}"
  if command -v hf >/dev/null 2>&1; then
    return
  fi
  echo "Installing Hugging Face CLI..."
  curl -LsSf https://hf.co/cli/install.sh | bash
  export PATH="${HOME}/.local/bin:${PATH}"
  if ! command -v hf >/dev/null 2>&1; then
    echo "hf CLI not on PATH. Open a new terminal or add ~/.local/bin to PATH." >&2
    exit 1
  fi
}

upload_dataset() {
  local repo="${HF_DATASET_REPO}"
  local token
  token="$(hf_token)"
  echo "Uploading JSONL to dataset ${repo} using HF_TOKEN from .env (not hf auth login)..."
  hf repos create "$repo" --type dataset --exist-ok --token "$token"
  hf upload "$repo" "$ROOT/socratic_train.jsonl" --repo-type=dataset --token "$token"
  if [[ -f "$ROOT/socratic_train_data.jsonl" ]]; then
    hf upload "$repo" "$ROOT/socratic_train_data.jsonl" --repo-type=dataset --token "$token"
  fi
}

ensure_uv
ensure_hf_cli
upload_dataset

echo "Installing Python dependencies with uv..."
uv sync

if command -v nvidia-smi >/dev/null 2>&1; then
  if ! uv run python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "PyTorch has no CUDA. Installing CUDA 12.4 wheel..."
    uv pip install torch --index-url https://download.pytorch.org/whl/cu124
  fi
else
  echo "nvidia-smi not found. Training needs an NVIDIA GPU." >&2
  exit 1
fi

if [[ "$DOWNLOAD" -eq 1 ]]; then
  echo "Downloading adapters + latest checkpoint from ${HF_HUB_REPO}..."
  uv run python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${HF_HUB_REPO}', local_dir='${ROOT}/socratic_finetuned_model')"
fi

TRAIN_ARGS=(--push-to-hub "${HF_HUB_REPO}")
if [[ "$FRESH" -eq 1 ]]; then
  TRAIN_ARGS+=(--no-resume)
fi

echo "Starting training (resumes latest checkpoint unless --fresh)..."
uv run python train.py "${TRAIN_ARGS[@]}"
echo "Done. Adapters/checkpoints are in ./socratic_finetuned_model"
