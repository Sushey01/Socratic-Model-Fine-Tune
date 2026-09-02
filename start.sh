#!/usr/bin/env bash
# One command on the training PC: secrets → SFT → ScienceQA → W&B → Hugging Face.
# Usage:  bash start.sh
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
      echo "  (default)  prompt for missing W&B/HF keys, SFT, ScienceQA→W&B, push model to Hub"
      echo "  --fresh    ignore existing checkpoints and start a new run"
      echo "  --download-checkpoints  pull HF_HUB_REPO then continue training"
      echo "  --gguf     after fine-tune: print merge/quantize/upload steps"
      exit 0
      ;;
  esac
done

if [[ "$GGUF" -eq 1 ]]; then
  cat <<'EOF'
GGUF is a follow-up after LoRA training, not part of the default train command.

1. Merge LoRA adapters in ./socratic_finetuned_model into microsoft/Phi-3-mini-4k-instruct.
2. Convert with llama.cpp convert_hf_to_gguf.py and quantize (e.g. Q4_K_M).
3. Upload gguf/socratic-phi3-q4_k_m.gguf to the same HF_HUB_REPO.

Keep using bash start.sh for the full SFT + ScienceQA + W&B + Hub pipeline.
EOF
  exit 0
fi

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

export HF_DATASET_REPO="${HF_DATASET_REPO:-Susu11/socraticfinetune}"
if [[ -z "${HF_HUB_REPO:-}" || "${HF_HUB_REPO}" == "YOUR_HF_USER/socratic-phi3" ]]; then
  export HF_HUB_REPO="Susu11/socratic-phi3"
fi
export WANDB_PROJECT="${WANDB_PROJECT:-socratic-phi3}"

hf_token() {
  echo "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
}

ask_hidden() {
  local prompt="$1"
  local value=""
  if [[ -t 0 ]]; then
    read -r -s -p "$prompt" value || true
    echo
  fi
  printf '%s' "$value"
}

ask_line() {
  local prompt="$1"
  local value=""
  if [[ -t 0 ]]; then
    read -r -p "$prompt" value || true
  fi
  printf '%s' "$value"
}

persist_env() {
  umask 077
  local tmp
  tmp="$(mktemp)"
  {
    echo "# local secrets — never commit"
    echo "HF_DATASET_REPO=${HF_DATASET_REPO}"
    echo "HF_HUB_REPO=${HF_HUB_REPO}"
    echo "WANDB_PROJECT=${WANDB_PROJECT}"
    if [[ -n "${WANDB_API_KEY:-}" ]]; then
      echo "WANDB_API_KEY=${WANDB_API_KEY}"
    fi
    if [[ -n "$(hf_token)" ]]; then
      echo "HF_TOKEN=$(hf_token)"
    fi
  } >"$tmp"
  mv "$tmp" "$ROOT/.env"
}

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  if grep -qE '^[[:space:]]*#[[:space:]]*WANDB_API_KEY=' "$ROOT/.env" 2>/dev/null; then
    echo "WANDB_API_KEY in .env is commented out. Enter the live key (or uncomment the line)."
  fi
  WANDB_API_KEY="$(ask_hidden "Weights & Biases API key (wandb.ai/authorize, hidden): ")"
  export WANDB_API_KEY
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is required for the full pipeline (ScienceQA charts on W&B)." >&2
  exit 1
fi

if [[ -z "$(hf_token)" ]]; then
  if grep -qE '^[[:space:]]*#[[:space:]]*HF_TOKEN=' "$ROOT/.env" 2>/dev/null; then
    echo "HF_TOKEN in .env is commented out. Enter the live write token (or uncomment the line)."
  fi
  HF_TOKEN="$(ask_hidden "Hugging Face write token (huggingface.co/settings/tokens, hidden): ")"
  export HF_TOKEN
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi
if [[ -z "$(hf_token)" ]]; then
  echo "HF_TOKEN is required to upload the dataset and push the fine-tuned model." >&2
  exit 1
fi

if [[ -z "${HF_HUB_REPO:-}" ]]; then
  HF_HUB_REPO="$(ask_line "Hugging Face model repo [Susu11/socratic-phi3]: ")"
  HF_HUB_REPO="${HF_HUB_REPO:-Susu11/socratic-phi3}"
  export HF_HUB_REPO
fi

persist_env
echo "Saved credentials to .env (gitignored). Next time bash start.sh will not ask again."

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
  echo "Uploading JSONL to dataset ${repo}..."
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

echo "Starting SFT + ScienceQA (W&B) + Hub push..."
uv run python train.py "${TRAIN_ARGS[@]}"
echo "Done. Adapters: ./socratic_finetuned_model  |  W&B project: ${WANDB_PROJECT}  |  Hub: ${HF_HUB_REPO}"
