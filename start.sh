#!/usr/bin/env bash
# Install uv + Python 3.12 if missing, then run start.py (no Hugging Face CLI).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not on PATH. Install from https://docs.astral.sh/uv/ then re-run." >&2
  exit 1
fi

uv python install 3.12
uv python pin 3.12
exec uv run --python 3.12 python start.py "$@"
