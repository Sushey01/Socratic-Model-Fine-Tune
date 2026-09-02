#!/usr/bin/env bash
# Thin wrapper. Real pipeline is start.py (no Hugging Face CLI).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
if command -v uv >/dev/null 2>&1; then
  exec uv run python start.py "$@"
fi
if command -v python >/dev/null 2>&1; then
  exec python start.py "$@"
fi
if command -v python3 >/dev/null 2>&1; then
  exec python3 start.py "$@"
fi
echo "Need Python 3.10+ or uv. Install Python, then: python start.py" >&2
exit 1
