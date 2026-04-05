#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "tools/setup_mlx_env.sh is intended for Apple Silicon macOS hosts." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv-mlx}"

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
pip install -r requirements-mlx.txt

if [[ "${DOWNLOAD_DATA:-0}" == "1" ]]; then
  python data/cached_challenge_fineweb.py \
    --variant "${DOWNLOAD_VARIANT:-sp1024}" \
    --train-shards "${TRAIN_SHARDS:-80}"
fi

echo "MLX environment ready in $VENV_DIR"

