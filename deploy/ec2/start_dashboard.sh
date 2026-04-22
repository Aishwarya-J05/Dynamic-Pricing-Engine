#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/opt/dynamic-pricing-engine}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8501}"

cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"

exec streamlit run app/dashboard.py \
  --server.address "$HOST" \
  --server.port "$PORT" \
  --server.headless true
