#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 <miniXX> <job_id> [local_dir]" >&2
  exit 1
fi

HOST="$1"
JOB_ID="$2"
LOCAL_DIR="${3:-/home/zaytor/transformer_research/parameter-golf/research/representation_learning/fetched/$JOB_ID}"
REMOTE_DIR="~/jobs/$JOB_ID"

mkdir -p "$LOCAL_DIR"

scp "$HOST:$REMOTE_DIR/model_representation.npz" "$LOCAL_DIR/"
scp "$HOST:$REMOTE_DIR/"*.json "$LOCAL_DIR/" 2>/dev/null || true
scp "$HOST:$REMOTE_DIR/"*.txt "$LOCAL_DIR/" 2>/dev/null || true
