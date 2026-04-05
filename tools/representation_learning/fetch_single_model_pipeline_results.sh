#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: $0 <miniXX> [remote_run_dir] [local_dir]" >&2
  exit 1
fi

HOST="$1"
REMOTE_RUN_DIR="${2:-~/transformer_research/parameter-golf/research/representation_learning/runs/qwen3_4b_v1}"
LOCAL_DIR="${3:-/home/zaytor/transformer_research/parameter-golf/research/representation_learning/runs/qwen3_4b_v1}"

mkdir -p "$LOCAL_DIR"

scp "$HOST:$REMOTE_RUN_DIR/calibration.jsonl" "$LOCAL_DIR/"
scp "$HOST:$REMOTE_RUN_DIR/calibration.jsonl.summary.json" "$LOCAL_DIR/"
scp "$HOST:$REMOTE_RUN_DIR/model_representation.npz" "$LOCAL_DIR/"
scp "$HOST:$REMOTE_RUN_DIR/model_representation.summary.json" "$LOCAL_DIR/"
scp "$HOST:$REMOTE_RUN_DIR/platonic_geometry.npz" "$LOCAL_DIR/"
scp "$HOST:$REMOTE_RUN_DIR/zero_shot_summary.json" "$LOCAL_DIR/"
scp "$HOST:$REMOTE_RUN_DIR/pipeline_summary.json" "$LOCAL_DIR/"
