#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DISPATCH="$HOME/cluster/dispatch.sh"
WRAPPER="$ROOT/run_representation_learning_job.py"
PIPELINE="$ROOT/tools/representation_learning/run_single_model_pipeline.py"
DEFAULT_ARGS_JSON="$ROOT/research/iterations/templates/representation_learning_qwen3_single_model.args.example.json"
REQUIREMENTS="$ROOT/requirements-representation-learning.txt"

HOST_ARGS=()
if [[ "${1:-}" == --host ]]; then
  [[ $# -ge 2 ]] || { echo "usage: $0 [--host miniXX] [args_json]" >&2; exit 1; }
  HOST_ARGS=(--host "$2")
  shift 2
fi

ARGS_JSON="${1:-$DEFAULT_ARGS_JSON}"

exec "$DISPATCH" "${HOST_ARGS[@]}" \
  "$WRAPPER" \
  "$PIPELINE" \
  "$ARGS_JSON" \
  --requirements \
  "$REQUIREMENTS" \
  "$ROOT/tools/representation_learning/assemble_reasoning_core.py" \
  "$ROOT/tools/representation_learning/build_platonic_geometry.py" \
  "$ROOT/tools/representation_learning/calibration_set.py" \
  "$ROOT/tools/representation_learning/eval_zero_shot_assembly.py" \
  "$ROOT/tools/representation_learning/extract_model_representation.py" \
  "$ROOT/tools/representation_learning/model_adapter.py" \
  "$ROOT/tools/representation_learning/schemas.py" \
  "$ROOT/curriculum.py"
