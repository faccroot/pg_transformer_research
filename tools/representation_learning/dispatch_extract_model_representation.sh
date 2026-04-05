#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DISPATCH="$HOME/cluster/dispatch.sh"
WRAPPER="$ROOT/run_representation_learning_job.py"
SCRIPT="$ROOT/tools/representation_learning/extract_model_representation.py"
REQUIREMENTS="$ROOT/requirements-representation-learning.txt"

HOST_ARGS=()
if [[ "${1:-}" == --host ]]; then
  [[ $# -ge 2 ]] || { echo "usage: $0 [--host miniXX] <args_json> [staged_file ...]" >&2; exit 1; }
  HOST_ARGS=(--host "$2")
  shift 2
fi

ARGS_JSON="${1:-}"
[[ -n "$ARGS_JSON" ]] || { echo "usage: $0 [--host miniXX] <args_json> [staged_file ...]" >&2; exit 1; }
shift || true

exec "$DISPATCH" "${HOST_ARGS[@]}" \
  "$WRAPPER" \
  "$SCRIPT" \
  "$ARGS_JSON" \
  --requirements \
  "$REQUIREMENTS" \
  "$ROOT/tools/representation_learning/model_adapter.py" \
  "$ROOT/tools/representation_learning/schemas.py" \
  "$ROOT/tools/representation_learning/concept_probes.py" \
  "$@"
