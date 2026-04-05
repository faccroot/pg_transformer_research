#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 <report_json> <canonical_dim> <num_layers> <rep1> <rep2> [rep3 ...]" >&2
  exit 1
fi

REPORT_JSON="$1"
CANONICAL_DIM="$2"
NUM_LAYERS="$3"
shift 3

python3 /home/zaytor/transformer_research/parameter-golf/tools/representation_learning/compare_model_representations.py \
  "$REPORT_JSON" \
  "$@" \
  --canonical-dim "$CANONICAL_DIM" \
  --num-layers "$NUM_LAYERS"

python3 - <<'PY' "$REPORT_JSON"
import json
import sys
path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    report = json.load(handle)
print("best_mean_chunk_loss_model", report.get("best_mean_chunk_loss_model"))
print("concept_winners")
for concept, payload in sorted(report.get("concepts", {}).items()):
    print(f"  {concept}: {payload.get('best_model')}")
PY
