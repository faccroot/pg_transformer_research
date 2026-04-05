#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLUSTER_DIR="${CLUSTER_DIR:-$HOME/cluster}"
INVENTORY="${INVENTORY:-$CLUSTER_DIR/inventory.yml}"
REPO_URL="${REPO_URL:-https://github.com/openai/parameter-golf.git}"
REMOTE_ROOT="${REMOTE_ROOT:-~/transformer_research}"
REMOTE_REPO="${REMOTE_REPO:-$REMOTE_ROOT/parameter-golf}"
TRAIN_SHARDS="${TRAIN_SHARDS:-1}"
VARIANT="${VARIANT:-sp1024}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
DOWNLOAD_DATA="${DOWNLOAD_DATA:-1}"
SYNC_THEORY="${SYNC_THEORY:-0}"

MINIS=(mini01 mini02 mini03 mini04 mini05 mini06 mini07 mini08 mini09 mini10 mini11 mini12 mini13 mini14)
RSYNC_EXCLUDES=(
  "--exclude=.git/"
  "--exclude=.venv/"
  "--exclude=.venv-mlx/"
  "--exclude=__pycache__/"
  "--exclude=.mypy_cache/"
  "--exclude=.DS_Store"
  "--exclude=data/datasets/"
  "--exclude=data/tokenizers/"
  "--exclude=data/manifest.json"
  "--exclude=data/docs_selected.jsonl"
  "--exclude=data/docs_selected.source_manifest.json"
  "--exclude=logs/"
  "--exclude=.logs/"
  "--exclude=records/"
  "--exclude=final_model.pt"
  "--exclude=final_model.int8.ptz"
)
if [[ "$SYNC_THEORY" != "1" ]]; then
  RSYNC_EXCLUDES+=("--exclude=theory/")
fi

pick_agent_socket() {
  if [[ -n "${SSH_AUTH_SOCK:-}" ]] && SSH_AUTH_SOCK="$SSH_AUTH_SOCK" ssh-add -l >/dev/null 2>&1; then
    echo "$SSH_AUTH_SOCK"
    return
  fi
  local sock
  while IFS= read -r sock; do
    if SSH_AUTH_SOCK="$sock" ssh-add -l >/dev/null 2>&1; then
      echo "$sock"
      return
    fi
  done < <(ls -t /tmp/ssh-*/agent.* 2>/dev/null || true)
  return 1
}

SSH_AUTH_SOCK="${SSH_AUTH_SOCK:-$(pick_agent_socket || true)}"
if [[ -z "${SSH_AUTH_SOCK:-}" ]]; then
  echo "No loaded SSH agent found." >&2
  exit 1
fi
export SSH_AUTH_SOCK

echo "Using SSH_AUTH_SOCK=$SSH_AUTH_SOCK"
echo "Verifying Mini reachability..."
ansible -i "$INVENTORY" online -m ping >/dev/null

echo "Creating repo root and cloning upstream repo if absent..."
ansible -i "$INVENTORY" online -m shell -a \
  "mkdir -p $REMOTE_ROOT && if [ ! -d $REMOTE_REPO/.git ]; then git clone $REPO_URL $REMOTE_REPO; else echo already_present; fi"

echo "Syncing local working tree to Minis..."
for host in "${MINIS[@]}"; do
  echo "  -> $host"
  rsync -az "${RSYNC_EXCLUDES[@]}" -e "ssh" "$ROOT/" "$host:$REMOTE_REPO/"
done

if [[ "$INSTALL_DEPS" == "1" ]]; then
  echo "Installing MLX-side Python dependencies..."
  ansible -i "$INVENTORY" online -m shell -a \
    "cd $REMOTE_REPO && /opt/homebrew/bin/python3 -m pip install -r requirements-mlx.txt tiktoken"
fi

if [[ "$DOWNLOAD_DATA" == "1" ]]; then
  echo "Bootstrapping cached challenge data variant=$VARIANT train_shards=$TRAIN_SHARDS ..."
  ansible -i "$INVENTORY" online -m shell -a \
    "cd $REMOTE_REPO && /opt/homebrew/bin/python3 data/cached_challenge_fineweb.py --variant $VARIANT --train-shards $TRAIN_SHARDS"
fi

echo "Verifying parameter-golf imports on Minis..."
ansible -i "$INVENTORY" online -m shell -a \
  "cd $REMOTE_REPO && /opt/homebrew/bin/python3 - <<'PY'
mods = ['mlx', 'sentencepiece', 'datasets', 'huggingface_hub', 'tiktoken']
for name in mods:
    __import__(name)
print('imports_ok')
PY"

echo "Bootstrap complete. Repo path on each Mini: $REMOTE_REPO"
