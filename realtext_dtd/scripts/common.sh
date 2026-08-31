#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_DIR

ENV_FILE="${ENV_FILE:-$REPO_DIR/configs/realtext_paths.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
else
  echo "Config not found: $ENV_FILE" >&2
  echo "Create it from configs/realtext_paths.env.example or pass ENV_FILE=/path/to/env." >&2
  exit 1
fi

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required config: $name" >&2
    exit 1
  fi
}
