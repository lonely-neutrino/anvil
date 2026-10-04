#!/usr/bin/env bash
# Create the Quest virtualenv and install the CUDA 12.8 PyTorch wheel.
#
# Set QUEST_PYTHON_MODULE and/or QUEST_CUDA_MODULE if Quest's module catalog
# requires them. The module names vary, so discover them with `module spider`
# before running this script.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if ! type module >/dev/null 2>&1; then
    for init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
        if [[ -r "$init" ]]; then
            # shellcheck disable=SC1090
            source "$init"
            break
        fi
    done
fi

if [[ -n "${QUEST_PYTHON_MODULE:-}" ]]; then
    module load "$QUEST_PYTHON_MODULE"
fi
if [[ -n "${QUEST_CUDA_MODULE:-}" ]]; then
    module load "$QUEST_CUDA_MODULE"
fi
if [[ -n "${QUEST_JAVA_MODULE:-}" ]]; then
    module load "$QUEST_JAVA_MODULE"
fi

PYTHON="${PYTHON:-python3}"
ENV_DIR="${ENV_DIR:-$ROOT/.venv}"
TORCH_VERSION="${TORCH_VERSION:-2.9.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

command -v "$PYTHON" >/dev/null || {
    echo "Python executable not found: $PYTHON" >&2
    exit 1
}

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    echo "[quest] creating $ENV_DIR"
    "$PYTHON" -m venv "$ENV_DIR"
fi

"$ENV_DIR/bin/python" -m pip install --upgrade pip uv
"$ENV_DIR/bin/uv" sync --locked --no-install-package torch
"$ENV_DIR/bin/uv" pip install \
    --python "$ENV_DIR/bin/python" \
    --index-url "$TORCH_INDEX_URL" \
    "torch==$TORCH_VERSION"

check_args=()
if [[ -n "${FORGE_DIR:-}" ]]; then
    check_args+=(--forge-dir "$FORGE_DIR")
fi
"$ENV_DIR/bin/python" scripts/quest/check_environment.py "${check_args[@]}"

echo "[quest] environment ready: $ENV_DIR"
