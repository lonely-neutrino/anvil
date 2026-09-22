#!/usr/bin/env bash
# Train the custom mono-green BC model, serve its checkpoint, and run mirrored
# model-vs-heuristic evaluation arms.
#
# This uses the existing virtualenv directly. Do not change it to plain
# `uv run`, because this checkout uses a ROCm-specific PyTorch installation.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Edit these values when switching to another corpus or deck.
PYTHON="$ROOT/.venv/bin/python"
STORE="data/trajectories/mono-green-heur-20000-20260902-234934"
DECK="monoGreenStompy.dck"
FORMAT="Constructed"
EMBED="data/embeddings/mono-green-stompy-v1-bge-m3"
POOL_MANIFEST="data/pool/custom/mono-green-stompy-v1.json"
POOL_VERSION="mono-green-stompy-v1"
TRAIN_OUT="data/training/mono-green-auto-$(date +%Y%m%d-%H%M%S)"
TRAIN_BATCH=32
TRAIN_STEPS=200000
TRAIN_WORKERS=1
EVAL_GAMES=400
EVAL_WORKERS=4
PORT=50070
SEED_BASE="$(date +%Y%m%d)"

[[ -x "$PYTHON" ]] || { echo "missing $PYTHON" >&2; exit 1; }
[[ -d "$STORE" ]] || { echo "missing store $STORE" >&2; exit 1; }
[[ -f "${EMBED}.json" && -f "${EMBED}.safetensors" ]] || {
    echo "missing embedding cache for $EMBED" >&2
    exit 1
}
[[ -f "$POOL_MANIFEST" ]] || { echo "missing $POOL_MANIFEST" >&2; exit 1; }
[[ ! -e "$TRAIN_OUT" ]] || { echo "output already exists: $TRAIN_OUT" >&2; exit 1; }

if [[ -z "${FORGE_DIR:-}" ]]; then
    export FORGE_DIR="$ROOT/../forge"
fi
[[ -d "$FORGE_DIR" ]] || { echo "missing Forge checkout $FORGE_DIR" >&2; exit 1; }

echo "[pipeline] training"
"$PYTHON" -m anvil.training.train \
    --store "$STORE" \
    --embed "$EMBED" \
    --pool-manifest "$POOL_MANIFEST" \
    --out "$TRAIN_OUT" \
    --batch "$TRAIN_BATCH" \
    --lr 3e-4 \
    --warmup 500 \
    --steps "$TRAIN_STEPS" \
    --pass-weight 0.1 \
    --workers "$TRAIN_WORKERS" \
    --eval-every 1000 \
    --eval-batches 20 \
    --final-eval-batches 100 \
    --seed 0

CKPT="$TRAIN_OUT/last.pt"
[[ -f "$CKPT" ]] || { echo "training did not create $CKPT" >&2; exit 1; }

port_open() {
    "$PYTHON" -c 'import socket, sys; s = socket.socket(); s.settimeout(.25); s.connect(("127.0.0.1", int(sys.argv[1]))); s.close()' "$PORT" \
        >/dev/null 2>&1
}

port_open && { echo "port $PORT is already in use" >&2; exit 1; }

SERVER_LOG="$TRAIN_OUT/eval-server.log"
echo "[pipeline] starting model server on port $PORT"
"$PYTHON" -u -m anvil.bridge.server \
    --mode model \
    --ckpt "$CKPT" \
    --port "$PORT" \
    --device cuda:0 \
    --pass-delta 0.0 \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -INT "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

ready=0
for ((i = 0; i < 120; i++)); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        tail -80 "$SERVER_LOG" >&2 || true
        echo "model server exited during startup" >&2
        exit 1
    fi
    if port_open; then
        ready=1
        break
    fi
    sleep 1
done
((ready == 1)) || { echo "server did not open port; see $SERVER_LOG" >&2; exit 1; }

EVAL_PREFIX="mono-green-auto-$(date +%Y%m%d-%H%M%S)"
EVAL_RUNS=()
shopt -s nullglob

run_eval() {
    local seat="$1"
    local purpose="${EVAL_PREFIX}-s${seat}"
    echo "[pipeline] evaluation seat $seat"
    "$PYTHON" -m anvil.bridge.harness launch \
        --decks "$DECK" "$DECK" \
        --format "$FORMAT" \
        --games "$EVAL_GAMES" \
        --workers "$EVAL_WORKERS" \
        --chunk 10 \
        --calibrated \
        --bridge "grpc:localhost:$PORT" \
        --bridge-seats "$seat" \
        --obs \
        --census \
        --pool-version "$POOL_VERSION" \
        --purpose "$purpose" \
        --seed-base "$SEED_BASE"

    local matches=("$ROOT/data/runs/${purpose}-"*)
    ((${#matches[@]} == 1)) || {
        echo "expected one run directory for $purpose" >&2
        exit 1
    }
    EVAL_RUNS+=("${matches[0]}")
}

run_eval 0
run_eval 1

REPORT="$TRAIN_OUT/arms-report.json"
"$PYTHON" scripts/arms_report.py \
    --arm "bc=${EVAL_RUNS[0]},${EVAL_RUNS[1]}" \
    --out "$REPORT"

echo "[pipeline] checkpoint: $CKPT"
echo "[pipeline] report: $REPORT"
echo "[pipeline] server log: $SERVER_LOG"
