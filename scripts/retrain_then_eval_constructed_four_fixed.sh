#!/usr/bin/env bash
# Retrain the constructed-four behavior-cloning model from the existing
# 40,000-game heuristic trajectory store using the corrected player-target
# convention, then reevaluate it against the heuristic from both registered
# seats.
#
# This deliberately does NOT regenerate or ingest the heuristic-vs-heuristic
# games. The existing trajectory store is loaded by the current dataset code,
# which applies the corrected self-relative player-target labels at load time.
# The old BC checkpoint is not reused because it was trained with the old
# target convention and has no compatible target-convention metadata.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
FORGE_DIR="${FORGE_DIR:-$ROOT/../forge}"
export FORGE_DIR

# Existing artifacts from constructed-four-auto-20260909-211341.
STORE="${STORE:-data/trajectories/constructed-four-heur-40000-20260909-211341-20260909-211341}"
EMBED="${EMBED:-data/embeddings/constructed-four-v1-bge-m3}"
POOL_MANIFEST="${POOL_MANIFEST:-data/pool/custom/constructed-four-v1.json}"
SOURCE_EVAL_RUN="${SOURCE_EVAL_RUN:-data/runs/constructed-four-auto-20260909-211341-s0-20260910-075517}"

# New outputs. Override these when deliberately repeating this recipe.
RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
TRAIN_OUT="${TRAIN_OUT:-data/training/constructed-four-auto-fixed-${RUN_STAMP}}"
EVAL_PREFIX="${EVAL_PREFIX:-constructed-four-auto-fixed-${RUN_STAMP}}"

# These match the original constructed-four BC evaluation arms.
FORMAT="${FORMAT:-Constructed}"
EVAL_GAMES="${EVAL_GAMES:-4000}"
EVAL_GAMES_PER_PAIR="${EVAL_GAMES_PER_PAIR:-10}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
EVAL_CHUNK="${EVAL_CHUNK:-10}"
EVAL_SEED_BASE="${EVAL_SEED_BASE:-42}"
EVAL_PORT="${EVAL_PORT:-50080}"
PAIRS_FILE="${PAIRS_FILE:-$SOURCE_EVAL_RUN/pairs.txt}"

# These match the original BC training config exactly.
TRAIN_BATCH="${TRAIN_BATCH:-32}"
TRAIN_LR="${TRAIN_LR:-3e-4}"
TRAIN_WARMUP="${TRAIN_WARMUP:-500}"
TRAIN_STEPS="${TRAIN_STEPS:-200000}"
TRAIN_PASS_WEIGHT="${TRAIN_PASS_WEIGHT:-0.1}"
TRAIN_WORKERS="${TRAIN_WORKERS:-1}"
TRAIN_EVAL_EVERY="${TRAIN_EVAL_EVERY:-1000}"
TRAIN_EVAL_BATCHES="${TRAIN_EVAL_BATCHES:-20}"
TRAIN_FINAL_EVAL_BATCHES="${TRAIN_FINAL_EVAL_BATCHES:-100}"
TRAIN_SEED="${TRAIN_SEED:-0}"

shopt -s nullglob

fail() {
    echo "[pipeline] ERROR: $*" >&2
    exit 1
}

absolute_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "$ROOT" "$1" ;;
    esac
}

PYTHON="$(absolute_path "$PYTHON")"
STORE_PATH="$(absolute_path "$STORE")"
EMBED_PATH="$(absolute_path "$EMBED")"
POOL_MANIFEST_PATH="$(absolute_path "$POOL_MANIFEST")"
PAIRS_PATH="$(absolute_path "$PAIRS_FILE")"
TRAIN_OUT_PATH="$(absolute_path "$TRAIN_OUT")"

[[ -x "$PYTHON" ]] || fail "missing executable Python: $PYTHON"
[[ -d "$FORGE_DIR" ]] || fail "missing Forge checkout: $FORGE_DIR"
[[ -d "$STORE_PATH" ]] || fail "missing trajectory store: $STORE_PATH"
[[ -f "$STORE_PATH/manifest.json" ]] || fail "missing store manifest: $STORE_PATH/manifest.json"
[[ -f "$EMBED_PATH.json" && -f "$EMBED_PATH.safetensors" ]] || \
    fail "missing embedding cache: $EMBED_PATH(.json/.safetensors)"
[[ -f "$POOL_MANIFEST_PATH" ]] || fail "missing pool manifest: $POOL_MANIFEST_PATH"
[[ -f "$PAIRS_PATH" ]] || fail "missing evaluation pairs file: $PAIRS_PATH"
[[ ! -e "$TRAIN_OUT_PATH" ]] || fail "training output already exists: $TRAIN_OUT_PATH"

existing_eval_runs=("$ROOT/data/runs/${EVAL_PREFIX}-"*)
(( ${#existing_eval_runs[@]} == 0 )) || \
    fail "evaluation prefix already has run directories: $EVAL_PREFIX"

port_open() {
    "$PYTHON" -c 'import socket, sys; s = socket.socket(); s.settimeout(.25); s.connect(("127.0.0.1", int(sys.argv[1]))); s.close()' "$1" \
        >/dev/null 2>&1
}

port_open "$EVAL_PORT" && fail "evaluation port $EVAL_PORT is already in use"

SERVER_PID=""

stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
}

trap stop_server EXIT
trap 'exit 130' INT TERM

echo "[pipeline] validating existing trajectory store"
"$PYTHON" -m anvil.store validate "$STORE_PATH"

echo "[pipeline] training corrected BC model into $TRAIN_OUT_PATH"
"$PYTHON" -m anvil.training.train \
    --store "$STORE_PATH" \
    --embed "$EMBED_PATH" \
    --pool-manifest "$POOL_MANIFEST_PATH" \
    --out "$TRAIN_OUT_PATH" \
    --batch "$TRAIN_BATCH" \
    --lr "$TRAIN_LR" \
    --warmup "$TRAIN_WARMUP" \
    --steps "$TRAIN_STEPS" \
    --pass-weight "$TRAIN_PASS_WEIGHT" \
    --workers "$TRAIN_WORKERS" \
    --eval-every "$TRAIN_EVAL_EVERY" \
    --eval-batches "$TRAIN_EVAL_BATCHES" \
    --final-eval-batches "$TRAIN_FINAL_EVAL_BATCHES" \
    --seed "$TRAIN_SEED"

CKPT="$TRAIN_OUT_PATH/last.pt"
[[ -f "$CKPT" ]] || fail "training did not create checkpoint: $CKPT"

echo "[pipeline] starting corrected model server on port $EVAL_PORT"
SERVER_LOG="$TRAIN_OUT_PATH/eval-server.log"
"$PYTHON" -u -m anvil.bridge.server \
    --mode model \
    --ckpt "$CKPT" \
    --port "$EVAL_PORT" \
    --device cuda:0 \
    --pass-delta 0.0 \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

ready=0
for ((i = 0; i < 120; i++)); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        tail -80 "$SERVER_LOG" >&2 || true
        fail "model server exited during startup"
    fi
    if port_open "$EVAL_PORT"; then
        ready=1
        break
    fi
    sleep 1
done
(( ready == 1 )) || fail "model server did not open port; see $SERVER_LOG"

EVAL_RUNS=()

run_eval() {
    local seat="$1"
    local purpose="${EVAL_PREFIX}-s${seat}"

    echo "[pipeline] evaluating registered seat $seat"
    "$PYTHON" -m anvil.bridge.harness launch \
        --pairs-file "$PAIRS_PATH" \
        --games-per-pair "$EVAL_GAMES_PER_PAIR" \
        --format "$FORMAT" \
        --games "$EVAL_GAMES" \
        --workers "$EVAL_WORKERS" \
        --chunk "$EVAL_CHUNK" \
        --calibrated \
        --bridge "grpc:localhost:$EVAL_PORT" \
        --bridge-seats "$seat" \
        --obs \
        --census \
        --pool-version constructed-four-v1 \
        --purpose "$purpose" \
        --seed-base "$EVAL_SEED_BASE"

    local matches=("$ROOT/data/runs/${purpose}-"*)
    (( ${#matches[@]} == 1 )) || \
        fail "expected one run directory for $purpose; found ${#matches[@]}"
    EVAL_RUNS+=("${matches[0]}")
}

run_eval 0
run_eval 1

REPORT="$TRAIN_OUT_PATH/arms-report.json"
"$PYTHON" "$ROOT/scripts/arms_report.py" \
    --arm "bc=${EVAL_RUNS[0]},${EVAL_RUNS[1]}" \
    --out "$REPORT"

echo "[pipeline] reused trajectory store: $STORE_PATH"
echo "[pipeline] reused evaluation pairs: $PAIRS_PATH"
echo "[pipeline] checkpoint: $CKPT"
echo "[pipeline] seat-0 evaluation: ${EVAL_RUNS[0]}"
echo "[pipeline] seat-1 evaluation: ${EVAL_RUNS[1]}"
echo "[pipeline] report: $REPORT"
echo "[pipeline] server log: $SERVER_LOG"
