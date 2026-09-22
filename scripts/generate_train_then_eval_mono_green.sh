#!/usr/bin/env bash
# Generate a fixed-deck heuristic corpus, ingest and validate it, then train
# the mono-green behavior-cloning model and run mirrored model-vs-heuristic
# evaluation arms.
#
# Use the existing virtualenv directly. This checkout has a ROCm-specific
# PyTorch installation, so do not change these invocations to plain `uv run`.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---------- experiment configuration ----------

PYTHON="$ROOT/.venv/bin/python"
DECK="monoGreenStompy.dck"
FORMAT="Constructed"
POOL_VERSION="mono-green-stompy-v1"
EMBED="data/embeddings/mono-green-stompy-v1-bge-m3"
POOL_MANIFEST="data/pool/custom/mono-green-stompy-v1.json"

# These generation values can be overridden without editing the script, for
# example: GEN_GAMES=2000 GEN_WORKERS=4 ./scripts/...
GEN_GAMES="${GEN_GAMES:-20000}"
GEN_WORKERS="${GEN_WORKERS:-8}"
GEN_CHUNK="${GEN_CHUNK:-10}"
GEN_BRIDGE="${GEN_BRIDGE:-local-random}"
GEN_TAGS="${GEN_TAGS:-none}"
GEN_SEED_BASE="${GEN_SEED_BASE:-$(date +%Y%m%d)}"

RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
GEN_PURPOSE="${GEN_PURPOSE:-mono-green-heur-${GEN_GAMES}-${RUN_STAMP}}"
TRAIN_OUT="${TRAIN_OUT:-data/training/mono-green-auto-${RUN_STAMP}}"
EVAL_PREFIX="${EVAL_PREFIX:-mono-green-auto-${RUN_STAMP}}"
EVAL_GAMES="${EVAL_GAMES:-400}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
EVAL_SEED_BASE="${EVAL_SEED_BASE:-$GEN_SEED_BASE}"
PORT="${PORT:-50070}"

shopt -s nullglob

[[ -x "$PYTHON" ]] || { echo "missing $PYTHON" >&2; exit 1; }
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

port_open() {
    "$PYTHON" -c 'import socket, sys; s = socket.socket(); s.settimeout(.25); s.connect(("127.0.0.1", int(sys.argv[1]))); s.close()' "$1" \
        >/dev/null 2>&1
}

port_open "$PORT" && {
    echo "port $PORT is already in use; choose another PORT" >&2
    exit 1
}

# A unique purpose keeps a rerun from silently selecting an older generation
# directory. If the caller supplied a purpose that already exists, stop and
# make that choice explicit instead.
existing_runs=("$ROOT/data/runs/${GEN_PURPOSE}-"*)
((${#existing_runs[@]} == 0)) || {
    echo "generation purpose already has run directories: $GEN_PURPOSE" >&2
    exit 1
}

# ---------- generation ----------

echo "[pipeline] generating $GEN_GAMES heuristic games"
"$PYTHON" -m anvil.bridge.harness launch \
    --decks "$DECK" "$DECK" \
    --format "$FORMAT" \
    --games "$GEN_GAMES" \
    --workers "$GEN_WORKERS" \
    --chunk "$GEN_CHUNK" \
    --bridge "$GEN_BRIDGE" \
    --tags "$GEN_TAGS" \
    --obs \
    --census \
    --pool-version "$POOL_VERSION" \
    --purpose "$GEN_PURPOSE" \
    --seed-base "$GEN_SEED_BASE"

generated_runs=("$ROOT/data/runs/${GEN_PURPOSE}-"*)
((${#generated_runs[@]} == 1)) || {
    echo "expected one generated run for purpose $GEN_PURPOSE" >&2
    exit 1
}
GEN_RUN="${generated_runs[0]}"

"$PYTHON" - "$GEN_RUN/summary.json" "$GEN_GAMES" <<'PY'
import json
import sys

summary_path, expected_text = sys.argv[1:]
expected = int(expected_text)
summary = json.load(open(summary_path))
actual = int(summary.get("games", 0))
skipped = summary.get("skipped", [])
if actual != expected or skipped:
    raise SystemExit(
        f"generation incomplete: requested {expected}, completed {actual}, "
        f"skipped {len(skipped)}"
    )
print(f"[pipeline] generation complete: {actual} games")
PY

# ---------- ingest and validation ----------

echo "[pipeline] ingesting and verifying $GEN_RUN"
"$PYTHON" -m anvil.store ingest \
    "$GEN_RUN" \
    --pool-version "$POOL_VERSION" \
    --verify

STORE="$ROOT/data/trajectories/$(basename "$GEN_RUN")"
[[ -f "$STORE/manifest.json" ]] || {
    echo "ingest did not create $STORE/manifest.json" >&2
    exit 1
}

echo "[pipeline] validating $STORE"
"$PYTHON" -m anvil.store validate "$STORE"

# ---------- behavior-cloning training ----------

echo "[pipeline] training from $STORE"
"$PYTHON" -m anvil.training.train \
    --store "$STORE" \
    --embed "$EMBED" \
    --pool-manifest "$POOL_MANIFEST" \
    --out "$TRAIN_OUT" \
    --batch 32 \
    --lr 3e-4 \
    --warmup 500 \
    --steps 200000 \
    --pass-weight 0.1 \
    --workers 1 \
    --eval-every 1000 \
    --eval-batches 20 \
    --final-eval-batches 100 \
    --seed 0

CKPT="$TRAIN_OUT/last.pt"
[[ -f "$CKPT" ]] || { echo "training did not create $CKPT" >&2; exit 1; }

# ---------- model-vs-heuristic evaluation ----------

echo "[pipeline] starting model server on port $PORT"
SERVER_LOG="$TRAIN_OUT/eval-server.log"
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
    if port_open "$PORT"; then
        ready=1
        break
    fi
    sleep 1
done
((ready == 1)) || {
    echo "server did not open port; see $SERVER_LOG" >&2
    exit 1
}

EVAL_RUNS=()

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
        --seed-base "$EVAL_SEED_BASE"

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

echo "[pipeline] generated run: $GEN_RUN"
echo "[pipeline] trajectory store: $STORE"
echo "[pipeline] checkpoint: $CKPT"
echo "[pipeline] report: $REPORT"
echo "[pipeline] server log: $SERVER_LOG"
