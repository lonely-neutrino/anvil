#!/usr/bin/env bash
# Generate a mono-red heuristic corpus, ingest and validate it, train and
# evaluate the behavior-cloning checkpoint, then run the mono-green-derived RL
# recipe with monoRedAggro.dck in both seats.
#
# This intentionally uses the repository's existing ROCm-aware virtualenv.
# Override the documented variables below from the environment when doing a
# smoke run or rerunning the recipe with a different output name.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---------- experiment configuration ----------

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
FORGE_DIR="${FORGE_DIR:-$ROOT/../forge}"
export FORGE_DIR

DECK="${DECK:-monoRedAggro.dck}"
DECK_DIR="${DECK_DIR:-/home/lonelyneutrino/.forge/decks/constructed}"
RED_DECK="$DECK_DIR/$DECK"
FORMAT="${FORMAT:-Constructed}"

POOL_VERSION="${POOL_VERSION:-mono-red-aggro-v1}"
POOL_MANIFEST="${POOL_MANIFEST:-data/pool/custom/${POOL_VERSION}.json}"
EMBED_MODEL="${EMBED_MODEL:-bge-m3}"
EMBED="${EMBED:-data/embeddings/${POOL_VERSION}-${EMBED_MODEL}}"
REBUILD_POOL="${REBUILD_POOL:-0}"
EMBED_BATCH="${EMBED_BATCH:-32}"

GEN_GAMES="${GEN_GAMES:-20000}"
GEN_WORKERS="${GEN_WORKERS:-8}"
GEN_CHUNK="${GEN_CHUNK:-10}"
GEN_BRIDGE="${GEN_BRIDGE:-local-random}"
GEN_TAGS="${GEN_TAGS:-none}"
GEN_SEED_BASE="${GEN_SEED_BASE:-$(date +%Y%m%d)}"

RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
GEN_PURPOSE="${GEN_PURPOSE:-mono-red-aggro-heur-${GEN_GAMES}-${RUN_STAMP}}"
TRAIN_OUT="${TRAIN_OUT:-data/training/mono-red-aggro-bc-${RUN_STAMP}}"
BC_STEPS="${BC_STEPS:-200000}"
# Optional compatible BC checkpoint to reuse instead of retraining.  The
# current model loader will graft fresh RL-only heads, including choose_color,
# onto older checkpoints.
BC_CKPT_INPUT="${BC_CKPT:-}"

EVAL_GAMES="${EVAL_GAMES:-400}"
EVAL_WORKERS="${EVAL_WORKERS:-12}"
EVAL_PORT="${EVAL_PORT:-50070}"
EVAL_PORT_2="${EVAL_PORT_2:-50071}"
EVAL_MAX_BATCH="${EVAL_MAX_BATCH:-16}"
EVAL_BATCH_WINDOW_MS="${EVAL_BATCH_WINDOW_MS:-12}"
EVAL_LAUNCH_DELAY_MS="${EVAL_LAUNCH_DELAY_MS:-2000}"
EVAL_SEED_BASE="${EVAL_SEED_BASE:-$GEN_SEED_BASE}"
EVAL_PREFIX="${EVAL_PREFIX:-mono-red-aggro-bc-${RUN_STAMP}}"

RL_NAME="${RL_NAME:-mono-red-rl-4000-${RUN_STAMP}}"
RL_ITERATIONS="${RL_ITERATIONS:-25}"
RL_GAMES="${RL_GAMES:-480}"
RL_GAMES_PER_PAIR="${RL_GAMES_PER_PAIR:-2}"
RL_WORKERS="${RL_WORKERS:-12}"
RL_CHUNK="${RL_CHUNK:-30}"
RL_PORT="${RL_PORT:-50077}"
# Two sampled model servers on the same GPU improved generation throughput in
# the local benchmark. Workers are routed round-robin across these ports.
RL_PORT_2="${RL_PORT_2:-50078}"
RL_MAX_BATCH="${RL_MAX_BATCH:-16}"
RL_BATCH_WINDOW_MS="${RL_BATCH_WINDOW_MS:-12}"
RL_LAUNCH_DELAY_MS="${RL_LAUNCH_DELAY_MS:-2000}"
RL_SEED_BASE="${RL_SEED_BASE:-$((GEN_SEED_BASE + 1))}"
RL_HEUR_FRAC="${RL_HEUR_FRAC:-0.5}"
RL_ARMS_EVERY="${RL_ARMS_EVERY:-5}"
RL_ARMS_GAMES="${RL_ARMS_GAMES:-500}"
RL_ARMS_SEED_BASE="${RL_ARMS_SEED_BASE:-20260710}"
RL_NO_INHIBIT="${RL_NO_INHIBIT:-1}"
ARMS_PAIRS="${ARMS_PAIRS:-data/pool/custom/mono-red-aggro-pairs.txt}"

shopt -s nullglob

fail() {
    echo "[pipeline] ERROR: $*" >&2
    exit 1
}

[[ -x "$PYTHON" ]] || fail "missing executable Python: $PYTHON"
[[ -d "$FORGE_DIR" ]] || fail "missing Forge checkout: $FORGE_DIR"
[[ -f "$RED_DECK" ]] || fail "missing deck: $RED_DECK"
[[ ! -e "$TRAIN_OUT" ]] || fail "BC output already exists: $TRAIN_OUT"
if [[ -n "$BC_CKPT_INPUT" ]]; then
    [[ -f "$BC_CKPT_INPUT" ]] || fail "missing reusable BC checkpoint: $BC_CKPT_INPUT"
fi
[[ ! -e "$ROOT/data/training/$RL_NAME" ]] || fail "RL output already exists: $ROOT/data/training/$RL_NAME"

port_open() {
    "$PYTHON" -c 'import socket, sys; s = socket.socket(); s.settimeout(.25); s.connect(("127.0.0.1", int(sys.argv[1]))); s.close()' "$1" \
        >/dev/null 2>&1
}

[[ "$EVAL_PORT" != "$EVAL_PORT_2" ]] || fail "EVAL_PORT and EVAL_PORT_2 must be different"
[[ "$EVAL_PORT" != "$RL_PORT" && "$EVAL_PORT" != "$RL_PORT_2" ]] || fail "EVAL_PORT and RL ports must be different"
[[ "$EVAL_PORT_2" != "$RL_PORT" && "$EVAL_PORT_2" != "$RL_PORT_2" ]] || fail "EVAL_PORT_2 and RL ports must be different"
[[ "$RL_PORT" != "$RL_PORT_2" ]] || fail "RL_PORT and RL_PORT_2 must be different"
port_open "$EVAL_PORT" && fail "evaluation port $EVAL_PORT is already in use; choose another EVAL_PORT"
port_open "$EVAL_PORT_2" && fail "evaluation port $EVAL_PORT_2 is already in use; choose another EVAL_PORT_2"
port_open "$RL_PORT" && fail "RL port $RL_PORT is already in use; choose another RL_PORT"
port_open "$RL_PORT_2" && fail "RL port $RL_PORT_2 is already in use; choose another RL_PORT_2"

# ---------- red-only pool manifest ----------

echo "[pipeline] preparing $POOL_MANIFEST"
pool_args=("$RED_DECK" "$POOL_MANIFEST" "$POOL_VERSION" "$DECK" "$REBUILD_POOL")
"$PYTHON" - "${pool_args[@]}" <<'PY'
import json
import sys
from pathlib import Path

deck_path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
pool_version = sys.argv[3]
deck_name = sys.argv[4]
rebuild = sys.argv[5] == "1"

names = set()
in_main = False
for raw in deck_path.read_text().splitlines():
    line = raw.strip()
    if line == "[Main]":
        in_main = True
        continue
    if line.startswith("["):
        in_main = False
    if in_main and "|" in line:
        entry = line.split("|", 1)[0].strip()
        parts = entry.split(maxsplit=1)
        name = parts[1].strip() if len(parts) == 2 and parts[0].isdigit() else entry
        if name:
            names.add(name)

expected_pool = set(names)
expected_decks = [deck_name]

if manifest_path.exists() and not rebuild:
    manifest = json.loads(manifest_path.read_text())
    actual_pool = set(manifest.get("pool", {}))
    actual_decks = [
        item.get("file") if isinstance(item, dict) else item
        for item in manifest.get("decks", [])
    ]
    if manifest.get("pool_version") != pool_version:
        raise SystemExit(
            f"{manifest_path} has pool_version {manifest.get('pool_version')!r}; "
            "set REBUILD_POOL=1 or choose another POOL_VERSION"
        )
    if actual_pool != expected_pool or actual_decks != expected_decks:
        raise SystemExit(
            f"{manifest_path} does not match {deck_path}; "
            "set REBUILD_POOL=1 or choose another POOL_VERSION"
        )
    print(f"[pipeline] using existing manifest with {len(actual_pool)} cards")
else:
    manifest = {
        "format": "constructed-custom",
        "pool_version": pool_version,
        "decks": [{"file": deck_name}],
        "pool": {name: {} for name in sorted(expected_pool)},
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[pipeline] wrote manifest with {len(expected_pool)} cards")
PY

# ---------- embedding cache ----------

if [[ ! -f "${EMBED}.json" || ! -f "${EMBED}.safetensors" ]]; then
    echo "[pipeline] generating $EMBED_MODEL embeddings"
    "$PYTHON" -m anvil.encoder embed \
        --model "$EMBED_MODEL" \
        --manifest "$POOL_MANIFEST" \
        --batch "$EMBED_BATCH"
else
    echo "[pipeline] using existing embedding cache $EMBED"
fi

[[ -f "${EMBED}.json" && -f "${EMBED}.safetensors" ]] || \
    fail "embedding generation did not create both files for $EMBED"

# ---------- heuristic generation ----------

GEN_RUN=""
STORE=""

if [[ -z "$BC_CKPT_INPUT" ]]; then
    existing_runs=("$ROOT/data/runs/${GEN_PURPOSE}-"*)
    (( ${#existing_runs[@]} == 0 )) || \
        fail "generation purpose already has run directories: $GEN_PURPOSE"

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
    (( ${#generated_runs[@]} == 1 )) || \
        fail "expected one generated run for $GEN_PURPOSE; found ${#generated_runs[@]}"
    GEN_RUN="${generated_runs[0]}"

    "$PYTHON" - "$GEN_RUN/summary.json" "$GEN_GAMES" <<'PY'
import json
import sys

summary = json.loads(open(sys.argv[1]).read())
expected = int(sys.argv[2])
actual = int(summary.get("games", 0))
skipped = summary.get("skipped", [])
if actual != expected or skipped:
    raise SystemExit(
        f"generation incomplete: expected {expected}, got {actual}, skipped={len(skipped)}"
    )
print(f"[pipeline] generation complete: {actual} games")
PY

    # ---------- ingest and validation ----------

    echo "[pipeline] ingesting $GEN_RUN"
    "$PYTHON" -m anvil.store ingest \
        "$GEN_RUN" \
        --pool-version "$POOL_VERSION" \
        --verify

    STORE="$ROOT/data/trajectories/$(basename "$GEN_RUN")"
    [[ -f "$STORE/manifest.json" ]] || fail "ingest did not create $STORE/manifest.json"

    echo "[pipeline] validating $STORE"
    "$PYTHON" -m anvil.store validate "$STORE"
else
    echo "[pipeline] skipping heuristic generation and ingest (BC_CKPT supplied)"
fi

# ---------- behavior-cloning checkpoint ----------

if [[ -n "$BC_CKPT_INPUT" ]]; then
    BC_CKPT="$BC_CKPT_INPUT"
    mkdir -p "$TRAIN_OUT"
    echo "[pipeline] reusing compatible BC checkpoint $BC_CKPT"
else
    echo "[pipeline] training BC model into $TRAIN_OUT"
    "$PYTHON" -m anvil.training.train \
        --store "$STORE" \
        --embed "$EMBED" \
        --pool-manifest "$POOL_MANIFEST" \
        --out "$TRAIN_OUT" \
        --batch 32 \
        --lr 3e-4 \
        --warmup 500 \
        --steps "$BC_STEPS" \
        --pass-weight 0.1 \
        --workers 1 \
        --eval-every 100000 \
        --eval-batches 20 \
        --final-eval-batches 100 \
        --seed 0

    BC_CKPT="$ROOT/$TRAIN_OUT/last.pt"
fi

[[ -f "$BC_CKPT" ]] || fail "BC training did not create $BC_CKPT"

# ---------- model-vs-heuristic evaluation ----------

existing_eval_runs=("$ROOT/data/runs/${EVAL_PREFIX}-"*)
(( ${#existing_eval_runs[@]} == 0 )) || \
    fail "evaluation prefix already has run directories: $EVAL_PREFIX"

echo "[pipeline] evaluating BC checkpoint against the heuristic"
SERVER_PIDS=()
SERVER_LOG="$TRAIN_OUT/eval-server.log"
EVAL_PORTS=("$EVAL_PORT" "$EVAL_PORT_2")

stop_eval_server() {
    for pid in "${SERVER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    for pid in "${SERVER_PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    SERVER_PIDS=()
}

trap stop_eval_server EXIT
trap 'stop_eval_server; exit 130' INT TERM

for i in "${!EVAL_PORTS[@]}"; do
    eval_port="${EVAL_PORTS[$i]}"
    if (( i == 0 )); then
        eval_log="$SERVER_LOG"
    else
        eval_log="$TRAIN_OUT/eval-server-$((i + 1)).log"
    fi
    "$PYTHON" -u -m anvil.bridge.server \
        --mode model \
        --ckpt "$BC_CKPT" \
        --port "$eval_port" \
        --device cuda:0 \
        --pass-delta 0.0 \
        --max-batch "$EVAL_MAX_BATCH" \
        --batch-window-ms "$EVAL_BATCH_WINDOW_MS" \
        >"$eval_log" 2>&1 &
    SERVER_PIDS+=("$!")
done

for i in "${!EVAL_PORTS[@]}"; do
    eval_port="${EVAL_PORTS[$i]}"
    if (( i == 0 )); then
        eval_log="$SERVER_LOG"
    else
        eval_log="$TRAIN_OUT/eval-server-$((i + 1)).log"
    fi
    ready=0
    for ((wait_i = 0; wait_i < 120; wait_i++)); do
        if ! kill -0 "${SERVER_PIDS[$i]}" 2>/dev/null; then
            tail -80 "$eval_log" >&2 || true
            fail "model server exited during startup; see $eval_log"
        fi
        if port_open "$eval_port"; then
            ready=1
            break
        fi
        sleep 1
    done
    (( ready == 1 )) || fail "model server did not open port; see $eval_log"
done

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
        --launch-delay-ms "$EVAL_LAUNCH_DELAY_MS" \
        --calibrated \
        --bridges "grpc:localhost:$EVAL_PORT" "grpc:localhost:$EVAL_PORT_2" \
        --bridge-seats "$seat" \
        --obs \
        --census \
        --pool-version "$POOL_VERSION" \
        --purpose "$purpose" \
        --seed-base "$EVAL_SEED_BASE"

    local matches=("$ROOT/data/runs/${purpose}-"*)
    (( ${#matches[@]} == 1 )) || \
        fail "expected one evaluation run for $purpose; found ${#matches[@]}"
    EVAL_RUNS+=("${matches[0]}")
}

run_eval 0
run_eval 1

BC_REPORT="$TRAIN_OUT/arms-report.json"
"$PYTHON" scripts/arms_report.py \
    --arm "bc=${EVAL_RUNS[0]},${EVAL_RUNS[1]}" \
    --out "$BC_REPORT"

stop_eval_server
trap - EXIT INT TERM

# ---------- periodic RL arms schedule ----------

echo "[pipeline] preparing RL arms schedule $ARMS_PAIRS"
"$PYTHON" - "$ARMS_PAIRS" "$DECK" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
deck = sys.argv[2]
expected = f"{deck}\t{deck}\n"

if path.exists():
    if path.read_text() != expected:
        raise SystemExit(
            f"{path} exists but is not the expected single-deck schedule; "
            "choose another ARMS_PAIRS or replace it deliberately"
        )
else:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected)
print(f"[pipeline] arms schedule ready: {path}")
PY

# ---------- RL self-play ----------

echo "[pipeline] starting RL loop $RL_NAME"
RL_ARGS=(
    --name "$RL_NAME"
    --ckpt "$BC_CKPT"
    --decks "$DECK" "$DECK"
    --format "$FORMAT"
    --pool-version "$POOL_VERSION"
    --iterations "$RL_ITERATIONS"
    --games "$RL_GAMES"
    --games-per-pair "$RL_GAMES_PER_PAIR"
    --workers "$RL_WORKERS"
    --chunk "$RL_CHUNK"
    --port "$RL_PORT"
    --ports "$RL_PORT" "$RL_PORT_2"
    --max-batch "$RL_MAX_BATCH"
    --batch-window-ms "$RL_BATCH_WINDOW_MS"
    --launch-delay-ms "$RL_LAUNCH_DELAY_MS"
    --seed-base "$RL_SEED_BASE"
    --temperature 1.0
    --replay 4
    --fresh-weight 1.0
    --replay-weight 0.33
    --rl-workers 4
    --epochs 1
    --lr 1e-5
    --ent-weight 0.003
    --ent-floor 0.08
    --rl-seg 64
    --guard-kl 2.0
    --guard-ent-mult 10.0
    --guard-veto-mult 50.0
    --guard-casts-floor 0.2
    --penalty 0.01
    --penalty-grouping first
    --heur-frac "$RL_HEUR_FRAC"
    --value-weight 0.5
    --traj-per-step 4
    --arms-every "$RL_ARMS_EVERY"
    --arms-pairs "$ARMS_PAIRS"
    --arms-games "$RL_ARMS_GAMES"
    --arms-seed-base "$RL_ARMS_SEED_BASE"
    --reask
)

if [[ "$RL_NO_INHIBIT" == "1" ]]; then
    RL_ARGS+=(--no-inhibit)
fi

"$PYTHON" -m anvil.training.selfplay "${RL_ARGS[@]}"

echo "[pipeline] complete"
echo "[pipeline] pool manifest: $ROOT/$POOL_MANIFEST"
echo "[pipeline] embedding cache: $ROOT/$EMBED"
if [[ -n "$GEN_RUN" ]]; then
    echo "[pipeline] generated run: $GEN_RUN"
    echo "[pipeline] trajectory store: $STORE"
else
    echo "[pipeline] generated run: skipped (BC_CKPT supplied)"
    echo "[pipeline] trajectory store: skipped (BC_CKPT supplied)"
fi
echo "[pipeline] BC checkpoint: $BC_CKPT"
echo "[pipeline] BC evaluation report: $ROOT/$BC_REPORT"
echo "[pipeline] BC evaluation server log: $ROOT/$SERVER_LOG"
echo "[pipeline] RL output: $ROOT/data/training/$RL_NAME"
echo "[pipeline] final RL checkpoint: $ROOT/data/training/$RL_NAME/iter-$(printf '%03d' $((RL_ITERATIONS - 1)))/train/last.pt"
