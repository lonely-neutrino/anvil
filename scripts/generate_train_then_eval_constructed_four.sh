#!/usr/bin/env bash
# Generate a four-deck constructed heuristic corpus, ingest and validate it,
# train the behavior-cloning model, and run mirrored model-vs-heuristic
# evaluation arms.
#
# The harness's built-in --pool mode reads the standard data/pool manifests.
# This experiment instead creates a deterministic, balanced pair schedule over
# the four Forge user decks in ~/.forge/decks/constructed and passes it with
# --pairs-file. Each complete schedule block contains every ordered matchup,
# including self-matches, exactly once.
#
# Use the existing virtualenv directly. This checkout has a ROCm-specific
# PyTorch installation, so do not change these invocations to plain uv run.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---------- experiment configuration ----------

PYTHON="$ROOT/.venv/bin/python"
FORMAT="Constructed"
DECK_DIR="${DECK_DIR:-${FORGE_USER_DIR:-$HOME/.forge}/decks/constructed}"
DECKS=(
    monoBlueTempo.dck
    monoGreenStompy.dck
    monoRedAggro.dck
    monoWhiteWeenie.dck
)
DECK_COUNT="${#DECKS[@]}"
MATCHUP_COUNT=$((DECK_COUNT * DECK_COUNT))

POOL_VERSION="${POOL_VERSION:-constructed-four-v1}"
POOL_MANIFEST="${POOL_MANIFEST:-data/pool/custom/${POOL_VERSION}.json}"
EMBED_MODEL="${EMBED_MODEL:-bge-m3}"
EMBED="${EMBED:-data/embeddings/${POOL_VERSION}-${EMBED_MODEL}}"
EMBED_BATCH="${EMBED_BATCH:-32}"
REBUILD_POOL="${REBUILD_POOL:-0}"

# These generation values can be overridden without editing the script.
GEN_GAMES="${GEN_GAMES:-20000}"
GEN_GAMES_PER_PAIR="${GEN_GAMES_PER_PAIR:-10}"
GEN_WORKERS="${GEN_WORKERS:-8}"
GEN_CHUNK="${GEN_CHUNK:-10}"
GEN_BRIDGE="${GEN_BRIDGE:-local-random}"
GEN_TAGS="${GEN_TAGS:-none}"
GEN_SEED_BASE="${GEN_SEED_BASE:-$(date +%Y%m%d)}"

RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
GEN_PURPOSE="${GEN_PURPOSE:-constructed-four-heur-${GEN_GAMES}-${RUN_STAMP}}"
TRAIN_OUT="${TRAIN_OUT:-data/training/constructed-four-auto-${RUN_STAMP}}"
EVAL_PREFIX="${EVAL_PREFIX:-constructed-four-auto-${RUN_STAMP}}"
# 4000 games / 16 ordered matchups = 250 games per matchup. Five games per
# scheduled pair makes that division exact.
EVAL_GAMES="${EVAL_GAMES:-4000}"
EVAL_GAMES_PER_PAIR="${EVAL_GAMES_PER_PAIR:-5}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
EVAL_SEED_BASE="${EVAL_SEED_BASE:-$GEN_SEED_BASE}"
PORT="${PORT:-50070}"

shopt -s nullglob

[[ -x "$PYTHON" ]] || { echo "missing $PYTHON" >&2; exit 1; }

if [[ -z "${FORGE_DIR:-}" ]]; then
    export FORGE_DIR="$ROOT/../forge"
fi
[[ -d "$FORGE_DIR" ]] || { echo "missing Forge checkout $FORGE_DIR" >&2; exit 1; }

for deck in "${DECKS[@]}"; do
    [[ -f "$DECK_DIR/$deck" ]] || {
        echo "missing constructed deck $DECK_DIR/$deck" >&2
        exit 1
    }
done

[[ ! -e "$TRAIN_OUT" ]] || { echo "output already exists: $TRAIN_OUT" >&2; exit 1; }

# ---------- custom pool and embedding cache ----------

# Derive the manifest from the exact deck files used by the harness. Existing
# manifests are validated and are only replaced with REBUILD_POOL=1.
pool_args=("$POOL_MANIFEST" "$POOL_VERSION")
for deck in "${DECKS[@]}"; do
    pool_args+=("$DECK_DIR/$deck")
done

"$PYTHON" - "${pool_args[@]}" "$REBUILD_POOL" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
pool_version = sys.argv[2]
deck_paths = [Path(p) for p in sys.argv[3:-1]]
rebuild = sys.argv[-1] == "1"

expected_pool = set()
for path in deck_paths:
    in_main = False
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line == "[Main]":
            in_main = True
            continue
        if line.startswith("["):
            in_main = False
        if in_main and "|" in line:
            # Forge .dck entries are written as `COUNT Card|SET|COLLECTOR`.
            # The embedding pool needs the card name without COUNT.
            entry = line.split("|", 1)[0].strip()
            count, separator, name = entry.partition(" ")
            if not separator or not count.isdigit():
                name = entry
            name = name.strip()
            if name:
                expected_pool.add(name)

expected_decks = [path.name for path in deck_paths]
if manifest_path.exists() and not rebuild:
    manifest = json.loads(manifest_path.read_text())
    actual_pool = set(manifest.get("pool", {}))
    actual_decks = [
        d.get("file") if isinstance(d, dict) else d
        for d in manifest.get("decks", [])
    ]
    if manifest.get("pool_version") != pool_version:
        raise SystemExit(
            f"{manifest_path} has pool_version {manifest.get('pool_version')!r}, "
            f"expected {pool_version!r}; set REBUILD_POOL=1 or choose another POOL_VERSION"
        )
    if actual_pool != expected_pool or actual_decks != expected_decks:
        raise SystemExit(
            f"{manifest_path} does not match the four current decklists; "
            "set REBUILD_POOL=1 to rebuild it"
        )
    print(f"[pipeline] using existing pool manifest {manifest_path} ({len(actual_pool)} cards)")
else:
    manifest = {
        "format": "constructed-custom",
        "pool_version": pool_version,
        "decks": [{"file": name} for name in expected_decks],
        "pool": {name: {} for name in sorted(expected_pool)},
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[pipeline] wrote pool manifest {manifest_path} ({len(expected_pool)} cards)")
PY

if [[ ! -f "${EMBED}.json" || ! -f "${EMBED}.safetensors" ]]; then
    echo "[pipeline] generating $EMBED_MODEL embeddings for $POOL_MANIFEST"
    "$PYTHON" -m anvil.encoder embed \
        --model "$EMBED_MODEL" \
        --manifest "$POOL_MANIFEST" \
        --batch "$EMBED_BATCH"
else
    echo "[pipeline] using existing embedding cache $EMBED"
fi

[[ -f "${EMBED}.json" && -f "${EMBED}.safetensors" ]] || {
    echo "embedding generation did not create both files for $EMBED" >&2
    exit 1
}

# ---------- helpers and run guards ----------

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
if [[ -n "${existing_runs[0]:-}" ]]; then
    echo "generation purpose already has run directories: $GEN_PURPOSE" >&2
    exit 1
fi

require_balanced_total() {
    local label="$1" total="$2" games_per_pair="$3"
    local block_size=$((MATCHUP_COUNT * games_per_pair))
    if (( total <= 0 || games_per_pair <= 0 || total % block_size != 0 )); then
        echo "$label must be divisible by $block_size " \
            "($MATCHUP_COUNT ordered matchups x $games_per_pair games per scheduled pair); " \
            "got total=$total games_per_pair=$games_per_pair" >&2
        exit 1
    fi
}

make_pairs() {
    local out="$1" n_pairs="$2" seed="$3"
    if (( n_pairs % MATCHUP_COUNT != 0 )); then
        echo "scheduled pair count $n_pairs is not divisible by $MATCHUP_COUNT" >&2
        exit 1
    fi
    "$PYTHON" scripts/make_constructed_four_pairs.py \
        --out "$out" \
        --pairs "$n_pairs" \
        --seed "$seed" \
        --decks "${DECKS[@]}"
}

validate_run_matchups() {
    local run_dir="$1" expected_games="$2" label="$3"
    "$PYTHON" - "$run_dir/games.jsonl" "$expected_games" "$label" "${DECKS[@]}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

games_path = Path(sys.argv[1])
expected_games = int(sys.argv[2])
label = sys.argv[3]
decks = sys.argv[4:]
expected_pairs = [(a, b) for a in decks for b in decks]
counts = Counter()

with games_path.open() as stream:
    for line in stream:
        if line.strip():
            record = json.loads(line)
            counts[tuple(record["decks"])] += 1

missing = [pair for pair in expected_pairs if pair not in counts]
unexpected = sorted(set(counts) - set(expected_pairs))
values = [counts[pair] for pair in expected_pairs]
if (
    sum(values) != expected_games
    or missing
    or unexpected
    or len(set(values)) != 1
):
    raise SystemExit(
        f"{label} matchup counts are not balanced: total={sum(values)} "
        f"expected={expected_games}, counts={dict(counts)}, "
        f"missing={missing}, unexpected={unexpected}"
    )

print(
    f"[pipeline] {label}: {expected_games} games, "
    f"{len(expected_pairs)} ordered matchups x {values[0]} games"
)
PY
}

GEN_PAIRS="$(mktemp "${TMPDIR:-/tmp}/constructed-four-gen-pairs.XXXXXX")"
EVAL_PAIRS="$(mktemp "${TMPDIR:-/tmp}/constructed-four-eval-pairs.XXXXXX")"
SERVER_PID=""

cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -INT "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    rm -f -- "$GEN_PAIRS" "$EVAL_PAIRS"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# ---------- generation ----------

require_balanced_total "GEN_GAMES" "$GEN_GAMES" "$GEN_GAMES_PER_PAIR"
GEN_N_PAIRS=$((GEN_GAMES / GEN_GAMES_PER_PAIR))
make_pairs "$GEN_PAIRS" "$GEN_N_PAIRS" "$GEN_SEED_BASE"

echo "[pipeline] generating $GEN_GAMES heuristic games over four constructed decks"
"$PYTHON" -m anvil.bridge.harness launch \
    --pairs-file "$GEN_PAIRS" \
    --games-per-pair "$GEN_GAMES_PER_PAIR" \
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
if [[ -z "${generated_runs[0]:-}" || -n "${generated_runs[1]:-}" ]]; then
    echo "expected one generated run for purpose $GEN_PURPOSE" >&2
    exit 1
fi
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
validate_run_matchups "$GEN_RUN" "$GEN_GAMES" "heuristic generation"

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

require_balanced_total "EVAL_GAMES" "$EVAL_GAMES" "$EVAL_GAMES_PER_PAIR"
EVAL_N_PAIRS=$((EVAL_GAMES / EVAL_GAMES_PER_PAIR))
make_pairs "$EVAL_PAIRS" "$EVAL_N_PAIRS" "$EVAL_SEED_BASE"

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
        --pairs-file "$EVAL_PAIRS" \
        --games-per-pair "$EVAL_GAMES_PER_PAIR" \
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
    if [[ -z "${matches[0]:-}" || -n "${matches[1]:-}" ]]; then
        echo "expected one run directory for $purpose" >&2
        exit 1
    fi
    validate_run_matchups "${matches[0]}" "$EVAL_GAMES" "BC evaluation seat $seat"
    EVAL_RUNS+=( "${matches[0]}" )
}

run_eval 0
run_eval 1

REPORT="$TRAIN_OUT/arms-report.json"
"$PYTHON" scripts/arms_report.py \
    --arm "bc=${EVAL_RUNS[0]},${EVAL_RUNS[1]}" \
    --out "$REPORT"

echo "[pipeline] pool manifest: $POOL_MANIFEST"
echo "[pipeline] embedding cache: $EMBED"
echo "[pipeline] generated run: $GEN_RUN"
echo "[pipeline] trajectory store: $STORE"
echo "[pipeline] checkpoint: $CKPT"
echo "[pipeline] report: $REPORT"
echo "[pipeline] server log: $SERVER_LOG"
