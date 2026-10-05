#!/usr/bin/env bash
# Generate a four-deck constructed heuristic corpus, train or reuse the
# behavior-cloning model, evaluate it, and continue into multi-deck RL.
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

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
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
# Experimental legality mask. Keep off until the flag-off identity and
# flag-on replay-parity smoke gates have passed; set TARGET_MASK=1 to enable
# it consistently for corpus generation, BC evaluation, RL, and RL arms.
TARGET_MASK="${TARGET_MASK:-0}"
TARGET_FORGE_ARGS=()
if [[ "$TARGET_MASK" == "1" ]]; then
    TARGET_FORGE_ARGS=(--forge-args "-targetmask legal-plans")
elif [[ "$TARGET_MASK" != "0" ]]; then
    echo "TARGET_MASK must be 0 or 1; got $TARGET_MASK" >&2
    exit 1
fi

# These generation values can be overridden without editing the script.
GEN_GAMES="${GEN_GAMES:-40000}"
GEN_GAMES_PER_PAIR="${GEN_GAMES_PER_PAIR:-10}"
GEN_WORKERS="${GEN_WORKERS:-8}"
GEN_CHUNK="${GEN_CHUNK:-10}"
GEN_CALIBRATED="${GEN_CALIBRATED:-0}"
GEN_RESUME_RUN="${GEN_RESUME_RUN:-}"
GEN_RESUME_WORKERS="${GEN_RESUME_WORKERS:-$GEN_WORKERS}"
GEN_RESUME_CHUNK="${GEN_RESUME_CHUNK:-$GEN_CHUNK}"
GEN_RESUME_CALIBRATED="${GEN_RESUME_CALIBRATED:-$GEN_CALIBRATED}"
GEN_BRIDGE="${GEN_BRIDGE:-local-random}"
GEN_TAGS="${GEN_TAGS:-none}"
GEN_SEED_BASE="${GEN_SEED_BASE:-$(date +%Y%m%d)}"

RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
GEN_PURPOSE="${GEN_PURPOSE:-constructed-four-heur-${GEN_GAMES}-${RUN_STAMP}}"
TRAIN_OUT="${TRAIN_OUT:-data/training/constructed-four-auto-${RUN_STAMP}}"
EVAL_PREFIX="${EVAL_PREFIX:-constructed-four-auto-${RUN_STAMP}}"
BC_STEPS="${BC_STEPS:-200000}"
BC_CKPT_INPUT="${BC_CKPT:-}"
SKIP_BC_EVAL="${SKIP_BC_EVAL:-0}"
# 4000 games / 16 ordered matchups = 250 games per matchup. Ten games per
# scheduled pair makes that division exact.
EVAL_GAMES="${EVAL_GAMES:-4000}"
EVAL_GAMES_PER_PAIR="${EVAL_GAMES_PER_PAIR:-10}"
EVAL_WORKERS="${EVAL_WORKERS:-12}"
EVAL_MAX_BATCH="${EVAL_MAX_BATCH:-16}"
EVAL_BATCH_WINDOW_MS="${EVAL_BATCH_WINDOW_MS:-12}"
EVAL_LAUNCH_DELAY_MS="${EVAL_LAUNCH_DELAY_MS:-2000}"
EVAL_SEED_BASE="${EVAL_SEED_BASE:-$GEN_SEED_BASE}"
EVAL_PORT="${EVAL_PORT:-${PORT:-50070}}"
EVAL_PORT_2="${EVAL_PORT_2:-50071}"

RL_NAME="${RL_NAME:-constructed-four-rl-${RUN_STAMP}}"
# The historical first RL stage used 20 x 2,000 games, which produces a
# 20,000-line schedule (1,250 lines per ordered matchup).
RL_ITERATIONS="${RL_ITERATIONS:-20}"
RL_GAMES="${RL_GAMES:-2000}"
RL_GAMES_PER_PAIR="${RL_GAMES_PER_PAIR:-2}"
RL_WORKERS="${RL_WORKERS:-12}"
RL_CHUNK="${RL_CHUNK:-30}"
RL_PORT="${RL_PORT:-50077}"
RL_PORT_2="${RL_PORT_2:-50078}"
RL_MAX_BATCH="${RL_MAX_BATCH:-16}"
RL_BATCH_WINDOW_MS="${RL_BATCH_WINDOW_MS:-12}"
RL_LAUNCH_DELAY_MS="${RL_LAUNCH_DELAY_MS:-2000}"
RL_SEED_BASE="${RL_SEED_BASE:-$((GEN_SEED_BASE + 1))}"
RL_HEUR_FRAC="${RL_HEUR_FRAC:-0.5}"
RL_REPLAY="${RL_REPLAY:-4}"
RL_FRESH_WEIGHT="${RL_FRESH_WEIGHT:-1.0}"
RL_REPLAY_WEIGHT="${RL_REPLAY_WEIGHT:-0.33}"
RL_LEARNER_WORKERS="${RL_LEARNER_WORKERS:-0}"
RL_EPOCHS="${RL_EPOCHS:-1}"
RL_LR="${RL_LR:-1e-5}"
RL_ENT_WEIGHT="${RL_ENT_WEIGHT:-0.003}"
RL_ENT_FLOOR="${RL_ENT_FLOOR:-0.08}"
RL_VALUE_WEIGHT="${RL_VALUE_WEIGHT:-0.5}"
RL_TRAJ_PER_STEP="${RL_TRAJ_PER_STEP:-4}"
RL_PENALTY="${RL_PENALTY:-0.01}"
RL_PENALTY_GROUPING="${RL_PENALTY_GROUPING:-first}"
RL_GUARD_KL="${RL_GUARD_KL:-2.0}"
RL_GUARD_ENT_MULT="${RL_GUARD_ENT_MULT:-10.0}"
RL_GUARD_VETO_MULT="${RL_GUARD_VETO_MULT:-50.0}"
RL_GUARD_CASTS_FLOOR="${RL_GUARD_CASTS_FLOOR:-0.2}"
RL_ARMS_EVERY="${RL_ARMS_EVERY:-10}"
RL_ARMS_GAMES="${RL_ARMS_GAMES:-4000}"
RL_ARMS_SEED_BASE="${RL_ARMS_SEED_BASE:-20260710}"
RL_NO_INHIBIT="${RL_NO_INHIBIT:-1}"
RL_RESUME="${RL_RESUME:-0}"
RL_PAIRS="${RL_PAIRS:-data/pool/custom/constructed-four-rl-pairs.txt}"
ARMS_PAIRS="${ARMS_PAIRS:-data/pool/custom/constructed-four-arms-pairs-800.txt}"

shopt -s nullglob

[[ -x "$PYTHON" ]] || { echo "missing $PYTHON" >&2; exit 1; }

if [[ -n "$BC_CKPT_INPUT" ]]; then
    [[ -f "$BC_CKPT_INPUT" ]] || {
        echo "missing reusable BC checkpoint $BC_CKPT_INPUT" >&2
        exit 1
    }
fi

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
if [[ "$RL_RESUME" != "0" && "$RL_RESUME" != "1" ]]; then
    echo "RL_RESUME must be 0 or 1; got $RL_RESUME" >&2
    exit 1
fi
if [[ "$RL_RESUME" == "1" && -z "$BC_CKPT_INPUT" ]]; then
    echo "RL_RESUME=1 requires BC_CKPT=<path to the existing BC checkpoint>" >&2
    exit 1
fi
if [[ "$SKIP_BC_EVAL" != "0" && "$SKIP_BC_EVAL" != "1" ]]; then
    echo "SKIP_BC_EVAL must be 0 or 1; got $SKIP_BC_EVAL" >&2
    exit 1
fi
for setting in GEN_CALIBRATED GEN_RESUME_CALIBRATED; do
    value="${!setting}"
    if [[ "$value" != "0" && "$value" != "1" ]]; then
        echo "$setting must be 0 or 1; got $value" >&2
        exit 1
    fi
done
if [[ "$GEN_RESUME_WORKERS" -le 0 || "$GEN_RESUME_CHUNK" -le 0 ]]; then
    echo "GEN_RESUME_WORKERS and GEN_RESUME_CHUNK must be positive" >&2
    exit 1
fi
GEN_PRIORITY_ARGS=()
if [[ "$GEN_CALIBRATED" == "1" ]]; then
    # In the harness, --calibrated means normal OS priority. It does not
    # change the Forge policy, seed stream, or trajectory flags.
    GEN_PRIORITY_ARGS+=(--calibrated)
fi
if [[ "$RL_RESUME" != "1" && -e "$ROOT/data/training/$RL_NAME" ]]; then
    echo "RL output already exists: $ROOT/data/training/$RL_NAME" >&2
    exit 1
fi

if [[ -n "$BC_CKPT_INPUT" && -n "$GEN_RESUME_RUN" ]]; then
    echo "BC_CKPT and GEN_RESUME_RUN cannot be used together" >&2
    exit 1
fi

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

if [[ "$SKIP_BC_EVAL" != "1" ]]; then
    [[ "$EVAL_PORT" != "$EVAL_PORT_2" ]] || {
        echo "EVAL_PORT and EVAL_PORT_2 must be different" >&2
        exit 1
    }

    port_open "$EVAL_PORT" && {
        echo "port $EVAL_PORT is already in use; choose another EVAL_PORT" >&2
        exit 1
    }
    port_open "$EVAL_PORT_2" && {
        echo "port $EVAL_PORT_2 is already in use; choose another EVAL_PORT_2" >&2
        exit 1
    }
    [[ "$EVAL_PORT" != "$RL_PORT" && "$EVAL_PORT" != "$RL_PORT_2" ]] || {
        echo "evaluation and RL ports must be different" >&2
        exit 1
    }
    [[ "$EVAL_PORT_2" != "$RL_PORT" && "$EVAL_PORT_2" != "$RL_PORT_2" ]] || {
        echo "evaluation and RL ports must be different" >&2
        exit 1
    }
fi
[[ "$RL_PORT" != "$RL_PORT_2" ]] || {
    echo "RL_PORT and RL_PORT_2 must be different" >&2
    exit 1
}
port_open "$RL_PORT" && {
    echo "port $RL_PORT is already in use; choose another RL_PORT" >&2
    exit 1
}
port_open "$RL_PORT_2" && {
    echo "port $RL_PORT_2 is already in use; choose another RL_PORT_2" >&2
    exit 1
}

if [[ -z "$BC_CKPT_INPUT" && -z "$GEN_RESUME_RUN" ]]; then
    # A unique purpose keeps a rerun from silently selecting an older
    # generation directory. If the caller supplied a purpose that already
    # exists, stop and make that choice explicit instead.
    existing_runs=("$ROOT/data/runs/${GEN_PURPOSE}-"*)
    if [[ -n "${existing_runs[0]:-}" ]]; then
        echo "generation purpose already has run directories: $GEN_PURPOSE" >&2
        exit 1
    fi
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

prepare_schedule() {
    local path="$1" required_pairs="$2" seed="$3" label="$4"
    if (( required_pairs <= 0 || required_pairs % MATCHUP_COUNT != 0 )); then
        echo "$label schedule requires a positive pair count divisible by " \
            "$MATCHUP_COUNT; got $required_pairs" >&2
        exit 1
    fi
    if [[ -e "$path" ]]; then
        [[ -f "$path" ]] || {
            echo "$label schedule path is not a regular file: $path" >&2
            exit 1
        }
        local available
        available="$(wc -l < "$path")"
        if (( available < required_pairs )); then
            echo "$label schedule $path has $available lines but needs at least " \
                "$required_pairs; choose another path or replace it deliberately" >&2
            exit 1
        fi
        echo "[pipeline] using existing $label schedule $path ($available lines)"
    else
        echo "[pipeline] creating $label schedule $path ($required_pairs lines)"
        make_pairs "$path" "$required_pairs" "$seed"
    fi

    "$PYTHON" - "$path" "$required_pairs" "$label" "${DECKS[@]}" <<'PY'
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1])
required = int(sys.argv[2])
label = sys.argv[3]
decks = sys.argv[4:]
expected = [(a, b) for a in decks for b in decks]
lines = path.read_text().splitlines()
prefix = lines[:required]
counts = Counter()
bad = []
for number, line in enumerate(prefix, 1):
    fields = line.split("\t")
    if len(fields) != 2:
        bad.append((number, line))
        continue
    counts[tuple(fields)] += 1

missing = [pair for pair in expected if pair not in counts]
unexpected = sorted(set(counts) - set(expected))
values = [counts[pair] for pair in expected]
if (
    len(prefix) != required
    or bad
    or missing
    or unexpected
    or len(set(values)) != 1
):
    raise SystemExit(
        f"{label} schedule prefix is not balanced: required={required}, "
        f"counts={dict(counts)}, missing={missing}, unexpected={unexpected}, "
        f"malformed={bad[:3]}"
    )

print(
    f"[pipeline] {label} schedule: {required} lines, "
    f"{len(expected)} ordered matchups x {values[0]} lines"
)
PY
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

validate_resume_run() {
    local run_dir="$1"
    "$PYTHON" - "$run_dir/run.json" "$GEN_GAMES" "$GEN_GAMES_PER_PAIR" \
        "$FORMAT" "$GEN_SEED_BASE" "$POOL_VERSION" "${DECKS[@]}" <<'PY'
import json
import hashlib
import sys
from collections import Counter
from pathlib import Path

manifest_path = Path(sys.argv[1])
expected_games, expected_gpp = map(int, sys.argv[2:4])
expected_format, expected_seed, expected_pool = sys.argv[4], int(sys.argv[5]), sys.argv[6]
decks = sys.argv[7:]
manifest = json.loads(manifest_path.read_text())
checks = {
    "games": (manifest.get("games"), expected_games),
    "games_per_pair": (manifest.get("games_per_pair"), expected_gpp),
    "format": (manifest.get("format"), expected_format),
    "seed_base": (manifest.get("seed_base"), expected_seed),
    "start_index": (manifest.get("start_index", 0), 0),
}
bad = {
    key: got_expected
    for key, got_expected in checks.items()
    if got_expected[0] != got_expected[1]
}
actual_pool = manifest.get("pool_version")
if actual_pool not in (None, expected_pool):
    bad["pool_version"] = (actual_pool, expected_pool)
for key in ("obs", "census"):
    if not manifest.get(key):
        bad[key] = (manifest.get(key), True)

expected_pairs = expected_games // expected_gpp
pairs_rel = manifest.get("pairs_file")
pairs_path = manifest_path.parent / pairs_rel if pairs_rel else None
if pairs_path is None or not pairs_path.is_file():
    bad["pairs_file"] = (pairs_rel, "a readable pinned pairs file")
else:
    pair_lines = pairs_path.read_text().splitlines()
    if manifest.get("n_pairs") != expected_pairs:
        bad["n_pairs"] = (manifest.get("n_pairs"), expected_pairs)
    if len(pair_lines) != expected_pairs:
        bad["pairs_file_lines"] = (len(pair_lines), expected_pairs)
    actual_sha = hashlib.sha256(pairs_path.read_bytes()).hexdigest()
    if manifest.get("pairs_sha256") != actual_sha:
        bad["pairs_sha256"] = (manifest.get("pairs_sha256"), actual_sha)
    expected_matchups = [(a, b) for a in decks for b in decks]
    counts = Counter()
    malformed = []
    for number, line in enumerate(pair_lines, 1):
        fields = line.split("\t")
        if len(fields) != 2:
            malformed.append((number, line))
        else:
            counts[tuple(fields)] += 1
    values = [counts[pair] for pair in expected_matchups]
    if (
        malformed
        or set(counts) != set(expected_matchups)
        or len(set(values)) != 1
    ):
        bad["pair_schedule"] = {
            "counts": dict(counts),
            "malformed": malformed[:3],
        }
if bad:
    raise SystemExit(f"resume run does not match the requested BC corpus: {bad}")
if actual_pool is None:
    print(
        "[pipeline] resume compatibility: explicit-pairs manifest has no "
        "pool_version; pinned pair schedule verified"
    )
print(f"[pipeline] resume manifest verified: {manifest_path.parent}")
PY
}

GEN_PAIRS="$(mktemp "${TMPDIR:-/tmp}/constructed-four-gen-pairs.XXXXXX")"
EVAL_PAIRS="$(mktemp "${TMPDIR:-/tmp}/constructed-four-eval-pairs.XXXXXX")"
SERVER_PIDS=()

stop_servers() {
    for pid in "${SERVER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -INT "$pid" 2>/dev/null || true
        fi
    done
    for pid in "${SERVER_PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    SERVER_PIDS=()
}

cleanup() {
    stop_servers
    rm -f -- "$GEN_PAIRS" "$EVAL_PAIRS"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

GEN_RUN=""
STORE=""
CKPT=""
BC_REPORT=""

if [[ -n "$BC_CKPT_INPUT" ]]; then
    # A supplied BC checkpoint is assumed to have already been generated,
    # trained, and evaluated by its producing pipeline.
    mkdir -p "$TRAIN_OUT"
    CKPT="$BC_CKPT_INPUT"
    echo "[pipeline] reusing BC checkpoint $CKPT"
    echo "[pipeline] skipping heuristic generation, BC training, and BC evaluation"
else

# ---------- generation ----------

require_balanced_total "GEN_GAMES" "$GEN_GAMES" "$GEN_GAMES_PER_PAIR"
if [[ -n "$GEN_RESUME_RUN" ]]; then
    [[ -d "$GEN_RESUME_RUN" ]] || {
        echo "GEN_RESUME_RUN is not a directory: $GEN_RESUME_RUN" >&2
        exit 1
    }
    GEN_RUN="$(cd "$GEN_RESUME_RUN" && pwd)"
    validate_resume_run "$GEN_RUN"
    resume_args=(
        "$GEN_RUN"
        --workers "$GEN_RESUME_WORKERS"
        --chunk "$GEN_RESUME_CHUNK"
    )
    if [[ "$GEN_RESUME_CALIBRATED" == "1" ]]; then
        resume_args+=(--calibrated)
    fi
    echo "[pipeline] resuming $GEN_RUN with workers=$GEN_RESUME_WORKERS chunk=$GEN_RESUME_CHUNK"
    "$PYTHON" -m anvil.bridge.harness resume "${resume_args[@]}"
else
    GEN_N_PAIRS=$((GEN_GAMES / GEN_GAMES_PER_PAIR))
    make_pairs "$GEN_PAIRS" "$GEN_N_PAIRS" "$GEN_SEED_BASE"

    echo "[pipeline] generating $GEN_GAMES heuristic games over four constructed decks"
    echo "[pipeline] generation workers=$GEN_WORKERS chunk=$GEN_CHUNK calibrated=$GEN_CALIBRATED"
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
        --seed-base "$GEN_SEED_BASE" \
        "${GEN_PRIORITY_ARGS[@]}" \
        "${TARGET_FORGE_ARGS[@]}"

    generated_runs=("$ROOT/data/runs/${GEN_PURPOSE}-"*)
    if [[ -z "${generated_runs[0]:-}" || -n "${generated_runs[1]:-}" ]]; then
        echo "expected one generated run for purpose $GEN_PURPOSE" >&2
        exit 1
    fi
    GEN_RUN="${generated_runs[0]}"
fi

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
    --steps "$BC_STEPS" \
    --pass-weight 0.1 \
    --workers 1 \
    --eval-every 1000 \
    --eval-batches 20 \
    --final-eval-batches 100 \
    --seed 0

CKPT="$TRAIN_OUT/last.pt"
[[ -f "$CKPT" ]] || { echo "training did not create $CKPT" >&2; exit 1; }

# ---------- model-vs-heuristic evaluation ----------

if [[ "$SKIP_BC_EVAL" == "1" ]]; then
    echo "[pipeline] skipping BC-vs-heuristic evaluation (SKIP_BC_EVAL=1)"
else
require_balanced_total "EVAL_GAMES" "$EVAL_GAMES" "$EVAL_GAMES_PER_PAIR"
EVAL_N_PAIRS=$((EVAL_GAMES / EVAL_GAMES_PER_PAIR))
make_pairs "$EVAL_PAIRS" "$EVAL_N_PAIRS" "$EVAL_SEED_BASE"

echo "[pipeline] starting model servers on ports $EVAL_PORT and $EVAL_PORT_2"
SERVER_LOG="$TRAIN_OUT/eval-server.log"
SERVER_LOG_2="$TRAIN_OUT/eval-server-2.log"
EVAL_PORTS=("$EVAL_PORT" "$EVAL_PORT_2")

for i in "${!EVAL_PORTS[@]}"; do
    eval_port="${EVAL_PORTS[$i]}"
    if (( i == 0 )); then
        eval_log="$SERVER_LOG"
    else
        eval_log="$SERVER_LOG_2"
    fi
    "$PYTHON" -u -m anvil.bridge.server \
        --mode model \
        --ckpt "$CKPT" \
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
        eval_log="$SERVER_LOG_2"
    fi
    ready=0
    for ((wait_i = 0; wait_i < 120; wait_i++)); do
        if ! kill -0 "${SERVER_PIDS[$i]}" 2>/dev/null; then
            tail -80 "$eval_log" >&2 || true
            echo "model server exited during startup; see $eval_log" >&2
            exit 1
        fi
        if port_open "$eval_port"; then
            ready=1
            break
        fi
        sleep 1
    done
    (( ready == 1 )) || {
        echo "model server did not open port; see $eval_log" >&2
        exit 1
    }
done

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
        --launch-delay-ms "$EVAL_LAUNCH_DELAY_MS" \
        --calibrated \
        --bridge "grpc:localhost:$EVAL_PORT,grpc:localhost:$EVAL_PORT_2" \
        --bridge-seats "$seat" \
        --obs \
        --census \
        --pool-version "$POOL_VERSION" \
        --purpose "$purpose" \
        --seed-base "$EVAL_SEED_BASE" \
        "${TARGET_FORGE_ARGS[@]}"

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

BC_REPORT="$TRAIN_OUT/arms-report.json"
"$PYTHON" scripts/arms_report.py \
    --arm "bc=${EVAL_RUNS[0]},${EVAL_RUNS[1]}" \
    --out "$BC_REPORT"

cleanup
trap - EXIT INT TERM
fi

fi

# ---------- RL self-play ----------

if (( RL_ITERATIONS <= 0 )); then
    echo "RL_ITERATIONS must be positive; got $RL_ITERATIONS" >&2
    exit 1
fi
# RL_GAMES=2000 and RL_GAMES_PER_PAIR=2 do not divide evenly across 16
# matchups in a single iteration (1,000 schedule lines is odd per matchup).
# Therefore the invariant is checked over the complete RL schedule: for the
# historical 20-iteration default this is 20,000 lines = 1,250 per matchup.
RL_TOTAL_GAMES=$((RL_ITERATIONS * RL_GAMES))
require_balanced_total "RL total games" "$RL_TOTAL_GAMES" "$RL_GAMES_PER_PAIR"
RL_N_PAIRS=$((RL_TOTAL_GAMES / RL_GAMES_PER_PAIR))
prepare_schedule "$RL_PAIRS" "$RL_N_PAIRS" "$RL_SEED_BASE" "RL"

# The harness default is five games per arms schedule line. Keep the arms
# schedule balanced over all 16 ordered matchups and let selfplay reuse it at
# each configured arms checkpoint.
ARMS_GAMES_PER_PAIR=5
if (( RL_ARMS_EVERY > 0 )); then
    require_balanced_total "RL_ARMS_GAMES" "$RL_ARMS_GAMES" "$ARMS_GAMES_PER_PAIR"
    ARMS_N_PAIRS=$((RL_ARMS_GAMES / ARMS_GAMES_PER_PAIR))
    prepare_schedule "$ARMS_PAIRS" "$ARMS_N_PAIRS" "$RL_ARMS_SEED_BASE" "arms"
fi

echo "[pipeline] starting RL loop $RL_NAME"
RL_ARGS=(
    --name "$RL_NAME"
    --ckpt "$CKPT"
    --pairs-file "$RL_PAIRS"
    --format "$FORMAT"
    --pool-version "$POOL_VERSION"
    --iterations "$RL_ITERATIONS"
    --games "$RL_GAMES"
    --games-per-pair "$RL_GAMES_PER_PAIR"
    --workers "$RL_WORKERS"
    --chunk "$RL_CHUNK"
    --port "$RL_PORT"
    --servers 2
    --device cuda:0
    --max-batch "$RL_MAX_BATCH"
    --batch-window-ms "$RL_BATCH_WINDOW_MS"
    --launch-delay-ms "$RL_LAUNCH_DELAY_MS"
    --seed-base "$RL_SEED_BASE"
    --temperature 1.0
    --replay "$RL_REPLAY"
    --fresh-weight "$RL_FRESH_WEIGHT"
    --replay-weight "$RL_REPLAY_WEIGHT"
    --rl-workers "$RL_LEARNER_WORKERS"
    --epochs "$RL_EPOCHS"
    --lr "$RL_LR"
    --ent-weight "$RL_ENT_WEIGHT"
    --ent-floor "$RL_ENT_FLOOR"
    --rl-seg 64
    --guard-kl "$RL_GUARD_KL"
    --guard-ent-mult "$RL_GUARD_ENT_MULT"
    --guard-veto-mult "$RL_GUARD_VETO_MULT"
    --guard-casts-floor "$RL_GUARD_CASTS_FLOOR"
    --penalty "$RL_PENALTY"
    --penalty-grouping "$RL_PENALTY_GROUPING"
    --heur-frac "$RL_HEUR_FRAC"
    --value-weight "$RL_VALUE_WEIGHT"
    --traj-per-step "$RL_TRAJ_PER_STEP"
    --arms-every "$RL_ARMS_EVERY"
    --arms-pairs "$ARMS_PAIRS"
    --arms-games "$RL_ARMS_GAMES"
    --arms-seed-base "$RL_ARMS_SEED_BASE"
    --reask
)

if [[ "$RL_NO_INHIBIT" == "1" ]]; then
    RL_ARGS+=(--no-inhibit)
fi
if [[ "$TARGET_MASK" == "1" ]]; then
    RL_ARGS+=(--target-mask)
fi

"$PYTHON" -m anvil.training.selfplay "${RL_ARGS[@]}"

echo "[pipeline] pool manifest: $POOL_MANIFEST"
echo "[pipeline] embedding cache: $EMBED"
if [[ -n "$GEN_RUN" ]]; then
    echo "[pipeline] generated run: $GEN_RUN"
    echo "[pipeline] trajectory store: $STORE"
else
    echo "[pipeline] generated run: skipped (BC_CKPT supplied)"
    echo "[pipeline] trajectory store: skipped (BC_CKPT supplied)"
fi
echo "[pipeline] checkpoint: $CKPT"
if [[ -n "$BC_REPORT" ]]; then
    echo "[pipeline] BC report: $BC_REPORT"
    echo "[pipeline] server logs: $SERVER_LOG and $SERVER_LOG_2"
elif [[ "$SKIP_BC_EVAL" == "1" ]]; then
    echo "[pipeline] BC report: skipped (SKIP_BC_EVAL=1)"
else
    echo "[pipeline] BC report: skipped (BC_CKPT supplied)"
fi
echo "[pipeline] RL output: $ROOT/data/training/$RL_NAME"
