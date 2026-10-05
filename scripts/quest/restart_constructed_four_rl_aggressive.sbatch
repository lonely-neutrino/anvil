#!/usr/bin/env bash
#SBATCH --job-name=anvil-4deck-rl-aggr
#SBATCH --account=p31830
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=47:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=anvil-4deck-rl-aggressive-%j.out
#SBATCH --error=anvil-4deck-rl-aggressive-%j.err

# Start a fresh RL loop from an already-trained four-deck BC checkpoint.
# This intentionally skips heuristic generation, ingest, validation, and BC
# training.  The default paths and seeds match constructed-four-rl-current;
# only the throughput settings are made more aggressive:
#
#   rollout workers: 12 -> 24
#   learner workers:   0 -> 12
#   learner segment:  64 -> 256
#   worker launch ramp: 2s -> 4s
#
# Override any setting with an environment variable before `sbatch`.  The
# default RL_NAME is new, so this cannot overwrite constructed-four-rl-current.

set -euo pipefail

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
if [[ -n "${QUEST_GIT_MODULE:-}" ]]; then
    module load "$QUEST_GIT_MODULE"
fi

ROOT_INPUT="${ANVIL_ROOT:-${SLURM_SUBMIT_DIR:?set ANVIL_ROOT or submit from the Anvil checkout}}"
ROOT="$(cd -- "$ROOT_INPUT" && pwd)"
FORGE_DIR="${FORGE_DIR:-$ROOT/../forge}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
DECK_DIR="${DECK_DIR:-$HOME/.forge/decks/constructed}"

# These are the paths used by the current Quest/local-style RL run.  They can
# be overridden if the current BC checkpoint or pair schedules live elsewhere.
BC_CKPT="${BC_CKPT:-$ROOT/data/training/constructed-four-bc-current/last.pt}"
RL_PAIRS="${RL_PAIRS:-$ROOT/data/pool/custom/constructed-four-rl-pairs-20480-current.txt}"
ARMS_PAIRS="${ARMS_PAIRS:-$ROOT/data/pool/custom/constructed-four-arms-pairs-800.txt}"

# A new output name is required for a fresh restart.  selfplay resumes an
# existing name from its loop_state, so fail rather than accidentally mixing
# this experiment with an earlier run.
RL_NAME="${RL_NAME:-constructed-four-rl-aggressive-w24-lw12-s256-ramp4}"
RL_OUT="$ROOT/data/training/$RL_NAME"

# Match constructed-four-rl-current unless noted in the header above.
RL_ITERATIONS="${RL_ITERATIONS:-20}"
RL_GAMES="${RL_GAMES:-2048}"
RL_GAMES_PER_PAIR="${RL_GAMES_PER_PAIR:-2}"
RL_WORKERS="${RL_WORKERS:-24}"
RL_CHUNK="${RL_CHUNK:-30}"
RL_SERVERS="${RL_SERVERS:-2}"
RL_MAX_BATCH="${RL_MAX_BATCH:-16}"
RL_BATCH_WINDOW_MS="${RL_BATCH_WINDOW_MS:-12}"
RL_LAUNCH_DELAY_MS="${RL_LAUNCH_DELAY_MS:-4000}"
RL_SEED_BASE="${RL_SEED_BASE:-20261002}"
RL_LEARNER_WORKERS="${RL_LEARNER_WORKERS:-12}"
RL_SEG="${RL_SEG:-256}"
RL_HEUR_FRAC="${RL_HEUR_FRAC:-1.0}"
RL_REPLAY="${RL_REPLAY:-4}"
RL_FRESH_WEIGHT="${RL_FRESH_WEIGHT:-1.0}"
RL_REPLAY_WEIGHT="${RL_REPLAY_WEIGHT:-0.33}"
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
RL_ARMS_EVERY="${RL_ARMS_EVERY:-5}"
RL_ARMS_GAMES="${RL_ARMS_GAMES:-4000}"
RL_ARMS_SEED_BASE="${RL_ARMS_SEED_BASE:-20260710}"

# Use a job-specific pair of local server ports.  The formula is the same
# family used by constructed_four.sbatch, but this script only needs one RL
# fleet, so it uses the first RL port in that range.
if [[ -z "${RL_PORT:-}" ]]; then
    port_base=$((50000 + (${SLURM_JOB_ID:-1} % 1000) * 5))
    RL_PORT="$((port_base + 2))"
fi

cd "$ROOT"
export ANVIL_ROOT="$ROOT"
export FORGE_DIR
export PYTHONUNBUFFERED=1

[[ -x "$PYTHON" ]] || {
    echo "missing Python environment: $PYTHON" >&2
    exit 1
}
[[ -d "$FORGE_DIR" ]] || {
    echo "missing Forge checkout: $FORGE_DIR" >&2
    exit 1
}
[[ -f "$BC_CKPT" ]] || {
    echo "missing BC checkpoint: $BC_CKPT" >&2
    exit 1
}
[[ -f "$RL_PAIRS" ]] || {
    echo "missing RL pair schedule: $RL_PAIRS" >&2
    exit 1
}
[[ ! -e "$RL_OUT" ]] || {
    echo "RL output already exists: $RL_OUT" >&2
    echo "Choose a new RL_NAME, or use the normal resume workflow deliberately." >&2
    exit 1
}

for deck in monoBlueTempo.dck monoGreenStompy.dck monoRedAggro.dck monoWhiteWeenie.dck; do
    [[ -f "$DECK_DIR/$deck" ]] || {
        echo "missing constructed deck: $DECK_DIR/$deck" >&2
        exit 1
    }
done

if (( RL_ITERATIONS <= 0 || RL_GAMES <= 0 || RL_GAMES_PER_PAIR <= 0 )); then
    echo "RL_ITERATIONS, RL_GAMES, and RL_GAMES_PER_PAIR must be positive" >&2
    exit 1
fi
required_total_games=$((RL_ITERATIONS * RL_GAMES))
required_pairs=$(((required_total_games + RL_GAMES_PER_PAIR - 1) / RL_GAMES_PER_PAIR))
available_pairs="$(wc -l < "$RL_PAIRS")"
(( available_pairs >= required_pairs )) || {
    echo "RL pair schedule has $available_pairs lines but needs at least $required_pairs" >&2
    exit 1
}

if (( RL_ARMS_EVERY < 0 )); then
    echo "RL_ARMS_EVERY must be nonnegative" >&2
    exit 1
fi
if (( RL_ARMS_EVERY > 0 )); then
    [[ -f "$ARMS_PAIRS" ]] || {
        echo "missing arms pair schedule: $ARMS_PAIRS" >&2
        exit 1
    }
    # arms-games uses the harness's five-games-per-pair default.  Keep this
    # check explicit because a missing/truncated arms schedule otherwise
    # appears late in the first arms checkpoint.
    if (( RL_ARMS_GAMES <= 0 )); then
        echo "RL_ARMS_GAMES must be positive when arms are enabled" >&2
        exit 1
    fi
    required_arms_pairs=$(((RL_ARMS_GAMES + 5 - 1) / 5))
    available_arms_pairs="$(wc -l < "$ARMS_PAIRS")"
    (( available_arms_pairs >= required_arms_pairs )) || {
        echo "arms pair schedule has $available_arms_pairs lines but needs at least $required_arms_pairs" >&2
        exit 1
    }
fi

"$PYTHON" scripts/quest/check_environment.py \
    --forge-dir "$FORGE_DIR" \
    --require-jar \
    --require-gpu

echo "[quest] starting fresh RL run: $RL_NAME"
echo "[quest] BC checkpoint: $BC_CKPT"
echo "[quest] RL schedule: $RL_PAIRS"
echo "[quest] rollout workers=$RL_WORKERS learner-workers=$RL_LEARNER_WORKERS rl-seg=$RL_SEG launch-delay-ms=$RL_LAUNCH_DELAY_MS"
echo "[quest] games/iteration=$RL_GAMES iterations=$RL_ITERATIONS servers=$RL_SERVERS port=$RL_PORT"

RL_ARGS=(
    --name "$RL_NAME"
    --ckpt "$BC_CKPT"
    --pairs-file "$RL_PAIRS"
    --format Constructed
    --pool-version constructed-four-v1
    --iterations "$RL_ITERATIONS"
    --games "$RL_GAMES"
    --games-per-pair "$RL_GAMES_PER_PAIR"
    --workers "$RL_WORKERS"
    --chunk "$RL_CHUNK"
    --port "$RL_PORT"
    --servers "$RL_SERVERS"
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
    --rl-seg "$RL_SEG"
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
    --no-inhibit
)

exec "$PYTHON" -m anvil.training.selfplay "${RL_ARGS[@]}"
