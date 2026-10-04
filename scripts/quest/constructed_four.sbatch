#!/usr/bin/env bash
#SBATCH --job-name=anvil-4deck
#SBATCH --account=REPLACE_WITH_QUEST_ACCOUNT
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=47:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=anvil-4deck-%j.out
#SBATCH --error=anvil-4deck-%j.err

# Run the existing four-deck pipeline inside one Quest GPU allocation. The
# default is a deliberately small end-to-end smoke. Set QUEST_MODE=full for
# the production defaults, or override any pipeline variable in the exported
# environment before calling sbatch.

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

ROOT="${ANVIL_ROOT:-${SLURM_SUBMIT_DIR:?set ANVIL_ROOT or submit from the Anvil checkout}}"
FORGE_DIR="${FORGE_DIR:?set FORGE_DIR to the built Forge checkout}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
DECK_DIR="${DECK_DIR:-$HOME/.forge/decks/constructed}"
QUEST_MODE="${QUEST_MODE:-smoke}"
RUN_TAG="${QUEST_RUN_TAG:-quest-4deck-${SLURM_JOB_ID:-manual}}"

cd "$ROOT"
[[ -x "$PYTHON" ]] || { echo "missing Python environment: $PYTHON" >&2; exit 1; }
[[ -d "$FORGE_DIR" ]] || { echo "missing Forge checkout: $FORGE_DIR" >&2; exit 1; }

for deck in monoBlueTempo.dck monoGreenStompy.dck monoRedAggro.dck monoWhiteWeenie.dck; do
    [[ -f "$DECK_DIR/$deck" ]] || {
        echo "missing constructed deck: $DECK_DIR/$deck" >&2
        exit 1
    }
done

"$PYTHON" scripts/quest/check_environment.py \
    --forge-dir "$FORGE_DIR" \
    --require-jar \
    --require-gpu

# Give every job its own run names and schedule paths. A resumed RL job must
# explicitly re-export the old RL_NAME, RL_PAIRS, and BC_CKPT values.
export ANVIL_ROOT="$ROOT"
export FORGE_DIR
export DECK_DIR
export GEN_PURPOSE="${GEN_PURPOSE:-$RUN_TAG-generation}"
export TRAIN_OUT="${TRAIN_OUT:-data/training/$RUN_TAG-bc}"
export EVAL_PREFIX="${EVAL_PREFIX:-$RUN_TAG-eval}"
export RL_NAME="${RL_NAME:-$RUN_TAG-rl}"
export RL_PAIRS="${RL_PAIRS:-data/pool/custom/$RUN_TAG-rl-pairs.txt}"
export ARMS_PAIRS="${ARMS_PAIRS:-data/pool/custom/$RUN_TAG-arms-pairs.txt}"

case "$QUEST_MODE" in
    smoke)
        export GEN_GAMES="${GEN_GAMES:-256}"
        export GEN_GAMES_PER_PAIR="${GEN_GAMES_PER_PAIR:-2}"
        export GEN_WORKERS="${GEN_WORKERS:-4}"
        export GEN_CHUNK="${GEN_CHUNK:-4}"
        export BC_STEPS="${BC_STEPS:-2000}"
        export EVAL_GAMES="${EVAL_GAMES:-256}"
        export EVAL_GAMES_PER_PAIR="${EVAL_GAMES_PER_PAIR:-2}"
        export EVAL_WORKERS="${EVAL_WORKERS:-4}"
        export RL_ITERATIONS="${RL_ITERATIONS:-1}"
        export RL_GAMES="${RL_GAMES:-256}"
        export RL_GAMES_PER_PAIR="${RL_GAMES_PER_PAIR:-2}"
        export RL_WORKERS="${RL_WORKERS:-4}"
        export RL_CHUNK="${RL_CHUNK:-4}"
        export RL_ARMS_EVERY="${RL_ARMS_EVERY:-0}"
        export RL_NO_INHIBIT="${RL_NO_INHIBIT:-1}"
        ;;
    full)
        # Keep the pipeline's measured defaults unless the caller overrides
        # them. In particular, one job stays on one GPU/node.
        export RL_NO_INHIBIT="${RL_NO_INHIBIT:-1}"
        ;;
    *)
        echo "QUEST_MODE must be smoke or full; got $QUEST_MODE" >&2
        exit 1
        ;;
esac

# Four model-server ports are local to the node. Deriving them from the job
# id avoids collisions when two allocations happen to share a host.
if [[ -z "${EVAL_PORT:-}" ]]; then
    port_base=$((50000 + (${SLURM_JOB_ID:-1} % 1000) * 5))
    export EVAL_PORT="$port_base"
    export EVAL_PORT_2="$((port_base + 1))"
    export RL_PORT="$((port_base + 2))"
    export RL_PORT_2="$((port_base + 3))"
fi

echo "[quest] mode=$QUEST_MODE root=$ROOT forge=$FORGE_DIR"
echo "[quest] rl-name=$RL_NAME"
exec bash scripts/generate_train_then_eval_constructed_four.sh
