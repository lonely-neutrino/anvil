#!/usr/bin/env bash
#SBATCH --job-name=anvil-rl-sweep
#SBATCH --account=p31830
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --array=0-12%4
#SBATCH --output=anvil-rl-sweep-%A_%a.out
#SBATCH --error=anvil-rl-sweep-%A_%a.err

# Run one short, isolated RL throughput configuration per Slurm array task.
# The configuration files are deliberately data-only: edit or replace one of
# them, then submit the same launcher with a matching --array range.

set -euo pipefail

ROOT_INPUT="${ANVIL_ROOT:-${SLURM_SUBMIT_DIR:?set ANVIL_ROOT or submit from the Anvil checkout}}"
ROOT="$(cd -- "$ROOT_INPUT" && pwd)"
CONFIG_FILE="${SWEEP_CONFIG_FILE:-$ROOT/scripts/quest/rl_sweep_generation.tsv}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-${SWEEP_TASK_ID:-0}}"

[[ -f "$CONFIG_FILE" ]] || {
    echo "missing sweep configuration: $CONFIG_FILE" >&2
    exit 1
}
[[ "$TASK_ID" =~ ^[0-9]+$ ]] || {
    echo "invalid sweep task index: $TASK_ID" >&2
    exit 1
}

# Skip blank/comment lines and select the zero-based data-row index. The
# header is commented so the same file remains readable to awk and humans.
CONFIG_LINE="$(awk -F '\t' -v target="$TASK_ID" '
    /^[[:space:]]*#/ || NF == 0 { next }
    { if (n == target) { print; exit } n++ }
' "$CONFIG_FILE")"

[[ -n "$CONFIG_LINE" ]] || {
    echo "no configuration row $TASK_ID in $CONFIG_FILE" >&2
    exit 1
}

IFS=$'\t' read -r \
    CONFIG_ID STAGE INITIAL_DELAY WORKERS CHUNK REPLACEMENT_DELAY \
    LEARNER_WORKERS SEG REPLAY SERVERS MAX_BATCH BATCH_WINDOW \
    ITERATIONS GAMES ARMS_EVERY <<< "$CONFIG_LINE"

for required in CONFIG_ID STAGE INITIAL_DELAY WORKERS CHUNK REPLACEMENT_DELAY \
    LEARNER_WORKERS SEG REPLAY SERVERS MAX_BATCH BATCH_WINDOW ITERATIONS GAMES ARMS_EVERY; do
    [[ -n "${!required:-}" ]] || {
        echo "configuration row $TASK_ID is missing field $required" >&2
        exit 1
    }
done

ARRAY_TAG="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
SWEEP_PREFIX="${SWEEP_PREFIX:-constructed-four-rl-sweep-${ARRAY_TAG}}"

# Give every task a unique self-play output directory. Do not honor an
# inherited RL_NAME here: a stale interactive export should never make two
# array tasks collide.
export ANVIL_ROOT="$ROOT"
export RL_NAME="${SWEEP_PREFIX}-${CONFIG_ID}"
export RL_LAUNCH_DELAY_MS="$INITIAL_DELAY"
export RL_WORKERS="$WORKERS"
export RL_CHUNK="$CHUNK"
export RL_REPLACEMENT_LAUNCH_DELAY_MS="$REPLACEMENT_DELAY"
export RL_LEARNER_WORKERS="$LEARNER_WORKERS"
export RL_SEG="$SEG"
export RL_REPLAY="$REPLAY"
export RL_SERVERS="$SERVERS"
export RL_MAX_BATCH="$MAX_BATCH"
export RL_BATCH_WINDOW_MS="$BATCH_WINDOW"
export RL_ITERATIONS="$ITERATIONS"
export RL_GAMES="$GAMES"
export RL_ARMS_EVERY="$ARMS_EVERY"

cd "$ROOT"
mkdir -p "$ROOT/data/training/sweeps/$SWEEP_PREFIX"
printf '%s\n' "$CONFIG_LINE" > "$ROOT/data/training/sweeps/$SWEEP_PREFIX/${CONFIG_ID}.tsv"
printf '%s\n' \
    "[sweep] task=$TASK_ID stage=$STAGE config=$CONFIG_ID" \
    "[sweep] output=$RL_NAME" \
    "[sweep] workers=$WORKERS chunk=$CHUNK initial-delay-ms=$INITIAL_DELAY replacement-delay-ms=$REPLACEMENT_DELAY" \
    "[sweep] learner-workers=$LEARNER_WORKERS seg=$SEG replay=$REPLAY servers=$SERVERS max-batch=$MAX_BATCH batch-window-ms=$BATCH_WINDOW" \
    "[sweep] iterations=$ITERATIONS games=$GAMES arms-every=$ARMS_EVERY"

# The restart wrapper performs the environment, checkpoint, schedule, deck,
# and GPU checks, then executes self-play with these exported values.
exec bash scripts/quest/restart_constructed_four_rl_aggressive.sbatch
