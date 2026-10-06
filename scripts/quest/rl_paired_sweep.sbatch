#!/usr/bin/env bash
#SBATCH --job-name=anvil-rl-paired
#SBATCH --account=p31830
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --array=0-4%2
#SBATCH --output=anvil-rl-paired-%A_%a.out
#SBATCH --error=anvil-rl-paired-%A_%a.err

# Run two configurations sequentially in one allocation.  A pair therefore
# shares a node, GPU, CPU allocation, filesystem state, and queue placement;
# the paired summary can detect the few things that are still not controlled.

set -euo pipefail

ROOT_INPUT="${ANVIL_ROOT:-${SLURM_SUBMIT_DIR:?set ANVIL_ROOT or submit from the Anvil checkout}}"
ROOT="$(cd -- "$ROOT_INPUT" && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
CONFIG_FILE="${PAIRED_CONFIG_FILE:-$ROOT/scripts/quest/rl_paired_aggressive.tsv}"
PAIRS_FILE="${PAIRED_PAIRS_FILE:-$ROOT/scripts/quest/rl_paired_aggressive_pairs.tsv}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-${PAIRED_TASK_ID:-0}}"

[[ -x "$PYTHON" ]] || {
    echo "missing Python environment: $PYTHON" >&2
    exit 1
}
[[ -f "$CONFIG_FILE" ]] || {
    echo "missing paired configuration: $CONFIG_FILE" >&2
    exit 1
}
[[ -f "$PAIRS_FILE" ]] || {
    echo "missing paired comparison table: $PAIRS_FILE" >&2
    exit 1
}
[[ "$TASK_ID" =~ ^[0-9]+$ ]] || {
    echo "invalid paired task index: $TASK_ID" >&2
    exit 1
}

pair_line="$(awk -F '\t' -v target="$TASK_ID" '
    /^[[:space:]]*#/ || NF == 0 { next }
    { if (n == target) { print; exit } n++ }
' "$PAIRS_FILE")"
[[ -n "$pair_line" ]] || {
    echo "no pair row $TASK_ID in $PAIRS_FILE" >&2
    exit 1
}

IFS=$'\t' read -r PAIR_ID CONFIG_A CONFIG_B ORDER <<< "$pair_line"
for required in PAIR_ID CONFIG_A CONFIG_B ORDER; do
    [[ -n "${!required:-}" ]] || {
        echo "pair row $TASK_ID is missing field $required" >&2
        exit 1
    }
done
case "$ORDER" in
    a-first)
        FIRST_CONFIG="$CONFIG_A"
        SECOND_CONFIG="$CONFIG_B"
        ;;
    b-first)
        FIRST_CONFIG="$CONFIG_B"
        SECOND_CONFIG="$CONFIG_A"
        ;;
    *)
        echo "pair $PAIR_ID has invalid order '$ORDER'" >&2
        exit 1
        ;;
esac

ARRAY_TAG="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
PAIRED_PREFIX="${PAIRED_PREFIX:-constructed-four-rl-paired-${ARRAY_TAG}}"
PAIR_ROOT="$ROOT/data/training/paired-sweeps/$PAIRED_PREFIX/$PAIR_ID"
mkdir -p "$PAIR_ROOT"

lookup_config() {
    local wanted="$1"
    awk -F '\t' -v wanted="$wanted" '
        /^[[:space:]]*#/ || NF == 0 { next }
        $1 == wanted { print; exit }
    ' "$CONFIG_FILE"
}

start_monitoring() {
    local run_root="$1"
    mkdir -p "$run_root"

    "$PYTHON" "$ROOT/scripts/quest/capture_quest_hardware.py" \
        --phase start --out "$run_root/hardware-start.json"

    : > "$run_root/gpu-utilization.csv"
    if command -v nvidia-smi >/dev/null 2>&1; then
        printf '%s\n' \
            'timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,pstate' \
            > "$run_root/gpu-utilization.csv"
        (
            while :; do
                nvidia-smi \
                    --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,pstate \
                    --format=csv,noheader,nounits \
                    >> "$run_root/gpu-utilization.csv" 2>> "$run_root/gpu-utilization.err" || true
                sleep 10
            done
        ) &
        GPU_MONITOR_PID=$!
    else
        printf '%s\n' 'nvidia-smi not found' > "$run_root/gpu-utilization.err"
        GPU_MONITOR_PID=""
    fi

    : > "$run_root/cpu-vmstat.log"
    if command -v vmstat >/dev/null 2>&1; then
        vmstat 10 > "$run_root/cpu-vmstat.log" 2>&1 &
        VMSTAT_PID=$!
    else
        printf '%s\n' 'vmstat not found' > "$run_root/cpu-vmstat.log"
        VMSTAT_PID=""
    fi
}

stop_monitoring() {
    if [[ -n "${GPU_MONITOR_PID:-}" ]]; then
        kill "$GPU_MONITOR_PID" 2>/dev/null || true
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
        GPU_MONITOR_PID=""
    fi
    if [[ -n "${VMSTAT_PID:-}" ]]; then
        kill "$VMSTAT_PID" 2>/dev/null || true
        wait "$VMSTAT_PID" 2>/dev/null || true
        VMSTAT_PID=""
    fi
}

run_config() {
    local slot="$1"
    local config_id="$2"
    local port_slot=0
    if [[ "$slot" == second ]]; then
        port_slot=1
    fi
    local line
    line="$(lookup_config "$config_id")"
    [[ -n "$line" ]] || {
        echo "pair $PAIR_ID references unknown configuration '$config_id'" >&2
        return 2
    }

    local stage initial_delay workers chunk replacement_delay learner_workers seg replay
    local servers max_batch batch_window iterations games arms_every
    IFS=$'\t' read -r \
        _ stage initial_delay workers chunk replacement_delay learner_workers seg replay \
        servers max_batch batch_window iterations games arms_every <<< "$line"
    for required in stage initial_delay workers chunk replacement_delay learner_workers seg replay \
        servers max_batch batch_window iterations games arms_every; do
        [[ -n "${!required:-}" ]] || {
            echo "configuration '$config_id' is missing field $required" >&2
            return 2
        }
    done

    local run_name="${PAIRED_PREFIX}-${PAIR_ID}-${slot}-${config_id}"
    local run_root="$PAIR_ROOT/$slot-$config_id"
    [[ ! -e "$run_root" ]] || {
        echo "paired diagnostics already exist: $run_root" >&2
        return 2
    }
    [[ ! -e "$ROOT/data/training/$run_name" ]] || {
        echo "RL output already exists: $ROOT/data/training/$run_name" >&2
        return 2
    }
    mkdir -p "$run_root"
    printf '%s\n' "$line" > "$run_root/config.tsv"
    {
        printf 'pair_id\tslot\tconfig_id\tstage\trl_name\n'
        printf '%s\t%s\t%s\t%s\t%s\n' "$PAIR_ID" "$slot" "$config_id" "$stage" "$run_name"
    } > "$run_root/identity.tsv"

    export RL_NAME="$run_name"
    export RL_LAUNCH_DELAY_MS="$initial_delay"
    export RL_WORKERS="$workers"
    export RL_CHUNK="$chunk"
    export RL_REPLACEMENT_LAUNCH_DELAY_MS="$replacement_delay"
    export RL_LEARNER_WORKERS="$learner_workers"
    export RL_SEG="$seg"
    export RL_REPLAY="$replay"
    export RL_SERVERS="$servers"
    export RL_MAX_BATCH="$max_batch"
    export RL_BATCH_WINDOW_MS="$batch_window"
    export RL_ITERATIONS="$iterations"
    export RL_GAMES="$games"
    export RL_ARMS_EVERY="$arms_every"

    local job_numeric="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-1}}"
    if [[ "$job_numeric" =~ ^[0-9]+$ ]]; then
        export RL_PORT="$((50000 + (job_numeric % 1000) * 10 + port_slot * 2))"
    else
        export RL_PORT="$((50000 + port_slot * 2))"
    fi
    export ANVIL_ROOT="$ROOT"
    export PYTHONUNBUFFERED=1

    echo "[paired] pair=$PAIR_ID slot=$slot stage=$stage config=$config_id"
    echo "[paired] rl-name=$run_name workers=$workers chunk=$chunk initial-delay-ms=$initial_delay replacement-delay-ms=$replacement_delay"
    echo "[paired] learner-workers=$learner_workers seg=$seg replay=$replay servers=$servers max-batch=$max_batch batch-window-ms=$batch_window"

    start_monitoring "$run_root"
    local started finished rc
    started="$(date +%s)"
    if bash "$ROOT/scripts/quest/restart_constructed_four_rl_aggressive.sbatch" \
        > "$run_root/run.out" 2> "$run_root/run.err"; then
        rc=0
    else
        rc=$?
    fi
    finished="$(date +%s)"
    stop_monitoring
    "$PYTHON" "$ROOT/scripts/quest/capture_quest_hardware.py" \
        --phase end --out "$run_root/hardware-end.json"
    printf '%s\n' "$rc" > "$run_root/return-code"
    printf '%s\n' "start_epoch=$started" "end_epoch=$finished" "elapsed_s=$((finished - started))" \
        > "$run_root/timing.txt"
    echo "[paired] pair=$PAIR_ID slot=$slot config=$config_id return-code=$rc elapsed-s=$((finished - started))"
    return "$rc"
}

status_file="$PAIR_ROOT/status.tsv"
printf 'slot\tconfig_id\trl_name\treturn_code\tstart_epoch\tend_epoch\n' > "$status_file"

first_rc=0
first_started="$(date +%s)"
if run_config first "$FIRST_CONFIG"; then
    first_rc=0
else
    first_rc=$?
fi
first_ended="$(date +%s)"
first_name="${PAIRED_PREFIX}-${PAIR_ID}-first-${FIRST_CONFIG}"
first_root="$PAIR_ROOT/first-$FIRST_CONFIG"
if [[ -f "$first_root/return-code" ]]; then
    first_rc="$(< "$first_root/return-code")"
fi
printf 'first\t%s\t%s\t%s\t%s\t%s\n' "$FIRST_CONFIG" "$first_name" "$first_rc" "$first_started" "$first_ended" >> "$status_file"

if (( first_rc != 0 )); then
    echo "[paired] first configuration failed; not running second configuration" >&2
    exit "$first_rc"
fi

second_rc=0
second_started="$(date +%s)"
if run_config second "$SECOND_CONFIG"; then
    second_rc=0
else
    second_rc=$?
fi
second_ended="$(date +%s)"
second_name="${PAIRED_PREFIX}-${PAIR_ID}-second-${SECOND_CONFIG}"
second_root="$PAIR_ROOT/second-$SECOND_CONFIG"
if [[ -f "$second_root/return-code" ]]; then
    second_rc="$(< "$second_root/return-code")"
fi
printf 'second\t%s\t%s\t%s\t%s\t%s\n' "$SECOND_CONFIG" "$second_name" "$second_rc" "$second_started" "$second_ended" >> "$status_file"

if (( second_rc != 0 )); then
    echo "[paired] second configuration failed" >&2
    exit "$second_rc"
fi
echo "[paired] pair $PAIR_ID completed successfully"
