#!/usr/bin/env bash
#SBATCH --job-name=anvil-rl-sweep
#SBATCH --account=p31830
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --array=0-12%4
#SBATCH --output=anvil-rl-sweep-%A_%a.out
#SBATCH --error=anvil-rl-sweep-%A_%a.err

# Run one RL throughput configuration per Slurm array task. The bottleneck
# submitter overrides --cpus-per-task and --array on the sbatch command line,
# because Slurm cannot vary either value between elements of one array.

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
export RL_SERVER_STATS_EVERY="${SWEEP_SERVER_STATS_EVERY:-15}"

expected_cpus=$((2 * WORKERS + 2))
allocated_cpus="${SLURM_CPUS_PER_TASK:-0}"
if [[ "$allocated_cpus" =~ ^[0-9]+$ ]] && (( allocated_cpus < expected_cpus )); then
    echo "allocation has $allocated_cpus CPUs but $CONFIG_ID requires at least $expected_cpus" >&2
    exit 1
fi

cd "$ROOT"
DIAG_ROOT="$ROOT/data/training/sweeps/$SWEEP_PREFIX/$CONFIG_ID"
mkdir -p "$DIAG_ROOT"
printf '%s\n' "$CONFIG_LINE" > "$DIAG_ROOT/config.tsv"
printf '%s\n' \
    "config_id=$CONFIG_ID" \
    "stage=$STAGE" \
    "workers=$WORKERS" \
    "expected_cpus=$expected_cpus" \
    "allocated_cpus=$allocated_cpus" \
    "slurm_job_id=${SLURM_JOB_ID:-}" \
    "slurm_array_job_id=${SLURM_ARRAY_JOB_ID:-}" \
    "slurm_array_task_id=${SLURM_ARRAY_TASK_ID:-}" \
    "server_stats_every=$RL_SERVER_STATS_EVERY" \
    > "$DIAG_ROOT/resource.txt"
printf '%s\n' \
    "[sweep] task=$TASK_ID stage=$STAGE config=$CONFIG_ID" \
    "[sweep] output=$RL_NAME" \
    "[sweep] workers=$WORKERS chunk=$CHUNK initial-delay-ms=$INITIAL_DELAY replacement-delay-ms=$REPLACEMENT_DELAY" \
    "[sweep] learner-workers=$LEARNER_WORKERS seg=$SEG replay=$REPLAY servers=$SERVERS max-batch=$MAX_BATCH batch-window-ms=$BATCH_WINDOW server-stats-every=$RL_SERVER_STATS_EVERY" \
    "[sweep] iterations=$ITERATIONS games=$GAMES arms-every=$ARMS_EVERY"

# Capture the allocation before and after the run. The capture is best effort:
# missing diagnostic utilities must not turn a valid RL result into a failure.
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
CAPTURE_SCRIPT="$ROOT/scripts/quest/capture_quest_hardware.py"
"$PYTHON" "$CAPTURE_SCRIPT" --phase start --out "$DIAG_ROOT/hardware-start.json" || true

GPU_MONITOR_PID=""
VMSTAT_PID=""
GPU_MONITOR_ID="${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-}}}"
GPU_MONITOR_ID="${GPU_MONITOR_ID%%,*}"

stop_monitoring() {
    if [[ -n "$GPU_MONITOR_PID" ]]; then
        kill "$GPU_MONITOR_PID" 2>/dev/null || true
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
        GPU_MONITOR_PID=""
    fi
    if [[ -n "$VMSTAT_PID" ]]; then
        kill "$VMSTAT_PID" 2>/dev/null || true
        wait "$VMSTAT_PID" 2>/dev/null || true
        VMSTAT_PID=""
    fi
}

finish_task() {
    local rc=$?
    local finished_epoch started_epoch elapsed_s
    finished_epoch="$(date +%s)"
    started_epoch="$(awk -F= '$1 == "started_epoch" {print $2}' "$DIAG_ROOT/start.txt" 2>/dev/null || true)"
    if [[ "$started_epoch" =~ ^[0-9]+$ ]]; then
        elapsed_s=$((finished_epoch - started_epoch))
    else
        elapsed_s=""
    fi
    trap - EXIT TERM INT
    stop_monitoring
    "$PYTHON" "$CAPTURE_SCRIPT" --phase end --out "$DIAG_ROOT/hardware-end.json" || true
    printf '%s\n' "$rc" > "$DIAG_ROOT/return-code"
    printf '%s\n' \
        "started_epoch=$started_epoch" \
        "finished_epoch=$finished_epoch" \
        "elapsed_s=$elapsed_s" \
        "return_code=$rc" \
        > "$DIAG_ROOT/timing.txt"
    exit "$rc"
}
trap finish_task EXIT
trap 'exit 143' TERM INT

if command -v nvidia-smi >/dev/null 2>&1; then
    printf '%s\n' \
        'timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,pstate' \
        > "$DIAG_ROOT/gpu-utilization.csv"
    (
        while :; do
            smi_args=(
                --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,pstate
                --format=csv,noheader,nounits
            )
            if [[ -n "$GPU_MONITOR_ID" ]]; then
                smi_args+=(--id="$GPU_MONITOR_ID")
            fi
            nvidia-smi "${smi_args[@]}" \
                >> "$DIAG_ROOT/gpu-utilization.csv" \
                2>> "$DIAG_ROOT/gpu-utilization.err" || true
            sleep 10
        done
    ) &
    GPU_MONITOR_PID=$!
else
    printf '%s\n' 'nvidia-smi not found' > "$DIAG_ROOT/gpu-utilization.err"
fi

if command -v vmstat >/dev/null 2>&1; then
    vmstat 10 > "$DIAG_ROOT/cpu-vmstat.log" 2>&1 &
    VMSTAT_PID=$!
else
    printf '%s\n' 'vmstat not found' > "$DIAG_ROOT/cpu-vmstat.log"
fi

printf '%s\n' "started_epoch=$(date +%s)" > "$DIAG_ROOT/start.txt"

# The restart wrapper performs the environment, checkpoint, schedule, deck,
# and GPU checks, then executes self-play. Tee preserves the normal Slurm log
# stream while retaining a per-configuration copy for the summarizer.
set +e
bash scripts/quest/restart_constructed_four_rl_aggressive.sbatch \
    > >(tee "$DIAG_ROOT/run.out") \
    2> >(tee "$DIAG_ROOT/run.err" >&2)
run_rc=$?
set -e
exit "$run_rc"
