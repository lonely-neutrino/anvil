#!/usr/bin/env bash
# Submit one CPU-sized Slurm array per rollout-worker count.
#
# Usage:
#   scripts/quest/submit_rl_bottleneck_sweep.sh phase1
#   scripts/quest/submit_rl_bottleneck_sweep.sh phase2 --prefix NAME
#   scripts/quest/submit_rl_bottleneck_sweep.sh phase1 --dry-run

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: submit_rl_bottleneck_sweep.sh PHASE [options]

PHASE:
  phase1       submit the five worker-scaling controls
  phase2       submit the 27 max_batch/batch_window configurations

Options:
  --prefix NAME       shared run prefix (required for phase2; generated for phase1)
  --array-limit N     maximum simultaneous tasks per worker group (default: 2)
  --constraint NAME   Slurm node feature (default: quest12)
  --mem VALUE         memory request (default: 128G)
  --time VALUE        time request (default: 03:00:00)
  --stats-every SEC   model-server stats interval (default: 15)
  --dry-run           print sbatch commands without submitting them
  -h, --help          show this help

The phase tables use the normal rl_sweep.sbatch 15-column TSV schema.  The
submitter partitions each table by rollout worker count and overrides
--cpus-per-task with 2 * workers + 2 for each resulting array.
EOF
}

if (( $# == 0 )); then
    usage >&2
    exit 2
fi

PHASE="$1"
shift
case "$PHASE" in
    phase1|phase2) ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "invalid phase '$PHASE' (expected phase1 or phase2)" >&2
        usage >&2
        exit 2
        ;;
esac

ROOT_INPUT="${ANVIL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
ROOT="$(cd -- "$ROOT_INPUT" && pwd)"
CONFIG_FILE="${BOTTLENECK_CONFIG_FILE:-$ROOT/scripts/quest/rl_bottleneck_${PHASE}.tsv}"
ARRAY_LIMIT="${BOTTLENECK_ARRAY_LIMIT:-2}"
CONSTRAINT="${BOTTLENECK_CONSTRAINT:-quest12}"
MEMORY="${BOTTLENECK_MEM:-128G}"
TIME_LIMIT="${BOTTLENECK_TIME:-03:00:00}"
STATS_EVERY="${BOTTLENECK_STATS_EVERY:-15}"
PREFIX="${SWEEP_PREFIX:-}"
DRY_RUN=0

while (( $# > 0 )); do
    case "$1" in
        --prefix)
            [[ $# -ge 2 ]] || { echo "--prefix needs a value" >&2; exit 2; }
            PREFIX="$2"
            shift 2
            ;;
        --array-limit)
            [[ $# -ge 2 ]] || { echo "--array-limit needs a value" >&2; exit 2; }
            ARRAY_LIMIT="$2"
            shift 2
            ;;
        --constraint)
            [[ $# -ge 2 ]] || { echo "--constraint needs a value" >&2; exit 2; }
            CONSTRAINT="$2"
            shift 2
            ;;
        --mem)
            [[ $# -ge 2 ]] || { echo "--mem needs a value" >&2; exit 2; }
            MEMORY="$2"
            shift 2
            ;;
        --time)
            [[ $# -ge 2 ]] || { echo "--time needs a value" >&2; exit 2; }
            TIME_LIMIT="$2"
            shift 2
            ;;
        --stats-every)
            [[ $# -ge 2 ]] || { echo "--stats-every needs a value" >&2; exit 2; }
            STATS_EVERY="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown option '$1'" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[[ -f "$CONFIG_FILE" ]] || {
    echo "missing configuration file: $CONFIG_FILE" >&2
    exit 1
}
[[ "$ARRAY_LIMIT" =~ ^[1-9][0-9]*$ ]] || {
    echo "--array-limit must be a positive integer" >&2
    exit 2
}
[[ "$STATS_EVERY" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
    echo "--stats-every must be a nonnegative number" >&2
    exit 2
}

if [[ -z "$PREFIX" ]]; then
    if [[ "$PHASE" == phase2 ]]; then
        echo "phase2 requires --prefix equal to the phase1 prefix" >&2
        exit 2
    fi
    PREFIX="constructed-four-rl-bottleneck-$(date +%Y%m%d-%H%M%S)"
fi

FORGE_DIR_VALUE="${FORGE_DIR:-$ROOT/../forge}"
DECK_DIR_VALUE="${DECK_DIR:-$HOME/.forge/decks/constructed}"
BC_CKPT_VALUE="${BC_CKPT:-$ROOT/data/training/constructed-four-bc-current/last.pt}"
RL_PAIRS_VALUE="${RL_PAIRS:-$ROOT/data/pool/custom/constructed-four-rl-pairs-20480-current.txt}"
GROUP_ROOT="$ROOT/data/training/sweep-configs/$PREFIX/$PHASE"
SUBMISSION_ROOT="$ROOT/data/training/sweeps/$PREFIX"
[[ ! -e "$GROUP_ROOT" ]] || {
    echo "configuration snapshot already exists: $GROUP_ROOT" >&2
    echo "Choose a new prefix or remove only this unused snapshot deliberately." >&2
    exit 1
}
mkdir -p "$GROUP_ROOT" "$SUBMISSION_ROOT"

# Read and validate the table while making one immutable snapshot per worker
# group. The task script selects rows by zero-based data-row index, so each
# group snapshot can be submitted as its own array with a different CPU size.
declare -A GROUP_FILES=()
declare -A GROUP_COUNTS=()
declare -A SEEN_CONFIGS=()
while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    IFS=$'\t' read -r -a fields <<< "$line"
    if (( ${#fields[@]} != 15 )); then
        echo "configuration row must have 15 tab-separated fields: $line" >&2
        exit 1
    fi
    config_id="${fields[0]}"
    workers="${fields[3]}"
    [[ "$config_id" =~ ^[A-Za-z0-9._-]+$ ]] || {
        echo "invalid configuration id '$config_id'" >&2
        exit 1
    }
    [[ "$workers" =~ ^[1-9][0-9]*$ ]] || {
        echo "invalid worker count '$workers' in '$config_id'" >&2
        exit 1
    }
    [[ -z "${SEEN_CONFIGS[$config_id]+x}" ]] || {
        echo "duplicate configuration id '$config_id'" >&2
        exit 1
    }
    SEEN_CONFIGS[$config_id]=1
    group_file="$GROUP_ROOT/w$(printf '%02d' "$workers").tsv"
    GROUP_FILES[$workers]="$group_file"
    GROUP_COUNTS[$workers]=$(( ${GROUP_COUNTS[$workers]:-0} + 1 ))
    printf '%s\n' "$line" >> "$group_file"
done < "$CONFIG_FILE"

(( ${#SEEN_CONFIGS[@]} > 0 )) || {
    echo "configuration file has no data rows: $CONFIG_FILE" >&2
    exit 1
}

mapfile -t WORKER_COUNTS < <(printf '%s\n' "${!GROUP_FILES[@]}" | sort -n)
submission_file="$SUBMISSION_ROOT/${PHASE}-submission.tsv"

# A dry run should not reserve the prefix by leaving behind snapshots that
# block the real submission.  The paths are created by this invocation only;
# remove those exact generated files on every exit path.
if (( DRY_RUN )); then
    cleanup_dry_run() {
        rm -rf -- "$GROUP_ROOT"
        rm -f -- "$submission_file"
        rmdir -- "$(dirname -- "$GROUP_ROOT")" 2>/dev/null || true
        rmdir -- "$SUBMISSION_ROOT" 2>/dev/null || true
    }
    trap cleanup_dry_run EXIT
fi

printf 'phase\tworkers\trequested_cpus\tconfig_rows\tjob_id\tconfig_file\n' > "$submission_file"

echo "phase=$PHASE"
echo "prefix=$PREFIX"
echo "config=$CONFIG_FILE"
echo "constraint=$CONSTRAINT mem=$MEMORY time=$TIME_LIMIT array_limit=$ARRAY_LIMIT"
echo "stats_every=$STATS_EVERY"

for workers in "${WORKER_COUNTS[@]}"; do
    group_file="${GROUP_FILES[$workers]}"
    row_count="${GROUP_COUNTS[$workers]}"
    cpus=$((2 * workers + 2))
    array_spec="0-$((row_count - 1))%$ARRAY_LIMIT"
    export_spec="ALL,ANVIL_ROOT=$ROOT,FORGE_DIR=$FORGE_DIR_VALUE,DECK_DIR=$DECK_DIR_VALUE,BC_CKPT=$BC_CKPT_VALUE,RL_PAIRS=$RL_PAIRS_VALUE,SWEEP_CONFIG_FILE=$group_file,SWEEP_PREFIX=$PREFIX,SWEEP_SERVER_STATS_EVERY=$STATS_EVERY"
    cmd=(
        sbatch --parsable
        --job-name="anvil-bn-w$workers"
        --constraint="$CONSTRAINT"
        --cpus-per-task="$cpus"
        --mem="$MEMORY"
        --time="$TIME_LIMIT"
        --array="$array_spec"
        --export="$export_spec"
        "$ROOT/scripts/quest/rl_sweep.sbatch"
    )
    printf '[submit] workers=%s cpus=%s rows=%s array=%s\n' "$workers" "$cpus" "$row_count" "$array_spec"
    if (( DRY_RUN )); then
        printf '[dry-run]'
        printf ' %q' "${cmd[@]}"
        printf '\n'
        job_id="DRY-RUN"
    else
        job_id="$("${cmd[@]}")"
        [[ -n "$job_id" ]] || {
            echo "sbatch returned an empty job id for workers=$workers" >&2
            exit 1
        }
        echo "[submit] job_id=$job_id"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$PHASE" "$workers" "$cpus" "$row_count" "$job_id" "$group_file" \
        >> "$submission_file"
done

echo "submission_manifest=$submission_file"
if [[ "$PHASE" == phase1 ]]; then
    echo "phase2 command: $0 phase2 --prefix $PREFIX"
fi
