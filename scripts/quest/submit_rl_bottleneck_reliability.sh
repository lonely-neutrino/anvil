#!/usr/bin/env bash
# Submit the fixed reliability check for the moderate-worker RL sweep.
#
# Usage:
#   scripts/quest/submit_rl_bottleneck_reliability.sh
#   scripts/quest/submit_rl_bottleneck_reliability.sh --prefix NAME
#   scripts/quest/submit_rl_bottleneck_reliability.sh --dry-run
#
# Additional options are passed to submit_rl_bottleneck_sweep.sh, including
# --constraint, --mem, --time, --stats-every, and --array-limit.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

export ANVIL_ROOT="$ROOT"
export BOTTLENECK_CONFIG_FILE="$SCRIPT_DIR/rl_bottleneck_reliability.tsv"

exec "$SCRIPT_DIR/submit_rl_bottleneck_sweep.sh" phase1 "$@"
