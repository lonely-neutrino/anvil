#!/usr/bin/env bash
# Run the mono-red pipeline recipe for any one Constructed deck mirror match.
#
# The implementation lives in generate_train_then_rl_mono_red.sh so that the
# two entry points stay behaviorally identical. This wrapper supplies a deck-
# derived identity for the manifest, embeddings, runs, and output directories.
#
# Usage:
#   ./scripts/generate_train_then_rl_single_deck.sh monoGreenStompy.dck
#
# The deck must already be present in the Forge constructed deck store. Set
# DECK_DIR when using a non-default store. Most pipeline variables from the
# underlying script remain overridable through the environment.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if (( $# > 1 )); then
    echo "usage: $0 DECK_FILE" >&2
    exit 2
fi

DECK_INPUT="${1:-${DECK:-}}"
if [[ -z "$DECK_INPUT" ]]; then
    echo "usage: $0 DECK_FILE" >&2
    echo "example: $0 monoGreenStompy.dck" >&2
    exit 2
fi

DECK="${DECK_INPUT##*/}"
DECK_STEM="${DECK%.dck}"

if [[ "$DECK_INPUT" == */* ]]; then
    DECK_DIR="${DECK_DIR:-${DECK_INPUT%/*}}"
else
    DECK_DIR="${DECK_DIR:-/home/lonelyneutrino/.forge/decks/constructed}"
fi

DECK_SLUG="${PIPELINE_SLUG:-$DECK_STEM}"
DECK_SLUG="$(printf '%s' "$DECK_SLUG" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//')"
[[ -n "$DECK_SLUG" ]] || DECK_SLUG="single-deck"

RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
GEN_GAMES="${GEN_GAMES:-20000}"

POOL_VERSION="${POOL_VERSION:-single-deck-${DECK_SLUG}-v1}"
POOL_MANIFEST="${POOL_MANIFEST:-data/pool/custom/${POOL_VERSION}.json}"
EMBED="${EMBED:-data/embeddings/${POOL_VERSION}-bge-m3}"
GEN_PURPOSE="${GEN_PURPOSE:-${DECK_SLUG}-heur-${GEN_GAMES}-${RUN_STAMP}}"
TRAIN_OUT="${TRAIN_OUT:-data/training/${DECK_SLUG}-bc-${RUN_STAMP}}"
EVAL_PREFIX="${EVAL_PREFIX:-${DECK_SLUG}-bc-${RUN_STAMP}}"
RL_NAME="${RL_NAME:-${DECK_SLUG}-rl-${RUN_STAMP}}"
ARMS_PAIRS="${ARMS_PAIRS:-data/pool/custom/${DECK_SLUG}-pairs.txt}"

export DECK DECK_DIR POOL_VERSION POOL_MANIFEST EMB GEN_GAMES
export GEN_PURPOSE TRAIN_OUT EVAL_PREFIX RL_NAME ARMS_PAIRS

exec "$ROOT/scripts/generate_train_then_rl_mono_red.sh"
