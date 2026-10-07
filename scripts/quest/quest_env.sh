#!/usr/bin/env bash
# Set up the Quest shell used to build Forge and submit Anvil jobs.
#
# This file is meant to be SOURCED, not executed:
#
#   source scripts/quest/quest_env.sh
#
# The defaults below are the toolchain paths observed in successful Quest
# jobs.  Override QUEST_GIT_ROOT, QUEST_JAVA_HOME, or QUEST_MAVEN_HOME before
# sourcing if Quest changes those installations.

# A child process cannot modify its parent's environment.  Give a useful
# error instead of silently doing nothing when someone runs ./quest_env.sh.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "source this file instead: source $0" >&2
    exit 2
fi

_quest_env_fail() {
    echo "[quest-env] ERROR: $*" >&2
    return 1
}

_quest_env_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ANVIL_ROOT="$(cd -- "$_quest_env_script_dir/../.." && pwd)"
export FORGE_DIR="${FORGE_DIR:-$ANVIL_ROOT/../forge}"
export PYTHON="${PYTHON:-$ANVIL_ROOT/.venv/bin/python}"
export DECK_DIR="${DECK_DIR:-$HOME/.forge/decks/constructed}"

# These are the paths visible in the successful Quest job environments.
QUEST_GIT_ROOT="${QUEST_GIT_ROOT:-/software/git/2.37.2}"
export JAVA_HOME="${QUEST_JAVA_HOME:-/software/java/jdk-17.0.2+8}"
QUEST_MAVEN_HOME="${QUEST_MAVEN_HOME:-/hpc/software/apache-maven-mvn3/apache-maven-3.9.7}"
export MAVEN_HOME="$QUEST_MAVEN_HOME"

_quest_prepend_path() {
    local directory="$1"
    [[ -d "$directory" ]] || return 0
    case ":${PATH:-}:" in
        *":$directory:"*) ;;
        *) PATH="$directory${PATH:+:$PATH}" ;;
    esac
}

# Put the known-good Java before anything Maven may have added for Java 11.
_quest_prepend_path "$QUEST_GIT_ROOT/bin"
_quest_prepend_path "$QUEST_GIT_ROOT/libexec/git-core"
_quest_prepend_path "$JAVA_HOME/bin"
_quest_prepend_path "$QUEST_MAVEN_HOME/bin"
export PATH
hash -r

# The batch wrappers reload these variables when they are nonempty.  This
# setup uses the explicit paths above, so clear stale Java/Git module names
# from an older shell rather than allowing one to reintroduce Java 11.
export QUEST_JAVA_MODULE=""
export QUEST_GIT_MODULE=""

[[ -x "$QUEST_GIT_ROOT/bin/git" ]] ||
    _quest_env_fail "Git installation not found at $QUEST_GIT_ROOT" || return 1
[[ -x "$JAVA_HOME/bin/java" ]] ||
    _quest_env_fail "Java installation not found at $JAVA_HOME" || return 1
[[ -x "$QUEST_MAVEN_HOME/bin/mvn" ]] ||
    _quest_env_fail "Maven installation not found at $QUEST_MAVEN_HOME" || return 1
[[ -x "$PYTHON" ]] ||
    _quest_env_fail "Python environment not found at $PYTHON" || return 1

java_major="$(java -version 2>&1 | sed -n 's/.*version "\([0-9][0-9]*\).*/\1/p' | head -1)"
[[ "$java_major" =~ ^[0-9]+$ ]] ||
    _quest_env_fail "could not determine Java version" || return 1
(( java_major >= 17 )) ||
    _quest_env_fail "Java 17+ required, but java reports major version $java_major" || return 1

echo "[quest-env] ANVIL_ROOT=$ANVIL_ROOT"
echo "[quest-env] FORGE_DIR=$FORGE_DIR"
echo "[quest-env] PYTHON=$PYTHON"
echo "[quest-env] JAVA_HOME=$JAVA_HOME"
echo "[quest-env] git=$(command -v git)"
echo "[quest-env] java=$(command -v java)"
echo "[quest-env] mvn=$(command -v mvn)"
git --version
java -version
mvn -version | sed -n '1,4p'

# This checks the login-node pieces only.  GPU availability is checked again
# inside the allocated GPU job by the RL wrapper.
if [[ "${QUEST_ENV_SKIP_CHECK:-0}" != 1 && -f "$ANVIL_ROOT/scripts/quest/check_environment.py" ]]; then
    "$PYTHON" "$ANVIL_ROOT/scripts/quest/check_environment.py" \
        --forge-dir "$FORGE_DIR" \
        --require-jar || return 1
fi

cd "$ANVIL_ROOT"
export PYTHONUNBUFFERED=1
export QUEST_ENV_READY=1
echo "[quest-env] ready; current directory: $PWD"

unset -f _quest_prepend_path _quest_env_fail
