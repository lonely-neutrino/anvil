# Quest: four-deck pipeline

The four-deck pipeline is a good fit for one Quest GPU node. It starts local
Forge JVM workers and local gRPC model servers, so it does not need MPI or a
multi-node launch. The repository already contains the experiment in
[`scripts/generate_train_then_eval_constructed_four.sh`](../scripts/generate_train_then_eval_constructed_four.sh).

This guide adds only the Quest-specific pieces: a CUDA environment, the Forge
build, and a Slurm wrapper. Do not copy the local `.venv`, `data/`, model
checkpoints, or trajectory stores from the ROCm workstation.

## 1. Put the code on Quest

Use a durable project directory for the checkout and large outputs. Quest's
home filesystem is small; scratch is faster but temporary. Replace the paths
and account names below with your allocation's values.

```bash
mkdir -p /projects/<account>/<netid>/mtg-ai
cd /projects/<account>/<netid>/mtg-ai
git clone https://github.com/lonely-neutrino/anvil.git
cd anvil
git fetch origin HPC
git switch --track origin/HPC
```

The Quest helper files must be present in the branch you fetch. If this branch
has not been pushed yet, push the local commit from the workstation first, or
temporarily use a private branch with the same files.

## 2. Build the matching Forge fork

Anvil needs the fork, not upstream Forge:

```bash
module spider java
module spider maven
# Load the Java 17+ and Maven modules selected by your Quest account.
module load java/<available-version>
module load maven/<available-version>

cd /projects/<account>/<netid>/mtg-ai
git clone --filter=blob:none https://github.com/Tyrathalis/forge.git
cd forge
mvn -pl forge-gui-desktop -am package -DskipTests
```

The expected jar is under
`forge-gui-desktop/target/*-jar-with-dependencies.jar`. The Anvil harness
records the jar hash in each run manifest.

## 3. Install Python for Quest's NVIDIA driver

Discover the site module names first; do not assume the version string shown in
an example is installed on your account:

```bash
module spider python
module spider cuda
module spider git
```

Then set the actual module names and bootstrap the environment from the Anvil
checkout. Quest's GPU nodes use a driver compatible with CUDA 12.8 or earlier,
so this setup uses the official PyTorch 2.9.0 CUDA 12.8 wheel rather than the
workstation's ROCm wheel or the lockfile's newer CUDA 13 wheel.

```bash
cd /projects/<account>/<netid>/mtg-ai/anvil
export QUEST_PYTHON_MODULE=python/<available-version>
export QUEST_CUDA_MODULE=cuda/<available-12.x-version>
export QUEST_JAVA_MODULE=java/<available-17+-version>
export QUEST_GIT_MODULE=git/<available-version>
export FORGE_DIR=/projects/<account>/<netid>/mtg-ai/forge
bash scripts/quest/bootstrap_env.sh
```

The bootstrap deliberately runs `uv sync --locked --no-install-package torch`
and then installs Torch from the CUDA 12.8 index. Plain `uv sync` would
replace that wheel with the lockfile's standard build.

## 4. Install the four decks

The deck assets are already tracked in the repository:

```bash
mkdir -p "$HOME/.forge/decks/constructed"
cp docs/design/community/constructed-four/*.dck \
   "$HOME/.forge/decks/constructed/"
```

If home is not suitable for Forge's user data, use a project path instead:

```bash
export DECK_DIR=/projects/<account>/<netid>/mtg-ai/forge-decks/constructed
mkdir -p "$DECK_DIR"
cp docs/design/community/constructed-four/*.dck "$DECK_DIR/"
```

## 5. Run the end-to-end smoke

Edit the account line in
[`scripts/quest/constructed_four.sbatch`](../scripts/quest/constructed_four.sbatch)
(`REPLACE_WITH_QUEST_ACCOUNT`), then export the paths and submit the small
smoke. It generates 256 games, trains for 2,000 steps, evaluates 256 games,
and runs one RL iteration; it is intended to expose CUDA, Forge, ports, deck,
and filesystem problems before spending a long allocation.

```bash
cd /projects/<account>/<netid>/mtg-ai/anvil
export ANVIL_ROOT=$PWD
export FORGE_DIR=/projects/<account>/<netid>/mtg-ai/forge
export DECK_DIR=$HOME/.forge/decks/constructed
export QUEST_MODE=smoke
sbatch scripts/quest/constructed_four.sbatch
```

Monitor it with:

```bash
squeue -u "$USER"
tail -f anvil-4deck-<jobid>.out
```

The job's output and data remain under the Anvil checkout. Keep the generated
checkpoint and run directories; they are useful inputs for the production run.

## 6. Reproduce the current BC-to-RL recipe

Use `QUEST_MODE=bc-rl` to run the current local-style recipe without the
intermediate BC-vs-heuristic evaluation. It generates the 40,000-game
heuristic store, trains a 200,000-step BC checkpoint at
`data/training/constructed-four-bc-current`, then runs 20 RL iterations into
`data/training/constructed-four-rl-current`. The mode uses the same 2,048 games
per RL iteration, 12 rollout workers, 16-sample server batches, and 12 ms
batching window as the local `constructed-four-rl-current` configuration. If
the named output directories already exist, the job stops rather than
overwriting them.

```bash
cd /projects/<account>/<netid>/mtg-ai/anvil
export ANVIL_ROOT=$PWD
export FORGE_DIR=/projects/<account>/<netid>/mtg-ai/forge
export DECK_DIR=$HOME/.forge/decks/constructed
export QUEST_MODE=bc-rl

JOB_ID=$(sbatch --parsable scripts/quest/constructed_four.sbatch)
echo "$JOB_ID"
```

The generated RL schedule is
`data/pool/custom/constructed-four-rl-pairs-20480-current.txt`; if it does not
exist, the wrapper creates the deterministic 20,480-line schedule. This mode
matches the local recipe and hyperparameters, but a freshly generated Quest
heuristic store/checkpoint will not be bit-for-bit identical unless the Forge
build and all source artifacts are identical.

## 7. Run production self-play

After the smoke succeeds, submit the same wrapper with `QUEST_MODE=full`.
Quest's GPU partition has a 48-hour wall-time ceiling. The measured local
40-iteration recipe is longer than that, so begin with the current 20-iteration
default and keep the run name, checkpoint, and pair schedule recorded. If a
job reaches its wall limit, resume the RL loop from its latest checkpoint:

```bash
export QUEST_MODE=full
export RL_NAME=quest-4deck-<jobid>-rl
export RL_PAIRS=data/pool/custom/quest-4deck-<jobid>-rl-pairs.txt
export ARMS_PAIRS=data/pool/custom/quest-4deck-<jobid>-arms-pairs.txt
export BC_CKPT=data/training/<bc-run>/last.pt
export RL_RESUME=1
sbatch scripts/quest/constructed_four.sbatch
```

For a resume, `RL_NAME`, `RL_PAIRS`, and `BC_CKPT` must point to the previous
run; also preserve `ARMS_PAIRS` if the run has already produced arms. `RL_RESUME=1` tells the pipeline that the existing RL output is
intentional; the self-play state file then skips completed iterations. The
wrapper still creates a fresh temporary `TRAIN_OUT` for the supplied BC
checkpoint, so do not point `TRAIN_OUT` at an existing directory.

Before scaling up, compare the smoke's generation rate and iteration time with
the wall limit. If the production job is close to 48 hours, stop between
iterations by creating `data/training/<RL_NAME>/STOP`, then resubmit the same
RL command with `RL_RESUME=1`.

## Useful checks

```bash
scontrol show job <jobid>
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,AllocTRES
du -sh data/runs data/trajectories data/training data/embeddings
```

The four-deck output is sizeable. Keep code and final checkpoints in project
storage; use scratch for disposable generation stores only if your group has a
retention/cleanup plan.
