# Quest RL throughput sweep

This study uses Slurm array jobs to test several independent RL throughput
configurations without manually waiting for one run before submitting the
next. Each array task gets one GPU, a unique RL output name, and a job-derived
model-server port range. All tasks can share the same BC checkpoint and pair
schedule read-only.

The first screen is intentionally short:

- two RL iterations;
- 2,048 rollout games per iteration;
- `RL_REPLAY=1` to keep the screening workload small;
- no arms evaluations (`RL_ARMS_EVERY=0`);
- initial launch ramp fixed at 4 seconds;
- replacement launch ramp varied by the table.

`RL_REPLAY=1` is only a screening choice. Validate the winning settings with
the production `RL_REPLAY=4` before treating them as the final recipe.

## Generation and serving screen

After pushing the `HPC` branch and pulling it onto Quest, set the shared input
paths. The checkpoint and pair schedule are read-only inputs to every task:

```bash
cd /gpfs/projects/p31830/mtg-ai/anvil
export ANVIL_ROOT=$PWD
export FORGE_DIR=/gpfs/projects/p31830/mtg-ai/forge
export DECK_DIR=$HOME/.forge/decks/constructed
export BC_CKPT=$PWD/data/training/constructed-four-bc-current/last.pt
export RL_PAIRS=$PWD/data/pool/custom/constructed-four-rl-pairs-20480-current.txt
export SWEEP_CONFIG_FILE=$PWD/scripts/quest/rl_sweep_generation.tsv

SWEEP_JOB=$(sbatch --parsable --array=0-12%4 scripts/quest/rl_sweep.sbatch)
echo "$SWEEP_JOB"
```

The `%4` limits the sweep to four simultaneous GPU jobs. Remove or increase
that limit only if Quest availability and fairshare make that appropriate.

Monitor the array with:

```bash
squeue -j "$SWEEP_JOB"
tail -f "anvil-rl-sweep-${SWEEP_JOB}_0.out"
```

When the tasks finish, summarize them. The `--skip-first` option excludes the
startup-heavy iteration 0 from speed columns, but startup errors remain in the
error columns:

```bash
.venv/bin/python scripts/quest/summarize_rl_sweep.py \
  --root data/training \
  --prefix "constructed-four-rl-sweep-${SWEEP_JOB}" \
  --config-file scripts/quest/rl_sweep_generation.tsv \
  --skip-first \
  --out "data/training/constructed-four-rl-sweep-${SWEEP_JOB}.tsv"
```

Prefer rows with `clean=1`, then compare `generation_games_per_hour` and
`combined_games_per_hour`. Inspect `statuses` and the `.err` files for any
bridge exceptions that were not represented in a monitor row.

## Learner screen

The learner table holds generation at the baseline and varies learner workers
and segment size:

```bash
export SWEEP_CONFIG_FILE=$PWD/scripts/quest/rl_sweep_training.tsv
LEARNER_JOB=$(sbatch --parsable --array=0-5%4 scripts/quest/rl_sweep.sbatch)
echo "$LEARNER_JOB"
```

Summarize it with the training table:

```bash
.venv/bin/python scripts/quest/summarize_rl_sweep.py \
  --root data/training \
  --prefix "constructed-four-rl-sweep-${LEARNER_JOB}" \
  --config-file scripts/quest/rl_sweep_training.tsv \
  --skip-first \
  --out "data/training/constructed-four-rl-sweep-${LEARNER_JOB}.tsv"
```

For this screen, compare `train_s` and `train_windows_per_second`; bridge
errors should remain zero because the generation settings are held constant.

## Validation

After selecting a few candidates, make a small validation TSV by copying the
winning rows and changing `replay` to `4`, `iterations` to at least `3` or `4`,
and `arms_every` to the production value. Submit that table with the same
launcher and use the measured steady-state rows to choose the final recipe.

Do not run two tasks with the same `SWEEP_PREFIX` and configuration IDs unless
you intentionally want them to collide; the launcher deliberately fails when
an output directory already exists.

## Controlled aggressive paired sweep

The independent array above is useful for a broad screen, but separate Slurm
allocations can land on different Quest nodes or GPUs.  For a cleaner timing
comparison, the paired launcher runs a baseline and a candidate sequentially in
the same one-GPU allocation.  It also records the node/GPU identity, a
10-second `nvidia-smi` utilization log, `vmstat` CPU samples, and start/end
hardware metadata for each side of the pair.

The first controlled screen repeats the current `w24/c15/r0` baseline and
tests a smaller chunk plus 28, 32, and 36 rollout workers.  The 28--36 worker
rows are deliberately stress tests: the current worker manifests advertise two
CPU threads each, so 24 workers already consume the 48-CPU allocation.  A
candidate above 24 may improve throughput only if its extra concurrency helps
more than the resulting oversubscription costs.

```bash
cd /gpfs/projects/p31830/mtg-ai/anvil
export ANVIL_ROOT=$PWD
export FORGE_DIR=/gpfs/projects/p31830/mtg-ai/forge
export DECK_DIR=$HOME/.forge/decks/constructed
export BC_CKPT=$PWD/data/training/constructed-four-bc-current/last.pt
export RL_PAIRS=$PWD/data/pool/custom/constructed-four-rl-pairs-20480-current.txt
export PAIRED_CONFIG_FILE=$PWD/scripts/quest/rl_paired_aggressive.tsv
export PAIRED_PAIRS_FILE=$PWD/scripts/quest/rl_paired_aggressive_pairs.tsv

PAIRED_JOB=$(sbatch --parsable --array=0-4%2 scripts/quest/rl_paired_sweep.sbatch)
echo "$PAIRED_JOB"
```

The `%2` allows two pairs at a time.  If you want to remove node-level
contention from the comparison and Quest policy permits it, submit with
`sbatch --exclusive --parsable --array=0-4%2 ...`; otherwise the paired node
and GPU checks still make each baseline/candidate comparison internally
matched.  Monitor a pair with:

```bash
squeue -j "$PAIRED_JOB"
tail -f "anvil-rl-paired-${PAIRED_JOB}_0.out"
```

Summarize after (or during) the array:

```bash
.venv/bin/python scripts/quest/summarize_rl_paired_sweep.py \
  --root data/training \
  --prefix "constructed-four-rl-paired-${PAIRED_JOB}" \
  --pairs-file scripts/quest/rl_paired_aggressive_pairs.tsv \
  --config-file scripts/quest/rl_paired_aggressive.tsv \
  --skip-first \
  --out "data/training/constructed-four-rl-paired-${PAIRED_JOB}.tsv"
```

Use rows with `paired_clean=1` for decisions.  `candidate_gen_delta_pct` and
`candidate_combined_delta_pct` are the candidate's percentage change from its
same-pair baseline; positive generation change is faster, while a negative
`candidate_train_delta_pct` means the candidate trained in fewer seconds.
`same_node` and `same_gpu` should both be 1.  The `gpu_*` and `cpu_busy_*`
columns are averages from the per-run diagnostic files, and the raw files are
under `data/training/paired-sweeps/<prefix>/<pair-id>/`.

If the aggressive screen finds a winner, edit the worker and chunk columns in
`rl_paired_serving.tsv` to that winner and submit the seven-pair serving
follow-up:

```bash
export PAIRED_CONFIG_FILE=$PWD/scripts/quest/rl_paired_serving.tsv
export PAIRED_PAIRS_FILE=$PWD/scripts/quest/rl_paired_serving_pairs.tsv
SERVING_JOB=$(sbatch --parsable --array=0-6%2 scripts/quest/rl_paired_sweep.sbatch)
echo "$SERVING_JOB"
```

That follow-up varies replacement delay, server count, model batch, and batch
window while keeping the selected rollout settings fixed.  Finally, validate
the chosen settings with the production replay/arms schedule (`replay=4`, at
least four iterations, and the normal `arms_every`) before using them for a
long run.
