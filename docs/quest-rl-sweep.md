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
