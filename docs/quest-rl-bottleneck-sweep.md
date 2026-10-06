# Quest moderate-worker RL bottleneck sweep

This study separates rollout-worker scaling from model-server batching while
requesting only the CPU capacity needed by each worker count. It targets the
A100 2000-series (`quest12`) nodes without `--exclusive`.

The screen is intentionally short: one startup-inclusive RL iteration, 2,048
games, `replay=1`, and no arms evaluation. Treat it as a throughput screen,
not as a production-training result.

## Shared inputs

From the Anvil checkout on Quest:

```bash
cd /gpfs/projects/p31830/mtg-ai/anvil
export ANVIL_ROOT=$PWD
export FORGE_DIR=/gpfs/projects/p31830/mtg-ai/forge
export DECK_DIR=$HOME/.forge/decks/constructed
export BC_CKPT=$PWD/data/training/constructed-four-bc-current/last.pt
export RL_PAIRS=$PWD/data/pool/custom/constructed-four-rl-pairs-20480-current.txt
```

The submitter uses 128 GB, a three-hour limit, two simultaneous tasks per
worker group, and a 15-second server telemetry interval by default. Override
those with `--mem`, `--time`, `--array-limit`, or `--stats-every` if needed.

## Phase 1

Phase 1 tests 8, 12, 16, 20, and 24 rollout workers at
`max_batch=16`, `batch_window_ms=12`, and `chunk=15`.

Use one shared prefix so Phase 2 can compare its repeated controls:

```bash
export SWEEP_PREFIX=constructed-four-rl-bottleneck-$(date +%Y%m%d-%H%M%S)
bash scripts/quest/submit_rl_bottleneck_sweep.sh phase1 --prefix "$SWEEP_PREFIX" --dry-run
bash scripts/quest/submit_rl_bottleneck_sweep.sh phase1 --prefix "$SWEEP_PREFIX"
```

The five worker groups request 18, 26, 34, 42, and 50 CPUs respectively.
The generated submission manifest is:

```text
data/training/sweeps/$SWEEP_PREFIX/phase1-submission.tsv
```

Each group is a separate Slurm array because an array cannot vary
`--cpus-per-task` between its elements.

## Phase 2

After checking that Phase 1 completed and did not show systematic bridge or
resource problems, submit the 27-row batching screen:

```bash
bash scripts/quest/submit_rl_bottleneck_sweep.sh phase2 --prefix "$SWEEP_PREFIX"
```

This crosses workers `8,16,24` with `max_batch=8,16,32` and
`batch_window_ms=3,12,24`. The `mb16/t12` row at each worker count repeats a
Phase 1 control.

## Monitoring and summary

The ordinary Slurm logs remain available as `anvil-rl-sweep-<job>_<task>.out`
and `.err`. Each configuration also records diagnostics under:

```text
data/training/sweeps/$SWEEP_PREFIX/<config-id>/
```

Important files include `hardware-start.json`, `hardware-end.json`,
`gpu-utilization.csv`, `cpu-vmstat.log`, `resource.txt`, `run.out`,
`run.err`, and `timing.txt`. Model-server occupancy lines are in the RL
run's `iter-*/server.log` (and, if enabled, `iter-*/arms-server.log`).

Summarize both phases together after Phase 2:

```bash
.venv/bin/python scripts/quest/summarize_rl_bottleneck_sweep.py \
  --root data/training \
  --prefix "$SWEEP_PREFIX" \
  --config-file scripts/quest/rl_bottleneck_phase1.tsv \
  --config-file scripts/quest/rl_bottleneck_phase2.tsv \
  --out "data/training/$SWEEP_PREFIX-summary.tsv"
```

The summary is startup-inclusive and reports generation/training time,
throughput, CPU/GPU telemetry, server batch/queue telemetry, node/GPU
identity, errors, and repeated-control drift. Draws are reported but do not
by themselves make a run contaminated.
