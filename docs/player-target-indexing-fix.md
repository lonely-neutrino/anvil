# Self-relative player-target indexing fix

Date: 2026-09-21

## Summary

The mono-red policy was using two incompatible coordinate systems for player
targets. Raw trajectory records identify players by registered seat (`pi: 0`,
`pi: 1`), while the model's player rows are ordered from the deciding
player's perspective: position 0 is always self, followed by opponents in
registered-seat order.

The dataset labels and cast-plan bridge did not consistently translate between
those systems. As a result, a policy trained to target the opponent could be
decoded as targeting itself, especially when the model was serving registered
seat 1. This was the leading explanation for the mono-red model's unusually
poor mirror performance and its apparently irrational burn decisions.

The fix standardizes the convention as:

```text
raw trajectory refs:  registered player indices
model target positions: self first, then remaining registered seats
bridge output:          model positions mapped back through the seat list
```

The corrected convention is named `self_first_registered_v1`.

## How the problem was discovered

The first signal was performance. The earlier mono-red BC evaluation won only
188 of 800 games against the heuristic, or 23.5%. The earlier mono-red RL run
improved this but still reached only about 37.1% (approximately ±3.0 percentage
points at the final evaluation), despite extensive training. This was much
worse than expected for a policy trained from the same mono-red heuristic.

The [earlier mono-red RL analysis](../data/training/mono-red-rl-4000-20260919-224841/analysis/analysis.md)
and the [earlier BC arms report](../data/training/mono-red-aggro-bc-20260919-224841/arms-report.json)
provided the initial performance evidence.

The decisive evidence came from inspecting the recorded cast targets in the
evaluation observation frames. For each burn spell, the audit classified the
target relative to the acting model seat: own player, opposing player, own
creature, or opposing creature.

| Evaluation | Burn casts | Own-player targets | Own-player rate | Own-creature targets |
|---|---:|---:|---:|---:|
| Earlier mono-red BC | 2,114 | 513 | 24.3% | 0 |
| Earlier mono-red RL | 4,390 | 593 | 13.5% | 1 |
| Heuristic comparison behavior | 4,629 | 0 | 0.0% | 0 |

The earlier BC evaluation also showed strong registered-seat asymmetry. Among
decisive games, the model won about 18.7% from seat 0 and 28.8% from seat 1.
That asymmetry was consistent with a perspective-dependent player-indexing
error rather than merely a weak red strategy.

The source of the mismatch was:

1. Raw trajectory target references remained in registered-seat coordinates.
2. `assemble` already put player features in self-first order.
3. Dataset target-label construction copied `ref["pi"]` directly into the
   model target index, instead of converting it to a self-relative position.
4. Cast-plan decoding treated `pick - n_ent` as a registered player index,
   even though the model head was operating over self-first positions.

For a two-player game from registered seat 1, the correct ordering is
`seats = [1, 0]`. Therefore model position 0 means registered player 1 (self),
and model position 1 means registered player 0 (opponent). The old bridge path
could reverse those meanings.

## How it was fixed

### Shared convention and seat helper

`anvil/encoder/transform.py` now defines:

- `PLAYER_TARGET_CONVENTION = "self_first_registered_v1"`;
- `player_seats(perspective, n_players)`, which returns the self-first seat
  ordering; and
- `player_target_position(...)`, which maps a registered player reference to
  its self-relative model position.

The same helper is used by feature assembly, dataset labels, combat metadata,
and cast-target metadata. This prevents the cast and combat paths from
silently developing different seat conventions again.

### Dataset labels

In `anvil/training/dataset.py`, player target labels are now converted with
`player_target_position` before `collate` adds the entity-row offset. Thus:

- self is always model position 0;
- the opponent is model position 1 in a two-player game; and
- entity target positions and the STOP slot are unchanged.

The raw trajectory stores do not need to be regenerated; this conversion
happens while labels are loaded.

### Bridge decoding

In `anvil/bridge/featurize.py`, bridge metadata now carries the shared
self-first `seats` list. In `anvil/bridge/server.py`, cast-plan decoding now
performs:

```python
ref.player = aux["seats"][player_pos]
```

instead of treating `player_pos` as an absolute registered seat. It also
validates the position before constructing the protobuf response.

### Checkpoint compatibility protection

`anvil/training/train.py` records the target convention in every new BC
checkpoint. The serving path and RL loading path in
`anvil/bridge/server.py` and `anvil/training/rl.py` reject checkpoints whose
metadata is missing or does not match `self_first_registered_v1`.

This is intentional: historical checkpoints were trained under the
inconsistent convention and must not silently be used as corrected models.
They remain available for retrospective analysis, but corrected BC and RL
training must start from a newly trained corrected checkpoint.

The `mu.jsonl` action representation was left unchanged because sampled RL
target records already use self-relative model positions.

### Documentation and tests

Target-head comments in `anvil/policy/model.py` and
`anvil/policy/sampling.py` now explicitly identify player positions as
self-first. The new regression coverage in
`tests/test_player_target_indexing.py` checks:

- two-player self and opponent labels from both registered seats;
- three-player seat ordering;
- seat-swapped examples producing the same relative target class;
- cast-plan decoding for `seats = [1, 0]`; and
- rejection of missing or old checkpoint-convention metadata.

The existing combat and train/serve parity tests were also retained. The full
test suite passed after the implementation: 241 passed, 34 skipped, with one
warning.

## Results after the fix

The corrected BC checkpoint is
`data/training/monoredaggro-bc-20260921-130022`. Its
[configuration](../data/training/monoredaggro-bc-20260921-130022/config.json)
records `self_first_registered_v1`, and its [evaluation report](../data/training/monoredaggro-bc-20260921-130022/arms-report.json)
contains 800 mirror games.

### Performance

| Metric | Earlier mono-red BC | Corrected mono-red BC |
|---|---:|---:|
| Model wins | 188 / 800 | 377 / 800 |
| Win rate | 23.5% | 47.1% |
| Approx. 95% CI | 20.6–26.4% | 43.7–50.6% |
| `no_shape_fit` vetoes | 77 | 16 |
| `unpayable` vetoes | 44 | 74 |
| Total veto rate | 1.32% | 0.94% |

The corrected model is therefore close to an even mirror result, and the
large improvement is far outside the earlier run's confidence interval. The
`no_shape_fit` failure mode fell substantially. `unpayable` failures increased,
so payment planning remains a separate area for improvement.

### Burn targeting

The corrected evaluation contained 2,276 model burn casts. Only 10 targeted
the model's own player: 5 from registered seat 0 and 5 from registered seat 1,
for an overall rate of 0.44%. No burn spell targeted one of the model's own
creatures.

The reduction from 24.3% own-player targeting to 0.44%, together with the
balanced 5/5 seat split, indicates that the systematic coordinate mismatch is
gone. The remaining ten cases are isolated policy decisions where the model
selected the self target; they are not evidence that the bridge is swapping
self and opponent positions.

### Registered-seat symmetry

Among decisive games, the corrected model won approximately:

- 193 / 392 = 49.2% from registered seat 0; and
- 184 / 400 = 46.0% from registered seat 1.

This is substantially more symmetric than the earlier 18.7% versus 28.8%
split and is consistent with the target decoder now respecting the deciding
player's perspective.

### Remaining issues

The indexing problem is resolved, but the evaluation was not completely clean:

- eight games still ended as `BridgePoisonedException` crashes, all in the
  registered-seat-0 evaluation arm;
- 74 `unpayable` vetoes remain; and
- the model is approximately competitive with the heuristic, not yet clearly
  stronger than it.

Those are follow-up serving/planning and policy-quality issues rather than the
self-relative player-target indexing bug.

## Operational consequence

The old mono-red and mono-green BC/RL checkpoints should not be used as inputs
to corrected serving or RL training. The raw trajectory stores can be reused,
because the corrected label conversion is performed at load time. The intended
rollout is:

1. train a corrected BC checkpoint from the existing raw trajectories;
2. verify its checkpoint metadata and mirror behavior; and
3. start RL from that corrected BC checkpoint.

