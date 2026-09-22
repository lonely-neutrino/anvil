"""Self-relative player-target labels and bridge translation."""

from collections import Counter
from types import SimpleNamespace

import torch

from anvil.bridge.server import ModelBackend
from anvil.encoder.transform import (
    PLAYER_TARGET_CONVENTION,
    player_seats,
    player_target_position,
    require_player_target_convention,
)
from anvil.training.dataset import T_MAX, collate


def _example(target_position: int):
    tgt_kind = torch.full((T_MAX + 1,), -1, dtype=torch.int64)
    tgt_idx = torch.full((T_MAX + 1,), -1, dtype=torch.int64)
    tgt_kind[:2] = torch.tensor([1, 2])
    tgt_idx[:2] = torch.tensor([target_position, 0])
    return {
        "entities": torch.zeros(1, 18),
        "ent_emb": torch.full((1,), -1, dtype=torch.int64),
        "globals": torch.zeros(10),
        "players": torch.zeros(2, 6),
        "history": torch.full((8, 3), -1, dtype=torch.int64),
        "cand_rows": torch.tensor([-1]),
        "cand_sa": torch.tensor([-1]),
        "cand_kind": torch.tensor([-1]),
        "cand_paykind": torch.tensor([-1]),
        "label": torch.tensor(0),
        "label_row": torch.tensor(-1),
        "tgt_kind": tgt_kind,
        "tgt_idx": tgt_idx,
        "x_val": torch.tensor(-1),
        "task": torch.tensor(0),
        "bool_label": torch.tensor(-1),
        "num_label": torch.tensor(-1),
        "num_lo": torch.tensor(0),
        "num_hi": torch.tensor(17),
        "ctx_row": torch.tensor(-1),
        "forced": torch.tensor(0),
        "has_outcome": torch.tensor(0),
        "won": torch.tensor(0),
        "cmb_rows": torch.empty(0, dtype=torch.int64),
        "cmb_count": torch.empty(0, dtype=torch.int64),
        "cmb_count_label": torch.empty(0, dtype=torch.int64),
        "blk_atk_rows": torch.empty(0, dtype=torch.int64),
        "atk_label": torch.empty(0, dtype=torch.int64),
        "atk_tgt_kind": torch.empty(0, dtype=torch.int64),
        "atk_tgt_idx": torch.empty(0, dtype=torch.int64),
        "blk_label": torch.empty(0, dtype=torch.int64),
    }


def test_player_seats_and_target_positions_are_self_first():
    assert player_seats(0, 2) == [0, 1]
    assert player_seats(1, 2) == [1, 0]
    assert player_seats(2, 3) == [2, 0, 1]

    assert player_target_position(0, 0, 2) == 0
    assert player_target_position(0, 1, 2) == 1
    assert player_target_position(1, 1, 2) == 0
    assert player_target_position(1, 0, 2) == 1
    assert player_target_position(2, 0, 3) == 1
    assert player_target_position(2, 1, 3) == 2


def test_collate_keeps_same_relative_target_class_when_seats_swap():
    # One entity row precedes two self-first player positions, so the
    # opponent target is class n+1 for either registered seat.
    out = collate([_example(1), _example(1)])
    assert out["tgt_labels"][:, 0].tolist() == [2, 2]
    assert out["tgt_labels"][:, 1].tolist() == [3, 3]  # STOP = n + p


def test_castplan_maps_self_first_positions_back_to_registered_seats():
    out = {
        "choice": torch.tensor([1]),
        "n_ent": torch.tensor(0),
        "stop_idx": 2,
        "tgt_picks": torch.tensor([[0, 1, 2]]),
        "x_cls": torch.tensor([0]),
    }
    aux = {
        "cand_first_opt": [-1, 0],
        "row_min_id": {},
        "stack_ids": set(),
        "seats": [1, 0],
    }
    backend = SimpleNamespace(n_sa=1, counts=Counter())

    plan = ModelBackend._castplan(backend, out, aux)

    assert [ref.player for ref in plan.target_refs] == [1, 0]


def test_target_convention_is_named():
    assert PLAYER_TARGET_CONVENTION == "self_first_registered_v1"


def test_old_or_missing_checkpoint_convention_is_rejected():
    require_player_target_convention(
        {"player_target_convention": PLAYER_TARGET_CONVENTION}, "test checkpoint"
    )
    try:
        require_player_target_convention({}, "test checkpoint")
    except ValueError as exc:
        assert "retrain from corrected labels" in str(exc)
    else:
        raise AssertionError("missing target convention was accepted")
