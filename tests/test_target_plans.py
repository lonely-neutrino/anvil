from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from anvil.bridge.server import ModelBackend
from anvil.policy.model import AnvilNet
from anvil.policy.target_plans import target_plan_fields
from anvil.training.dataset import T_MAX


def _fields(opts, wire=(0, 1), rows=None, n_cands=2):
    return target_plan_fields(
        opts, list(wire), n_cands, rows or {10: 0, 11: 1}, 0, 2, T_MAX
    )


def test_complete_plan_maps_players_entities_and_concrete_refs():
    opts = [{
        "tc": 1,
        "tp": [{"xm": 5, "r": [{"p": 1}, {"e": 11, "ns": 1}]}],
    }]
    fields, aux = _fields(opts)
    assert fields["tp_enforce"].tolist() == [False, True]
    assert fields["tp_cand_allow"].tolist() == [True, True]
    assert fields["tp_kind"].tolist() == [[1, 0, 2, -1, -1]]
    assert fields["tp_idx"].tolist() == [[1, 1, 0, -1, -1]]
    assert aux["target_plans"][0]["refs"] == [{"p": 1}, {"e": 11, "ns": 1}]


def test_fallback_alias_disables_canonical_enforcement():
    opts = [
        {"tc": 1, "tp": [{"xm": 3, "r": [{"e": 10}]}]},
        {"tc": 0, "tr": "modal"},
    ]
    fields, aux = _fields(opts, wire=(0, 1, 1))
    assert fields["tp_enforce"].tolist() == [False, False]
    assert fields["tp_cand_allow"].tolist() == [True, True]
    assert aux["target_wire_complete"] == [False, True, False]


def test_authoritative_empty_plan_masks_candidate():
    fields, _ = _fields([{"tc": 1, "tp": []}])
    assert fields["tp_enforce"].tolist() == [False, True]
    assert fields["tp_cand_allow"].tolist() == [True, False]


def test_prefix_and_x_masks_follow_surviving_plans():
    batch = {
        "tp_mask": torch.tensor([[True, True]]),
        "tp_cand": torch.tensor([[1, 1]]),
        "tp_wire": torch.tensor([[1, 1]]),
        "tp_tokens": torch.tensor([[[2, 5, -1], [3, 5, -1]]]),
        "tp_xmask": torch.tensor([[1 << 2, 1 << 4]]),
        "tp_enforce": torch.tensor([[False, True]]),
        "tp_forced_wire": torch.tensor([-1]),
    }
    active, enforce = AnvilNet._tp_start(batch, torch.tensor([1]))
    assert enforce.tolist() == [True]
    assert AnvilNet._tp_allowed(batch, active, 0, 6).tolist() == [
        [False, False, True, True, False, False]
    ]
    active = AnvilNet._tp_advance(batch, active, 0, torch.tensor([3]), torch.tensor([True]))
    assert active.tolist() == [[False, True]]
    assert AnvilNet._tp_allowed(batch, active, 1, 6).tolist() == [
        [False, False, False, False, False, True]
    ]
    assert AnvilNet._tp_x_allowed(batch, active, 6).tolist() == [
        [False, False, False, False, True, False]
    ]


def test_castplan_uses_exact_wire_and_concrete_plan():
    backend = SimpleNamespace(torch=torch, counts=Counter(), n_sa=10)
    out = {
        "choice": torch.tensor([1]),
        "tp_plan_idx": torch.tensor([0]),
        "x_cls": torch.tensor([4]),
        "n_ent": 2,
        "stop_idx": 4,
        "tgt_picks": torch.tensor([[0, 4, 4, 4, 4]]),
    }
    aux = {
        "cand_first_opt": [-1, 0],
        "target_plans": [{"wire": 2, "refs": [{"e": 99}, {"p": 1}]}],
    }
    cp = ModelBackend._castplan(backend, out, aux)
    assert cp.spell_option == 2
    assert [(r.entity, r.player) for r in cp.target_refs] == [(99, 0), (0, 1)]
    assert cp.x_value == 4


def test_authoritative_cast_without_matching_plan_is_loud_decline():
    backend = SimpleNamespace(torch=torch, counts=Counter(), n_sa=10)
    out = {
        "choice": torch.tensor([1]),
        "tp_plan_idx": torch.tensor([-1]),
        "tp_enforced": torch.tensor([True]),
        "x_cls": torch.tensor([0]),
        "n_ent": 1,
        "stop_idx": 3,
        "tgt_picks": torch.tensor([[3, 3, 3, 3, 3]]),
    }
    aux = {"cand_first_opt": [-1, 0], "target_plans": []}
    with pytest.raises(ValueError, match="no matching plan"):
        ModelBackend._castplan(backend, out, aux)
    assert backend.counts["target_mask_missing_match"] == 1
