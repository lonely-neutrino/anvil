"""M11 single-color action surface: fixed WUBRG classes and RL replay wiring."""

import torch

from anvil.policy.sampling import make_noise, mu_record
from anvil.bridge.featurize import TAG_TASK
from anvil.bridge.server import MODEL_TAGS
from anvil.training.dataset import (
    COLOR_CLASSES,
    COLOR_INDEX,
    TASKS,
    T_MAX,
    X_CLASSES,
    color_class,
    collate,
)
from anvil.training.rl import apply_mu_labels, composite_entropy, composite_logp, mu_matches


def _example(mask=None):
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
        "tgt_kind": torch.full((T_MAX + 1,), -1, dtype=torch.int64),
        "tgt_idx": torch.full((T_MAX + 1,), -1, dtype=torch.int64),
        "x_val": torch.tensor(-1),
        "task": torch.tensor(TASKS["choose_color"]),
        "bool_label": torch.tensor(-1),
        "num_label": torch.tensor(-1),
        "num_lo": torch.tensor(0),
        "num_hi": torch.tensor(X_CLASSES - 1),
        "ctx_row": torch.tensor(-1),
        "forced": torch.tensor(0),
        "has_outcome": torch.tensor(0),
        "won": torch.tensor(0),
        "color_mask": torch.tensor(
            [True, False, True, False, False] if mask is None else mask, dtype=torch.bool
        ),
        "color_label": torch.tensor(-1),
        "cmb_rows": torch.empty(0, dtype=torch.int64),
        "cmb_count": torch.empty(0, dtype=torch.int64),
        "cmb_count_label": torch.empty(0, dtype=torch.int64),
        "blk_atk_rows": torch.empty(0, dtype=torch.int64),
        "atk_label": torch.empty(0, dtype=torch.int64),
        "atk_tgt_kind": torch.empty(0, dtype=torch.int64),
        "atk_tgt_idx": torch.empty(0, dtype=torch.int64),
        "blk_label": torch.empty(0, dtype=torch.int64),
    }


def test_color_classes_are_canonical_wubrg():
    assert COLOR_INDEX == {"white": 0, "blue": 1, "black": 2, "red": 3, "green": 4}
    assert COLOR_CLASSES == 5
    assert [color_class(x) for x in ("white", "U", "black", "r", "green")] == [0, 1, 2, 3, 4]
    assert color_class("colorless") is None


def test_color_tag_is_model_routed():
    assert TAG_TASK["mtg.choose_color"] == "choose_color"
    assert "mtg.choose_color" in MODEL_TAGS.split(",")


def test_color_sampling_noise_and_replay_round_trip():
    import torch.nn.functional as F

    ex = _example()
    noise = make_noise(ex, "choose_color", seed=123)
    assert noise["color"].shape == (COLOR_CLASSES,)

    batch = collate([ex])
    assert batch["color_mask"].tolist() == [[True, False, True, False, False]]
    logits = torch.tensor([[0.0, -1e9, 2.0, -1e9, -1e9]])
    out = {
        "n_ent": 1,
        "stop_idx": 4,
        "color": torch.tensor([2]),
        "logp_color": F.log_softmax(logits, dim=-1).gather(1, torch.tensor([[2]])).squeeze(1),
        "ent_color": torch.tensor([0.3653]),
    }
    rec = mu_record(7, 11, "choose_color", ex, {}, out)
    assert rec["c"] == 2 and rec["task"] == "choose_color"
    ex2 = apply_mu_labels(dict(ex), rec)
    assert ex2["color_label"].item() == 2
    assert mu_matches(ex2, rec)
    assert not mu_matches(ex2, {"task": "choose_color", "c": 1})


def test_color_contributes_only_on_color_rows():
    import torch.nn.functional as F

    ex = _example()
    ex = apply_mu_labels(ex, {"task": "choose_color", "c": 2})
    batch = collate([ex])
    fwd = {
        "policy_logits": torch.zeros(1, 1),
        "tgt_logits": torch.zeros(1, T_MAX + 1, 4),
        "x_logits": torch.zeros(1, X_CLASSES),
        "bool_logit": torch.zeros(1),
        "num_logits": torch.zeros(1, X_CLASSES),
        "color_logits": torch.tensor([[0.0, -1e9, 2.0, -1e9, -1e9]]),
        "atk_logits": torch.zeros(1, 1),
        "cmb_count_logits": torch.zeros(1, 1, 12),
        "atk_tgt_logits": torch.zeros(1, 1, 3),
        "blk_logits": torch.zeros(1, 1, 2),
    }
    terms = composite_logp(fwd, batch)
    expected = F.log_softmax(fwd["color_logits"], dim=-1)[0, 2]
    assert torch.allclose(terms["color"][0], expected)
    assert torch.allclose(terms["logp"][0], expected)
    assert composite_entropy(fwd, batch)[0] > 0
