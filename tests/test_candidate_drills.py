"""Pure contracts for generic counterfactual priority-candidate drills."""

import json

import pytest

from anvil.training.candidate_labels import (
    candidate_inverse_frequency_weights,
    load_rows,
)
from anvil.training.dataset import norm_sa


def test_x_suffix_normalization_is_state_independent():
    assert norm_sa("Brave the Elements (X=0)") == "Brave the Elements"
    assert norm_sa("Brave the Elements (X=7)") == norm_sa("Brave the Elements (X=1)")


def test_miner_deduplicates_wire_options_by_entity_and_normalized_sa(tmp_path, monkeypatch):
    import scripts.mine_priority_candidates as miner

    class Game:
        header = {"g": 17}
        decisions = [
            {
                "m": "chooseSpellAbilityToPlay",
                "s": 4,
                "p": 1,
                "t": 7,
                "phase": "COMBAT_DECLARE_ATTACKERS",
                "opts": [
                    {"e": 1234, "sa": "Brave the Elements (X=0)"},
                    {"e": 1234, "sa": "Brave the Elements (X=3)"},
                    {"e": 5678, "sa": "Other spell"},
                ],
            }
        ]

    class Store:
        def game_indices(self):
            return [17]

        def game(self, _game):
            return Game()

    monkeypatch.setattr(miner, "open_store", lambda _path: Store())
    out = tmp_path / "points.jsonl"
    stats = miner.mine(
        [tmp_path / "source-store"],
        out,
        model_seats={1},
        sa_pattern="Brave the Elements",
    )
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert stats["points"] == 1
    assert len(rows) == 1
    assert rows[0]["store"] == "source-store"
    assert rows[0]["window"] == 4
    assert rows[0]["candidate"] == {"entity": 1234, "sa": "Brave the Elements"}


def test_miner_excludes_nonempty_stack_priority_windows(tmp_path, monkeypatch):
    import scripts.mine_priority_candidates as miner

    class Game:
        header = {}
        decisions = [
            {
                "m": "chooseSpellAbilityToPlay",
                "s": 9,
                "p": 0,
                "obs": {
                    "stack": [{"e": 99, "lbl": "trigger"}],
                    "ents": [{"e": 99, "z": "stack"}],
                },
                "opts": [{"e": 1234, "sa": "Brave the Elements"}],
            }
        ]

    class Store:
        def game_indices(self):
            return [3]

        def game(self, _game):
            return Game()

    monkeypatch.setattr(miner, "open_store", lambda _path: Store())
    out = tmp_path / "points.jsonl"
    stats = miner.mine([tmp_path / "source-store"], out, model_seats={0})
    assert stats["stack_filtered"] == 1
    assert out.read_text() == ""


def test_candidate_labels_keep_multiple_arms_and_clip_signed_advantage(tmp_path):
    path = tmp_path / "labels.jsonl"
    rows = [
        {
            "source": "source-store",
            "i": 17,
            "window": 4,
            "candidate": {"entity": 1234, "sa": "Brave the Elements"},
            "paired": 32,
            "natural_wins": 0,
            "forced_wins": 32,
            "forced_realized": 31,
            "forced_total": 32,
        },
        {
            "source": "source-store",
            "i": 17,
            "window": 4,
            "candidate": {"entity": 5678, "sa": "Another spell"},
            "paired": 32,
            "natural_wins": 32,
            "forced_wins": 0,
            "forced_realized": 32,
            "forced_total": 32,
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    got = load_rows([str(path)])
    assert len(got) == 2
    assert {row["key"] for row in got} == {("source-store", 17, 4)}
    assert sorted(row["adv"] for row in got) == [-0.25, 0.25]
    assert load_rows.last_stats["valid"] == 2


def test_candidate_pass_uses_one_hot_target_and_signed_contrast():
    torch = pytest.importorskip("torch")
    from anvil.training.rl import candidate_pass

    logits = torch.tensor([[0.0, 1.0, -1.0], [0.5, 0.5, 0.5]], requires_grad=True)
    seg = {
        "candidate_adv": torch.tensor([0.2, -0.1]),
        "candidate_tmask": torch.tensor([[False, True, False], [False, False, True]]),
        "x": torch.zeros(2),
    }
    fwd = {"policy_logits": logits}

    def fake_forward_segments(_net, segs, grad):
        assert grad is False
        yield segs[0], fwd

    raw = candidate_pass(None, [seg], fake_forward_segments, 1.0, grad=False)
    lp = logits.log_softmax(1)
    expected = -(0.2 * (lp[0, 1] - lp[0, 0]) + (-0.1) * (lp[1, 2] - lp[1, 0])) / 2
    assert abs(raw - float(expected.detach())) < 1e-6
    assert all(int(row.sum()) == 1 for row in seg["candidate_tmask"])



def test_featurizer_keeps_duplicate_host_aliases_for_candidate_joins(monkeypatch):
    import numpy as np

    import anvil.bridge.featurize as wire

    class SaVocab:
        @staticmethod
        def id(_sa):
            return 7

    fake_out = {
        "entity_row_of": {100: 0, 200: 0},
        "entities": np.zeros((1, 25), dtype=np.float32),
        "entity_names": ["identical permanent"],
        "globals": np.zeros(10, dtype=np.float32),
        "players": np.zeros((2, 6), dtype=np.float32),
        "history": [],
    }
    monkeypatch.setattr(wire, "assemble", lambda *args, **kwargs: fake_out)
    feat = wire.Featurizer.__new__(wire.Featurizer)
    feat.sa_vocab = SaVocab()
    feat.embed = type("Embed", (), {"row": staticmethod(lambda _name: 0)})()

    dec = {
        "p": 1,
        "obs": {"ents": []},
        "opts": [
            {"e": 100, "sa": "Brave the Elements (X=0)", "kind": "spell"},
            {"e": 200, "sa": "Brave the Elements (X=3)", "kind": "spell"},
        ],
    }
    _ex, aux = feat.example(dec, {"players": [{}, {}]}, "priority")

    assert aux["wire_to_candidate"] == [0, 1, 1]
    assert aux["candidate_key_aliases"][1] == [
        (100, "Brave the Elements"),
        (200, "Brave the Elements"),
    ]


def test_forced_cast_plan_preserves_requested_wire_option():
    torch = pytest.importorskip("torch")
    from collections import Counter

    from anvil.bridge.server import ModelBackend

    backend = object.__new__(ModelBackend)
    backend.counts = Counter()
    backend.n_sa = 1
    out = {
        "choice": torch.tensor([1]),
        "n_ent": 0,
        "stop_idx": 0,
        "tgt_picks": torch.empty((1, 0), dtype=torch.long),
        "x_cls": torch.tensor([0]),
    }
    cp = backend._castplan(
        out,
        {"cand_first_opt": [-1, 0], "row_min_id": {}, "seats": []},
        forced_option=2,
    )
    assert cp.spell_option == 2


def test_candidate_inverse_frequency_weights_have_mean_one():
    rows = [
        {"entity": 100, "sa": "Common"},
        {"entity": 100, "sa": "Common"},
        {"entity": 100, "sa": "Common"},
        {"entity": 200, "sa": "Rare"},
    ]
    weights = candidate_inverse_frequency_weights(rows)
    assert weights == pytest.approx([2 / 3, 2 / 3, 2 / 3, 2.0])
    assert sum(weights) / len(weights) == pytest.approx(1.0)


def test_candidate_wire_game_ids_use_drill_backend():
    from anvil.bridge.server import is_drill_game_id

    assert is_drill_game_id("g17.f0r0")
    assert is_drill_game_id("g17.w4.r0.n")
    assert not is_drill_game_id("g17")


def test_rollout_runs_complete_requires_full_games_and_candidate_labels(tmp_path):
    import scripts.run_brave_candidate_pipeline as pipeline

    plan = tmp_path / "plan"
    plan.mkdir()
    target = plan / "candidate-points.tsv"
    target.write_text(
        "gameIdx\twindowId\tordinal\tturn\tphase\tseat\tentityId\tnormalizedSA\n"
        "17\t4\t0\t7\tMAIN1\t1\t1234\tBrave the Elements\n"
    )
    (plan / "manifest.json").write_text(
        json.dumps(
            {
                "tag": "bravei000",
                "arms": [
                    {
                        "store": "source-store",
                        "candidate_file": str(target),
                    }
                ],
            }
        )
    )

    run = tmp_path / "run"
    worker = run / "workers" / "inv-0001"
    worker.mkdir(parents=True)
    (run / "run.json").write_text(
        json.dumps(
            {
                "purpose": "drillbravei000-source-store",
                "start_index": 17,
                "games": 1,
            }
        )
    )
    (worker / "games.jsonl").write_text(json.dumps({"i": 17}) + "\n")
    (worker / "labels.jsonl").write_text(
        json.dumps(
            {
                "ev": "candidate",
                "i": 17,
                "window": 4,
                "candidate": {"entity": 1234, "sa": "Brave the Elements"},
            }
        )
        + "\n"
    )

    assert pipeline.rollout_runs_complete([run], plan)

    (worker / "labels.jsonl").write_text("")
    assert not pipeline.rollout_runs_complete([run], plan)

    (worker / "labels.jsonl").write_text(
        json.dumps(
            {
                "ev": "candidate",
                "i": 17,
                "window": 4,
                "candidate": {"entity": 1234, "sa": "Brave the Elements"},
            }
        )
        + "\n"
    )
    (worker / "games.jsonl").write_text("")
    assert not pipeline.rollout_runs_complete([run], plan)


def test_policy_probability_mass_scores_brave_against_pass(monkeypatch):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    import anvil.bridge.server as server
    import scripts.run_brave_candidate_pipeline as pipeline

    class FakeBackend:
        def __init__(self, *_args, **_kwargs):
            self.pass_delta = 0.0
            self.feat = SimpleNamespace(
                example=lambda _dec, _header, _task: (
                    {},
                    {
                        "candidate_key_aliases": [
                            [],
                            [(1234, "Brave the Elements")],
                            [(5678, "Other spell")],
                        ]
                    },
                )
            )
            self.batcher = SimpleNamespace(
                submit=lambda *_args, **_kwargs: {
                    "choice_probs": torch.tensor([[0.2, 0.3, 0.5]])
                }
            )

    monkeypatch.setattr(server, "ModelBackend", FakeBackend)
    report = pipeline._policy_probability_mass(
        [({"players": [{}, {}]}, {"p": 0})],
        "Brave the Elements",
        "fake.pt",
        "cpu",
    )

    assert report == {
        "windows": 1,
        "unmapped_windows": 0,
        "brave": pytest.approx(0.3),
        "pass": pytest.approx(0.2),
    }


def test_legacy_campaign_validation_accepts_old_default_sampling():
    import scripts.run_brave_candidate_pipeline as pipeline

    expected = {"version": 3, "sa_regex": "Brave", "sample_mainline": True}
    campaign = {
        "version": 2,
        "config": {"version": 2, "sa_regex": "Brave"},
    }

    pipeline.validate_campaign(campaign, expected)

    with pytest.raises(SystemExit):
        pipeline.validate_campaign(
            campaign,
            {**expected, "sample_mainline": False},
        )
