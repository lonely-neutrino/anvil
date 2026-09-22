import json

from anvil.evals.model_match import aggregate, summarize_run, winner_seat


def test_winner_seat_uses_registered_one_based_name():
    assert winner_seat("Anvil(1)-Green Stompy") == 0
    assert winner_seat("Anvil(2)-Green Stompy") == 1
    assert winner_seat("draw") is None


def test_summarize_run_separates_nondecisive_games(tmp_path):
    (tmp_path / "games.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"i": 0, "status": "won", "winner": "Anvil(1)-A"},
                {"i": 1, "status": "won", "winner": "Anvil(2)-B"},
                {"i": 2, "status": "crash_or_hang:BridgePoisonedException", "winner": ""},
            )
        )
        + "\n"
    )

    row = summarize_run(tmp_path, model_a_seat=0)

    assert row["games"] == 3
    assert row["decisive"] == 2
    assert row["nondecisive"] == 1
    assert row["a_wins"] == 1
    assert row["b_wins"] == 1
    assert row["crashes"] == 1


def test_aggregate_reports_decisive_and_all_game_rates():
    report = aggregate(
        "a.pt",
        "b.pt",
        [
            {
                "run": "a0-b1",
                "model_a_seat": 0,
                "games": 4,
                "decisive": 4,
                "nondecisive": 0,
                "a_wins": 3,
                "b_wins": 1,
                "unattributed_decisive": 0,
                "crashes": 0,
                "draw_clock": 0,
            },
            {
                "run": "b0-a1",
                "model_a_seat": 1,
                "games": 4,
                "decisive": 3,
                "nondecisive": 1,
                "a_wins": 1,
                "b_wins": 2,
                "unattributed_decisive": 0,
                "crashes": 1,
                "draw_clock": 1,
            },
        ],
        seed_base=7,
        games_per_orientation=4,
    )

    assert report["a_wins"] == 4
    assert report["b_wins"] == 3
    assert report["games"] == 8
    assert report["decisive"] == 7
    assert report["a_winrate_all"] == 0.5
    assert report["a_winrate_decisive"] == 4 / 7
