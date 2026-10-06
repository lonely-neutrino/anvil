import json

from scripts.quest.summarize_rl_sweep import summarize_run


def _write_run(tmp_path, *, statuses):
    run = tmp_path / "sweep-foo"
    run.mkdir()
    (run / "loop_config.json").write_text(
        json.dumps(
            {
                "iterations": 2,
                "games": 100,
                "workers": 24,
                "chunk": 21,
                "launch_delay_ms": 4000,
                "replacement_launch_delay_ms": 500,
                "rl_workers": 12,
                "rl_seg": 256,
                "replay": 1,
            }
        )
    )
    rows = [
        {
            "gen_s": 10,
            "train_s": 5,
            "games": {"games": 100, "statuses": statuses},
            "rl": {"win_per_s": 10},
            "flags": [],
            "guard": [],
            "census": {},
        },
        {
            "gen_s": 8,
            "train_s": 4,
            "games": {"games": 100, "statuses": {"won": 100}},
            "rl": {"win_per_s": 12},
            "flags": [],
            "guard": [],
            "census": {},
        },
    ]
    (run / "monitor.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    (run / "loop_state.json").write_text(json.dumps({"iteration": 2}))
    return run


def test_summary_skips_first_speed_row_but_keeps_startup_errors(tmp_path):
    run = _write_run(
        tmp_path,
        statuses={"won": 99, "crash_or_hang:BridgePoisonedException": 1},
    )

    row = summarize_run(
        run,
        prefix="sweep",
        config_meta={"foo": {"stage": "generation"}},
        skip_first=True,
    )

    assert row["stage"] == "generation"
    assert row["gen_s"] == 8.0
    assert row["train_s"] == 4.0
    assert row["bridge_errors"] == 1
    assert row["clean"] == 0
