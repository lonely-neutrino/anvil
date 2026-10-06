import json

from scripts.quest.summarize_rl_paired_sweep import summarize_pair


def _write_run(training, name, *, win_per_s, statuses=None, steady_gen_s=8):
    run = training / name
    (run / "iter-000" / "train").mkdir(parents=True)
    (run / "iter-001" / "train").mkdir(parents=True)
    (run / "loop_config.json").write_text(
        json.dumps(
            {
                "iterations": 2,
                "games": 100,
                "workers": 24,
                "chunk": 15,
                "launch_delay_ms": 4000,
                "replacement_launch_delay_ms": 0,
                "rl_workers": 8,
                "rl_seg": 256,
                "replay": 1,
            }
        )
    )
    status = statuses or {"won": 100}
    monitor = [
        {
            "gen_s": 10,
            "train_s": 5,
            "games": {"games": 100, "statuses": status},
            "rl": {"win_per_s": 10},
            "flags": [],
            "guard": [],
            "census": {},
        },
        {
            "gen_s": steady_gen_s,
            "train_s": 4,
            "games": {"games": 100, "statuses": {"won": 100}},
            "rl": {"win_per_s": 12},
            "flags": [],
            "guard": [],
            "census": {},
        },
    ]
    (run / "monitor.jsonl").write_text("\n".join(json.dumps(row) for row in monitor) + "\n")
    (run / "loop_state.json").write_text(json.dumps({"iteration": 2}))
    for index, step in enumerate((200500, 201000)):
        (run / f"iter-{index:03d}" / "train" / "metrics.jsonl").write_text(
            json.dumps(
                {
                    "step": step,
                    "traj": 2000,
                    "win_per_s": win_per_s,
                    "phase": {"load": 0.25, "fwd_bwd": 0.5, "fwd_nograd": 0.2},
                }
            )
            + "\n"
        )


def _write_diagnostics(pair_root, slot, config, run_name, *, gpu_uuid="GPU-1"):
    diag = pair_root / f"{slot}-{config}"
    diag.mkdir(parents=True)
    (diag / "return-code").write_text("0\n")
    (diag / "hardware-start.json").write_text(
        json.dumps(
            {
                "hostname": "qgpu0001",
                "gpus": [{"uuid": gpu_uuid, "index": "0", "name": "A100"}],
            }
        )
    )
    (diag / "gpu-utilization.csv").write_text(
        "timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,pstate\n"
        "now,0,80,20,100,40000,200,60,P0\n"
    )
    (diag / "cpu-vmstat.log").write_text(
        "procs -----------memory---------- ---swap-- -----io---- -system-- ------cpu-----\n"
        " r  b   swpd   free   buff  cache   si   so    bi    bo   in   cs us sy id wa st\n"
        " 1  0      0      1      1      1    0    0     0     0    0    0 50 10 40 0 0\n"
    )
    return diag


def test_paired_summary_reports_same_hardware_and_candidate_speed(tmp_path):
    training = tmp_path / "training"
    training.mkdir()
    prefix = "constructed-four-rl-paired-1"
    pair = {
        "pair_id": "candidate-vs-baseline",
        "config_a": "baseline",
        "config_b": "candidate",
        "order": "a-first",
    }
    pair_root = training / "paired-sweeps" / prefix / pair["pair_id"]
    pair_root.mkdir(parents=True)
    base_name = f"{prefix}-{pair['pair_id']}-first-baseline"
    candidate_name = f"{prefix}-{pair['pair_id']}-second-candidate"
    _write_run(training, base_name, win_per_s=10)
    _write_run(training, candidate_name, win_per_s=20, steady_gen_s=6.4)
    _write_diagnostics(pair_root, "first", "baseline", base_name)
    _write_diagnostics(pair_root, "second", "candidate", candidate_name)
    (pair_root / "status.tsv").write_text(
        "slot\tconfig_id\trl_name\treturn_code\tstart_epoch\tend_epoch\n"
        f"first\tbaseline\t{base_name}\t0\t1\t2\n"
        f"second\tcandidate\t{candidate_name}\t0\t3\t4\n"
    )

    row = summarize_pair(
        pair_root,
        pair,
        training_root=training,
        prefix=prefix,
        config_meta={"baseline": {"stage": "baseline"}, "candidate": {"stage": "generation"}},
        skip_first=True,
    )

    assert row["same_node"] == 1
    assert row["same_gpu"] == 1
    assert row["candidate_gen_ratio"] == 1.25
    assert row["candidate_gen_delta_pct"] == 25.0
    assert row["candidate_train_win_ratio"] == 2.0
    assert row["gpu_mean_a"] == 80.0
    assert row["cpu_busy_mean_a"] == 60.0
    assert row["paired_clean"] == 1


def test_paired_summary_rejects_bridge_errors(tmp_path):
    training = tmp_path / "training"
    training.mkdir()
    prefix = "paired-2"
    pair = {"pair_id": "x", "config_a": "baseline", "config_b": "candidate", "order": "a-first"}
    pair_root = training / "paired-sweeps" / prefix / "x"
    pair_root.mkdir(parents=True)
    base_name = f"{prefix}-x-first-baseline"
    candidate_name = f"{prefix}-x-second-candidate"
    _write_run(training, base_name, win_per_s=10)
    _write_run(
        training,
        candidate_name,
        win_per_s=10,
        statuses={"won": 99, "crash_or_hang:BridgePoisonedException": 1},
    )
    _write_diagnostics(pair_root, "first", "baseline", base_name)
    _write_diagnostics(pair_root, "second", "candidate", candidate_name)
    (pair_root / "status.tsv").write_text(
        "slot\tconfig_id\trl_name\treturn_code\tstart_epoch\tend_epoch\n"
        f"first\tbaseline\t{base_name}\t0\t1\t2\n"
        f"second\tcandidate\t{candidate_name}\t0\t3\t4\n"
    )

    row = summarize_pair(
        pair_root,
        pair,
        training_root=training,
        prefix=prefix,
        config_meta={"baseline": {"stage": "baseline"}, "candidate": {"stage": "generation"}},
        skip_first=False,
    )

    assert row["candidate_bridge_errors"] == 1
    assert row["candidate_clean"] == 0
    assert row["paired_clean"] == 0
