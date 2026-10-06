#!/usr/bin/env python3
"""Summarize sequential, same-allocation Quest RL benchmark pairs.

Each row compares the configuration designated as the baseline with the other
configuration in a pair.  A pair is only marked clean when both runs finished,
had no bridge/crash/flag/guard events, and hardware capture saw the same node
and GPU.  Ordinary game draws are not treated as errors.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

from scripts.quest.summarize_rl_sweep import _config_index, _jsonl, summarize_run


def _read_pairs(path: Path) -> list[dict[str, str]]:
    fields: list[str] | None = None
    rows: list[dict[str, str]] = []
    for raw in path.read_text().splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            header = stripped[1:].strip()
            if header.startswith("pair_id\t"):
                fields = header.split("\t")
            continue
        if fields is None:
            raise ValueError(f"{path}: expected a commented TSV header")
        values = raw.split("\t")
        if len(values) != len(fields):
            raise ValueError(f"{path}: row has {len(values)} fields; expected {len(fields)}")
        rows.append(dict(zip(fields, values)))
    return rows


def _json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace("%", "")
    if text in {"", "N/A", "Not Supported", "-"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _gpu_utilization(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"samples": 0, "mean": "", "median": "", "max": ""}
    values: list[float] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = _float(row.get("utilization.gpu"))
            if value is not None:
                values.append(value)
    return {
        "samples": len(values),
        "mean": round(statistics.mean(values), 2) if values else "",
        "median": round(statistics.median(values), 2) if values else "",
        "max": round(max(values), 2) if values else "",
    }


def _cpu_busy(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"samples": 0, "mean": "", "median": "", "max": ""}
    header: list[str] | None = None
    values: list[float] = []
    for raw in path.read_text().splitlines():
        parts = raw.split()
        if "us" in parts and "id" in parts:
            header = parts
            continue
        if header is None or len(parts) < len(header):
            continue
        try:
            float(parts[0])
        except ValueError:
            continue
        idle = _float(parts[header.index("id")])
        if idle is not None:
            values.append(100.0 - idle)
    return {
        "samples": len(values),
        "mean": round(statistics.mean(values), 2) if values else "",
        "median": round(statistics.median(values), 2) if values else "",
        "max": round(max(values), 2) if values else "",
    }


def _train_metrics(run_dir: Path, skip_first: bool) -> dict[str, object]:
    files = sorted(run_dir.glob("iter-*/train/metrics.jsonl"))
    if skip_first and len(files) > 1:
        files = files[1:]
    rows = []
    for path in files:
        metrics = _jsonl(path)
        if metrics:
            rows.append(metrics[-1])
    win_rates = [_float(row.get("win_per_s")) for row in rows]
    win_rates = [value for value in win_rates if value is not None]
    phase_values: dict[str, list[float]] = {}
    for row in rows:
        for name, value in (row.get("phase", {}) or {}).items():
            number = _float(value)
            if number is not None:
                phase_values.setdefault(name, []).append(number)
    phase = {
        name: round(statistics.mean(values), 4)
        for name, values in phase_values.items()
        if values
    }
    last = rows[-1] if rows else {}
    return {
        "win_per_s": round(statistics.mean(win_rates), 2) if win_rates else "",
        "step": last.get("step", ""),
        "traj": last.get("traj", ""),
        "phase": phase,
    }


def _hardware_identity(path: Path) -> tuple[str, str, str]:
    data = _json(path)
    slurm = data.get("slurm", {}) or {}
    node = str(data.get("hostname") or slurm.get("SLURM_JOB_NODELIST") or "")
    gpus = data.get("gpus", []) or []
    if not gpus:
        return node, "", ""
    gpu = gpus[0]
    identity = str(gpu.get("uuid") or gpu.get("index") or "")
    name = str(gpu.get("name") or "")
    return node, identity, name


def _return_code(path: Path, status: dict[str, str]) -> str:
    code_path = path / "return-code"
    if code_path.exists():
        return code_path.read_text().strip()
    return status.get("return_code", "")


def _side(
    *,
    training_root: Path,
    prefix: str,
    config_id: str,
    slot: str,
    status: dict[str, str],
    config_meta: dict[str, dict[str, str]],
    skip_first: bool,
) -> dict[str, object]:
    run_name = status.get("rl_name", "")
    run_dir = training_root / run_name if run_name else Path()
    run_summary: dict = {}
    if run_name and run_dir.is_dir():
        run_prefix = (
            run_name[: -(len(config_id) + 1)]
            if run_name.endswith(f"-{config_id}")
            else prefix
        )
        run_summary = summarize_run(
            run_dir,
            prefix=run_prefix,
            config_meta=config_meta,
            skip_first=skip_first,
        )
    diagnostic_dir = training_root / "paired-sweeps" / prefix
    # The caller supplies status from a pair directory; locate that directory
    # by the slot/config suffix without making the run-name convention part of
    # the summary's public interface.
    pair_root = diagnostic_dir / status.get("pair_id", "")
    if not pair_root.is_dir():
        pair_root = diagnostic_dir
    run_diag = pair_root / f"{slot}-{config_id}"
    hardware = _hardware_identity(run_diag / "hardware-start.json")
    gpu = _gpu_utilization(run_diag / "gpu-utilization.csv")
    cpu = _cpu_busy(run_diag / "cpu-vmstat.log")
    train = _train_metrics(run_dir, skip_first) if run_name and run_dir.is_dir() else {}
    return {
        "config_id": config_id,
        "slot": slot,
        "run_name": run_name,
        "return_code": _return_code(run_diag, status),
        "summary": run_summary,
        "train": train,
        "node": hardware[0],
        "gpu": hardware[1],
        "gpu_name": hardware[2],
        "gpu_util": gpu,
        "cpu_busy": cpu,
        "stage": config_meta.get(config_id, {}).get("stage", ""),
    }


def _status_rows(pair_root: Path) -> dict[str, dict[str, str]]:
    path = pair_root / "status.tsv"
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        return {row.get("slot", ""): row for row in csv.DictReader(handle, delimiter="\t")}


def _number(side: dict[str, object], section: str, key: str) -> float | None:
    value = side.get(section, {})
    if not isinstance(value, dict):
        return None
    return _float(value.get(key))


def _summary_number(side: dict[str, object], key: str) -> float | None:
    value = side.get("summary", {})
    if not isinstance(value, dict):
        return None
    return _float(value.get(key))


def _ratio(candidate: float | None, baseline: float | None) -> float | str:
    if candidate is None or baseline in (None, 0):
        return ""
    return round(candidate / baseline, 4)


def _pct_delta(candidate: float | None, baseline: float | None) -> float | str:
    ratio = _ratio(candidate, baseline)
    return round((float(ratio) - 1.0) * 100.0, 2) if ratio != "" else ""


def _phase(side: dict[str, object], key: str) -> float | str:
    train = side.get("train", {})
    phase = train.get("phase", {}) if isinstance(train, dict) else {}
    value = phase.get(key) if isinstance(phase, dict) else None
    return value if value is not None else ""


def _clean(side: dict[str, object]) -> int:
    summary = side.get("summary", {})
    if not isinstance(summary, dict) or not summary.get("complete"):
        return 0
    try:
        return int(
            str(side.get("return_code", "1")) == "0"
            and int(summary.get("bridge_errors", 0) or 0) == 0
            and int(summary.get("crash_or_error_statuses", 0) or 0) == 0
            and int(summary.get("flags", 0) or 0) == 0
            and int(summary.get("guards", 0) or 0) == 0
        )
    except (TypeError, ValueError):
        return 0


def summarize_pair(
    pair_root: Path,
    pair: dict[str, str],
    *,
    training_root: Path,
    prefix: str,
    config_meta: dict[str, dict[str, str]],
    skip_first: bool,
) -> dict[str, object]:
    statuses = _status_rows(pair_root)
    config_a = pair.get("config_a", "")
    config_b = pair.get("config_b", "")
    a_status = next((row for row in statuses.values() if row.get("config_id") == config_a), {})
    b_status = next((row for row in statuses.values() if row.get("config_id") == config_b), {})
    if config_a == config_b:
        rows = list(statuses.values())
        a_status = rows[0] if rows else a_status
        b_status = rows[1] if len(rows) > 1 else b_status

    side_a = _side(
        training_root=training_root,
        prefix=prefix,
        config_id=config_a,
        slot=a_status.get("slot", "first"),
        status={**a_status, "pair_id": pair.get("pair_id", "")},
        config_meta=config_meta,
        skip_first=skip_first,
    )
    side_b = _side(
        training_root=training_root,
        prefix=prefix,
        config_id=config_b,
        slot=b_status.get("slot", "second"),
        status={**b_status, "pair_id": pair.get("pair_id", "")},
        config_meta=config_meta,
        skip_first=skip_first,
    )

    stage_a = config_meta.get(config_a, {}).get("stage", "")
    stage_b = config_meta.get(config_b, {}).get("stage", "")
    if stage_a == "baseline" and stage_b != "baseline":
        baseline, candidate = side_a, side_b
    elif stage_b == "baseline" and stage_a != "baseline":
        baseline, candidate = side_b, side_a
    else:
        baseline, candidate = side_a, side_b

    same_node = int(bool(side_a["node"] and side_a["node"] == side_b["node"]))
    same_gpu = int(bool(side_a["gpu"] and side_a["gpu"] == side_b["gpu"]))
    base_summary = baseline.get("summary", {})
    candidate_summary = candidate.get("summary", {})
    row = {
        "pair_id": pair.get("pair_id", pair_root.name),
        "config_a": config_a,
        "config_b": config_b,
        "order": pair.get("order", ""),
        "baseline_config": baseline["config_id"],
        "candidate_config": candidate["config_id"],
        "same_node": same_node,
        "same_gpu": same_gpu,
        "node_a": side_a["node"],
        "node_b": side_b["node"],
        "gpu_a": side_a["gpu"],
        "gpu_b": side_b["gpu"],
        "gpu_name_a": side_a["gpu_name"],
        "gpu_name_b": side_b["gpu_name"],
        "baseline_complete": base_summary.get("complete", 0),
        "candidate_complete": candidate_summary.get("complete", 0),
        "baseline_return_code": baseline.get("return_code", ""),
        "candidate_return_code": candidate.get("return_code", ""),
        "baseline_clean": _clean(baseline),
        "candidate_clean": _clean(candidate),
        "paired_clean": int(same_node and same_gpu and _clean(baseline) and _clean(candidate)),
        "baseline_gen_games_per_hour": _summary_number(baseline, "generation_games_per_hour") or "",
        "candidate_gen_games_per_hour": _summary_number(candidate, "generation_games_per_hour") or "",
        "candidate_gen_ratio": _ratio(
            _summary_number(candidate, "generation_games_per_hour"),
            _summary_number(baseline, "generation_games_per_hour"),
        ),
        "candidate_gen_delta_pct": _pct_delta(
            _summary_number(candidate, "generation_games_per_hour"),
            _summary_number(baseline, "generation_games_per_hour"),
        ),
        "baseline_train_s": _summary_number(baseline, "train_s") or "",
        "candidate_train_s": _summary_number(candidate, "train_s") or "",
        "candidate_train_ratio": _ratio(
            _summary_number(candidate, "train_s"), _summary_number(baseline, "train_s")
        ),
        "candidate_train_delta_pct": _pct_delta(
            _summary_number(candidate, "train_s"), _summary_number(baseline, "train_s")
        ),
        "baseline_combined_games_per_hour": _summary_number(baseline, "combined_games_per_hour") or "",
        "candidate_combined_games_per_hour": _summary_number(candidate, "combined_games_per_hour") or "",
        "candidate_combined_ratio": _ratio(
            _summary_number(candidate, "combined_games_per_hour"),
            _summary_number(baseline, "combined_games_per_hour"),
        ),
        "candidate_combined_delta_pct": _pct_delta(
            _summary_number(candidate, "combined_games_per_hour"),
            _summary_number(baseline, "combined_games_per_hour"),
        ),
        "baseline_train_win_per_s": _number(baseline, "train", "win_per_s") or "",
        "candidate_train_win_per_s": _number(candidate, "train", "win_per_s") or "",
        "candidate_train_win_ratio": _ratio(
            _number(candidate, "train", "win_per_s"), _number(baseline, "train", "win_per_s")
        ),
        "baseline_train_load": _phase(baseline, "load"),
        "candidate_train_load": _phase(candidate, "load"),
        "baseline_train_fwd_bwd": _phase(baseline, "fwd_bwd"),
        "candidate_train_fwd_bwd": _phase(candidate, "fwd_bwd"),
        "baseline_train_fwd_nograd": _phase(baseline, "fwd_nograd"),
        "candidate_train_fwd_nograd": _phase(candidate, "fwd_nograd"),
        "baseline_bridge_errors": _summary_number(baseline, "bridge_errors") or 0,
        "candidate_bridge_errors": _summary_number(candidate, "bridge_errors") or 0,
        "baseline_crash_or_error_statuses": _summary_number(baseline, "crash_or_error_statuses") or 0,
        "candidate_crash_or_error_statuses": _summary_number(candidate, "crash_or_error_statuses") or 0,
        "baseline_flags": _summary_number(baseline, "flags") or 0,
        "candidate_flags": _summary_number(candidate, "flags") or 0,
        "baseline_guards": _summary_number(baseline, "guards") or 0,
        "candidate_guards": _summary_number(candidate, "guards") or 0,
    }
    for label, side in (("a", side_a), ("b", side_b)):
        gpu = side["gpu_util"]
        cpu = side["cpu_busy"]
        row[f"gpu_samples_{label}"] = gpu.get("samples", "")
        row[f"gpu_mean_{label}"] = gpu.get("mean", "")
        row[f"gpu_median_{label}"] = gpu.get("median", "")
        row[f"gpu_max_{label}"] = gpu.get("max", "")
        row[f"cpu_busy_samples_{label}"] = cpu.get("samples", "")
        row[f"cpu_busy_mean_{label}"] = cpu.get("mean", "")
        row[f"cpu_busy_median_{label}"] = cpu.get("median", "")
        row[f"cpu_busy_max_{label}"] = cpu.get("max", "")
    row["baseline_statuses"] = base_summary.get("statuses", "")
    row["candidate_statuses"] = candidate_summary.get("statuses", "")
    return row


FIELDS = [
    "pair_id",
    "config_a",
    "config_b",
    "order",
    "baseline_config",
    "candidate_config",
    "same_node",
    "same_gpu",
    "node_a",
    "node_b",
    "gpu_a",
    "gpu_b",
    "gpu_name_a",
    "gpu_name_b",
    "baseline_complete",
    "candidate_complete",
    "baseline_return_code",
    "candidate_return_code",
    "baseline_clean",
    "candidate_clean",
    "paired_clean",
    "baseline_gen_games_per_hour",
    "candidate_gen_games_per_hour",
    "candidate_gen_ratio",
    "candidate_gen_delta_pct",
    "baseline_train_s",
    "candidate_train_s",
    "candidate_train_ratio",
    "candidate_train_delta_pct",
    "baseline_combined_games_per_hour",
    "candidate_combined_games_per_hour",
    "candidate_combined_ratio",
    "candidate_combined_delta_pct",
    "baseline_train_win_per_s",
    "candidate_train_win_per_s",
    "candidate_train_win_ratio",
    "baseline_train_load",
    "candidate_train_load",
    "baseline_train_fwd_bwd",
    "candidate_train_fwd_bwd",
    "baseline_train_fwd_nograd",
    "candidate_train_fwd_nograd",
    "baseline_bridge_errors",
    "candidate_bridge_errors",
    "baseline_crash_or_error_statuses",
    "candidate_crash_or_error_statuses",
    "baseline_flags",
    "candidate_flags",
    "baseline_guards",
    "candidate_guards",
    "gpu_samples_a",
    "gpu_mean_a",
    "gpu_median_a",
    "gpu_max_a",
    "gpu_samples_b",
    "gpu_mean_b",
    "gpu_median_b",
    "gpu_max_b",
    "cpu_busy_samples_a",
    "cpu_busy_mean_a",
    "cpu_busy_median_a",
    "cpu_busy_max_a",
    "cpu_busy_samples_b",
    "cpu_busy_mean_b",
    "cpu_busy_median_b",
    "cpu_busy_max_b",
    "baseline_statuses",
    "candidate_statuses",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/training"))
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--pairs-file", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--skip-first", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    training_root = root if root.name == "training" else root / "data" / "training"
    config_meta = _config_index(args.config_file)
    pairs = _read_pairs(args.pairs_file)
    sweep_root = training_root / "paired-sweeps" / args.prefix
    rows = [
        summarize_pair(
            sweep_root / pair["pair_id"],
            pair,
            training_root=training_root,
            prefix=args.prefix,
            config_meta=config_meta,
            skip_first=args.skip_first,
        )
        for pair in pairs
    ]

    destination = args.out.open("w", newline="") if args.out else None
    try:
        writer = csv.DictWriter(
            destination or sys.stdout,
            fieldnames=FIELDS,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if destination:
            destination.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
