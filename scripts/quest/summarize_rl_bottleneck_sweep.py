#!/usr/bin/env python3
"""Summarize the moderate-worker RL bottleneck sweep.

The bottleneck sweep deliberately uses one startup-inclusive iteration per
configuration. This summarizer therefore never drops iteration zero. It joins
the normal RL monitor output with the per-task hardware, CPU, GPU, and server
telemetry captured by ``rl_sweep.sbatch``.

Example::

    .venv/bin/python scripts/quest/summarize_rl_bottleneck_sweep.py \
        --root data/training \
        --prefix constructed-four-rl-bottleneck-20261006-120000 \
        --config-file scripts/quest/rl_bottleneck_phase1.tsv \
        --config-file scripts/quest/rl_bottleneck_phase2.tsv \
        --out data/training/constructed-four-rl-bottleneck-summary.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

try:
    from scripts.quest.summarize_rl_sweep import _config_index, _jsonl, summarize_run
except ModuleNotFoundError:  # direct ``python scripts/quest/...`` invocation
    from summarize_rl_sweep import _config_index, _jsonl, summarize_run


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


def _int(value: object) -> int | None:
    number = _float(value)
    return int(number) if number is not None else None


def _json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _key_values(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        values[key] = value
    return values


def _memory_gb(value: object) -> float | str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    match = re.fullmatch(r"([0-9.]+)\s*([KMGTP]?)", text, re.IGNORECASE)
    if not match:
        return ""
    number = float(match.group(1))
    suffix = match.group(2).upper()
    scale = {"": 1 / 1024, "K": 1 / 1024, "M": 1 / 1024, "G": 1, "T": 1024, "P": 1024 * 1024}
    # Slurm's *_MEM_PER_NODE is normally an integer number of MiB. A suffix
    # makes the input self-describing and is interpreted directly.
    if suffix == "":
        return round(number / 1024, 3)
    if suffix == "K":
        return round(number / (1024 * 1024), 3)
    if suffix == "M":
        return round(number / 1024, 3)
    return round(number * scale[suffix], 3)


def _summary(values: list[float]) -> tuple[int, float | str, float | str, float | str]:
    if not values:
        return 0, "", "", ""
    return (
        len(values),
        round(statistics.mean(values), 2),
        round(statistics.median(values), 2),
        round(max(values), 2),
    )


def _hardware(diag_root: Path) -> dict[str, object]:
    data = _json(diag_root / "hardware-start.json")
    slurm = data.get("slurm", {}) or {}
    gpus = data.get("gpus", []) or []
    gpu_tokens: list[str] = []
    for key in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        value = str(slurm.get(key) or "")
        gpu_tokens.extend(token.strip() for token in value.split(",") if token.strip())
    gpu = next(
        (
            candidate
            for candidate in gpus
            if isinstance(candidate, dict)
            and any(
                token in {str(candidate.get("index") or ""), str(candidate.get("uuid") or "")}
                for token in gpu_tokens
            )
        ),
        None,
    )
    # A one-GPU cgroup is unambiguous even when Quest does not export a GPU
    # selector. With several visible GPUs, leave the identity blank rather
    # than silently attributing the run to GPU 0.
    if gpu is None and len(gpus) == 1:
        gpu = gpus[0]
    if not isinstance(gpu, dict):
        gpu = {}
    return {
        "node": str(data.get("hostname") or slurm.get("SLURM_JOB_NODELIST") or ""),
        "node_list": str(slurm.get("SLURM_JOB_NODELIST") or ""),
        "gpu_index": str(gpu.get("index") or ""),
        "gpu_name": str(gpu.get("name") or ""),
        "gpu_uuid": str(gpu.get("uuid") or ""),
        "gpu_total_memory_gb": _memory_gb(gpu.get("memory.total")),
        "job_id": str(slurm.get("SLURM_JOB_ID") or ""),
        "array_job_id": str(slurm.get("SLURM_ARRAY_JOB_ID") or ""),
        "array_task_id": str(slurm.get("SLURM_ARRAY_TASK_ID") or ""),
        "requested_cpus": _int(slurm.get("SLURM_CPUS_PER_TASK")),
        "requested_memory_gb": _memory_gb(slurm.get("SLURM_MEM_PER_NODE")),
        "cuda_visible_devices": str(slurm.get("CUDA_VISIBLE_DEVICES") or ""),
    }


def _gpu_utilization(path: Path, target_index: str) -> dict[str, object]:
    if not path.exists():
        return {
            "samples": 0,
            "util_mean": "",
            "util_median": "",
            "util_max": "",
            "memory_used_mean": "",
            "memory_used_max": "",
            "memory_total": "",
        }
    util: list[float] = []
    memory_used: list[float] = []
    memory_total: list[float] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if target_index:
        selected = [row for row in rows if str(row.get("index", "")).strip() == target_index]
        if selected:
            rows = selected
    for row in rows:
        value = _float(row.get("utilization.gpu"))
        if value is not None:
            util.append(value)
        value = _float(row.get("memory.used"))
        if value is not None:
            memory_used.append(value)
        value = _float(row.get("memory.total"))
        if value is not None:
            memory_total.append(value)
    samples, mean, median, maximum = _summary(util)
    return {
        "samples": samples,
        "util_mean": mean,
        "util_median": median,
        "util_max": maximum,
        "memory_used_mean": round(statistics.mean(memory_used), 2) if memory_used else "",
        "memory_used_max": round(max(memory_used), 2) if memory_used else "",
        "memory_total": round(statistics.median(memory_total), 2) if memory_total else "",
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
        idle_index = header.index("id")
        idle = _float(parts[idle_index])
        if idle is not None:
            values.append(100.0 - idle)
    samples, mean, median, maximum = _summary(values)
    return {"samples": samples, "mean": mean, "median": median, "max": maximum}


_SERVER_STATS = re.compile(
    r"stats:\s+(?P<asks>[0-9.]+) asks in [0-9.]+s "
    r"\((?P<rps>[0-9.]+) rps\), mean batch (?P<batch>[0-9.]+), "
    r"wait p50 (?P<p50>[0-9.]+) / p90 (?P<p90>[0-9.]+) / p99 (?P<p99>[0-9.]+) ms, "
    r"forward (?P<forward_ms>[0-9.]+) ms/batch \((?P<busy>[0-9.]+)% busy\), "
    r"queue max (?P<queue>[0-9.]+)"
)


def _server_stats(run_dir: Path) -> dict[str, object]:
    values: dict[str, list[float]] = {
        "rps": [],
        "batch": [],
        "p50": [],
        "p90": [],
        "p99": [],
        "forward_ms": [],
        "busy": [],
        "queue": [],
    }
    logs = sorted(run_dir.glob("iter-*/server.log")) + sorted(run_dir.glob("iter-*/arms-server.log"))
    for path in logs:
        for match in _SERVER_STATS.finditer(path.read_text(errors="replace")):
            for key, field in (
                ("rps", "rps"),
                ("batch", "batch"),
                ("p50", "p50"),
                ("p90", "p90"),
                ("p99", "p99"),
                ("forward_ms", "forward_ms"),
                ("busy", "busy"),
                ("queue", "queue"),
            ):
                values[key].append(float(match.group(field)))
    return {
        "samples": len(values["batch"]),
        "rps_mean": round(statistics.mean(values["rps"]), 2) if values["rps"] else "",
        "mean_batch": round(statistics.mean(values["batch"]), 2) if values["batch"] else "",
        "wait_p50_ms": round(statistics.mean(values["p50"]), 2) if values["p50"] else "",
        "wait_p90_ms": round(statistics.mean(values["p90"]), 2) if values["p90"] else "",
        "wait_p99_ms": round(statistics.mean(values["p99"]), 2) if values["p99"] else "",
        "forward_ms": round(statistics.mean(values["forward_ms"]), 2)
        if values["forward_ms"]
        else "",
        "forward_busy_pct": round(statistics.mean(values["busy"]), 2) if values["busy"] else "",
        "queue_max": round(max(values["queue"]), 2) if values["queue"] else "",
    }


def _timing(diag_root: Path) -> dict[str, object]:
    values = _key_values(diag_root / "timing.txt")
    if not values:
        values = _key_values(diag_root / "finish.txt")
    return {"wall_elapsed_s": _float(values.get("elapsed_s")) or ""}


def _return_code(diag_root: Path) -> str:
    path = diag_root / "return-code"
    return path.read_text().strip() if path.exists() else ""


def _status_counts(monitor: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in monitor:
        for status, count in (row.get("games", {}).get("statuses", {}) or {}).items():
            counts[status] = counts.get(status, 0) + int(count)
    return counts


def _log_error_counts(diag_root: Path) -> tuple[int, int]:
    error_text = "\n".join(
        path.read_text(errors="replace")
        for path in (diag_root / "run.err", diag_root / "run.out")
        if path.exists()
    ).lower()
    bridge = error_text.count("bridgepoisonedexception")
    crash = int("traceback (most recent call last)" in error_text or "error:" in error_text)
    return bridge, crash


def _config_metadata(paths: list[Path]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for path in paths:
        phase = path.stem.rsplit("_", 1)[-1]
        for config_id, values in _config_index(path).items():
            if config_id in result and result[config_id] != values:
                raise ValueError(f"configuration {config_id!r} differs between tables")
            result[config_id] = {**values, "phase": phase}
    return result


def _run_row(
    *,
    training_root: Path,
    diagnostic_root: Path,
    prefix: str,
    config_id: str,
    config_meta: dict[str, dict[str, str]],
) -> dict[str, object]:
    run_dir = training_root / f"{prefix}-{config_id}"
    diag_root = diagnostic_root / config_id
    base = summarize_run(
        run_dir,
        prefix=prefix,
        config_meta=config_meta,
        skip_first=False,
    ) if run_dir.is_dir() else {
        "run": run_dir.name,
        "config_id": config_id,
        "stage": config_meta.get(config_id, {}).get("stage", ""),
        "complete": 0,
        "iterations": 0,
        "measured_iterations": 0,
        "workers": config_meta.get(config_id, {}).get("workers", ""),
        "chunk": config_meta.get(config_id, {}).get("chunk", ""),
        "initial_delay_ms": config_meta.get(config_id, {}).get("initial_delay_ms", ""),
        "replacement_delay_ms": config_meta.get(config_id, {}).get("replacement_delay_ms", ""),
        "learner_workers": config_meta.get(config_id, {}).get("learner_workers", ""),
        "seg": config_meta.get(config_id, {}).get("seg", ""),
        "replay": config_meta.get(config_id, {}).get("replay", ""),
        "servers": config_meta.get(config_id, {}).get("servers", ""),
        "max_batch": config_meta.get(config_id, {}).get("max_batch", ""),
        "batch_window_ms": config_meta.get(config_id, {}).get("batch_window_ms", ""),
        "games": 0,
        "gen_s": "",
        "train_s": "",
        "total_s": "",
        "generation_games_per_hour": "",
        "combined_games_per_hour": "",
        "train_windows_per_second": "",
        "bridge_errors": 0,
        "crash_or_error_statuses": 0,
        "fallbacks": 0,
        "flags": 0,
        "guards": 0,
        "statuses": "{}",
    }

    base["phase"] = config_meta.get(config_id, {}).get("phase", "")
    base["arms_every"] = config_meta.get(config_id, {}).get("arms_every", "")
    base["iterations_requested"] = config_meta.get(config_id, {}).get("iterations", "")

    monitor = _jsonl(run_dir / "monitor.jsonl")
    statuses = _status_counts(monitor)
    draws = statuses.get("draw", 0)
    failed_games = sum(count for status, count in statuses.items() if status not in {"won", "draw"})
    flags = sum(len(row.get("flags", []) or []) for row in monitor)
    guards = sum(len(row.get("guard", []) or []) for row in monitor)
    fallbacks = sum(int(row.get("census", {}).get("fallback", 0) or 0) for row in monitor)
    log_bridge, log_crash = _log_error_counts(diag_root)
    hardware = _hardware(diag_root)
    gpu = _gpu_utilization(diag_root / "gpu-utilization.csv", str(hardware["gpu_index"]))
    cpu = _cpu_busy(diag_root / "cpu-vmstat.log")
    server = _server_stats(run_dir)
    timing = _timing(diag_root)
    resources = _key_values(diag_root / "resource.txt")
    expected_cpus = _int(resources.get("expected_cpus"))
    actual_cpus = _int(resources.get("allocated_cpus")) or hardware["requested_cpus"]
    return_code = _return_code(diag_root)
    if return_code == "":
        return_code = "0" if base.get("complete") else ""

    reasons: list[str] = []
    if not base.get("complete"):
        reasons.append("incomplete")
    if return_code not in {"", "0"}:
        reasons.append(f"return_code={return_code}")
    if int(base.get("bridge_errors", 0) or 0) or log_bridge:
        reasons.append("bridge_error")
    if int(base.get("crash_or_error_statuses", 0) or 0) or log_crash:
        reasons.append("crash_or_error")
    if failed_games:
        reasons.append("failed_games")
    if flags:
        reasons.append("flags")
    if guards:
        reasons.append("guards")
    if fallbacks:
        reasons.append("fallbacks")
    if expected_cpus is not None and actual_cpus is not None and actual_cpus < expected_cpus:
        reasons.append("cpu_request_mismatch")

    return {
        **base,
        "phase": base["phase"],
        "draw_games": draws,
        "failed_games": failed_games,
        "log_bridge_errors": log_bridge,
        "log_crash_or_error": log_crash,
        "flags": flags,
        "guards": guards,
        "fallbacks": fallbacks,
        "return_code": return_code,
        "expected_cpus": expected_cpus if expected_cpus is not None else "",
        "requested_cpus": actual_cpus if actual_cpus is not None else "",
        "requested_memory_gb": hardware["requested_memory_gb"],
        "node": hardware["node"],
        "node_list": hardware["node_list"],
        "gpu_index": hardware["gpu_index"],
        "gpu_name": hardware["gpu_name"],
        "gpu_uuid": hardware["gpu_uuid"],
        "cuda_visible_devices": hardware["cuda_visible_devices"],
        "gpu_inventory_memory_gb": hardware["gpu_total_memory_gb"],
        "gpu_samples": gpu["samples"],
        "gpu_util_mean_pct": gpu["util_mean"],
        "gpu_util_median_pct": gpu["util_median"],
        "gpu_util_max_pct": gpu["util_max"],
        "gpu_memory_used_mean_mib": gpu["memory_used_mean"],
        "gpu_memory_used_max_mib": gpu["memory_used_max"],
        "gpu_memory_total_mib": gpu["memory_total"],
        "cpu_samples": cpu["samples"],
        "cpu_busy_mean_pct": cpu["mean"],
        "cpu_busy_median_pct": cpu["median"],
        "cpu_busy_max_pct": cpu["max"],
        "server_stats_samples": server["samples"],
        "server_rps_mean": server["rps_mean"],
        "server_mean_batch": server["mean_batch"],
        "server_wait_p50_ms": server["wait_p50_ms"],
        "server_wait_p90_ms": server["wait_p90_ms"],
        "server_wait_p99_ms": server["wait_p99_ms"],
        "server_forward_ms": server["forward_ms"],
        "server_forward_busy_pct": server["forward_busy_pct"],
        "server_queue_max": server["queue_max"],
        "wall_elapsed_s": timing["wall_elapsed_s"],
        "clean": int(not reasons),
        "contaminated": int(bool(reasons)),
        "contamination_reasons": ";".join(reasons),
        "statuses": json.dumps(statuses, sort_keys=True),
    }


CONTROL_KEYS = (
    "stage",
    "workers",
    "chunk",
    "initial_delay_ms",
    "replacement_delay_ms",
    "learner_workers",
    "seg",
    "replay",
    "servers",
    "max_batch",
    "batch_window_ms",
    "iterations",
    "games",
    "arms_every",
)


FIELDS = [
    "run",
    "config_id",
    "phase",
    "stage",
    "complete",
    "iterations",
    "measured_iterations",
    "workers",
    "chunk",
    "initial_delay_ms",
    "replacement_delay_ms",
    "learner_workers",
    "seg",
    "replay",
    "servers",
    "max_batch",
    "batch_window_ms",
    "iterations_requested",
    "games",
    "gen_s",
    "train_s",
    "total_s",
    "generation_games_per_hour",
    "combined_games_per_hour",
    "train_windows_per_second",
    "draw_games",
    "failed_games",
    "bridge_errors",
    "crash_or_error_statuses",
    "log_bridge_errors",
    "log_crash_or_error",
    "fallbacks",
    "flags",
    "guards",
    "return_code",
    "expected_cpus",
    "requested_cpus",
    "requested_memory_gb",
    "node",
    "node_list",
    "gpu_index",
    "gpu_name",
    "gpu_uuid",
    "cuda_visible_devices",
    "gpu_inventory_memory_gb",
    "gpu_samples",
    "gpu_util_mean_pct",
    "gpu_util_median_pct",
    "gpu_util_max_pct",
    "gpu_memory_used_mean_mib",
    "gpu_memory_used_max_mib",
    "gpu_memory_total_mib",
    "cpu_samples",
    "cpu_busy_mean_pct",
    "cpu_busy_median_pct",
    "cpu_busy_max_pct",
    "server_stats_samples",
    "server_rps_mean",
    "server_mean_batch",
    "server_wait_p50_ms",
    "server_wait_p90_ms",
    "server_wait_p99_ms",
    "server_forward_ms",
    "server_forward_busy_pct",
    "server_queue_max",
    "wall_elapsed_s",
    "control_group",
    "control_n",
    "control_gen_delta_pct",
    "control_drift_flag",
    "clean",
    "contaminated",
    "contamination_reasons",
    "statuses",
]


def _control_key(row: dict[str, object]) -> tuple[str, ...]:
    return tuple(str(row.get(key, "")) for key in CONTROL_KEYS)


def _control_label(row: dict[str, object]) -> str:
    return (
        f"w{row.get('workers', '')}-mb{row.get('max_batch', '')}-"
        f"t{row.get('batch_window_ms', '')}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/training"))
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--config-file", type=Path, action="append", required=True)
    parser.add_argument("--control-threshold-pct", type=float, default=15.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config_meta = _config_metadata(args.config_file)
    rows = [
        _run_row(
            training_root=args.root,
            diagnostic_root=args.root / "sweeps" / args.prefix,
            prefix=args.prefix,
            config_id=config_id,
            config_meta=config_meta,
        )
        for config_id in config_meta
    ]

    control_groups: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for row in rows:
        control_groups.setdefault(_control_key(row), []).append(row)
    for row in rows:
        group = control_groups[_control_key(row)]
        valid_times = [float(other["gen_s"]) for other in group if other.get("gen_s") not in {"", None}]
        delta = ""
        drift = 0
        if len(valid_times) >= 2 and min(valid_times) > 0:
            delta = round((max(valid_times) - min(valid_times)) / min(valid_times) * 100.0, 2)
            drift = int(delta > args.control_threshold_pct)
        row["control_group"] = _control_label(row) if len(group) >= 2 else ""
        row["control_n"] = len(group)
        row["control_gen_delta_pct"] = delta
        row["control_drift_flag"] = drift

    rows.sort(key=lambda row: (-int(row.get("clean", 0)), float(row.get("gen_s") or 1e99)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in FIELDS} for row in rows)
    print(f"wrote {args.out} ({len(rows)} configurations)")
    print(f"clean={sum(int(row.get('clean', 0)) for row in rows)} contaminated={sum(int(row.get('contaminated', 0)) for row in rows)}")
    for row in rows:
        print(
            f"{row['config_id']} clean={row['clean']} workers={row['workers']} "
            f"gen_s={row['gen_s']} gpu_mean={row['gpu_util_mean_pct']} "
            f"cpu_busy={row['cpu_busy_mean_pct']} control_drift={row['control_drift_flag']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
