#!/usr/bin/env python3
"""Summarize Quest RL sweep directories.

Example:

    .venv/bin/python scripts/quest/summarize_rl_sweep.py \
        --root data/training \
        --prefix constructed-four-rl-sweep-912345 \
        --config-file scripts/quest/rl_sweep_generation.tsv \
        --skip-first \
        --out data/training/constructed-four-rl-sweep-912345.tsv

The speed columns can skip iteration 0, while error columns always inspect
every recorded iteration so a startup bridge failure is not hidden by the
steady-state timing view.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _config_index(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    fields: list[str] | None = None
    out: dict[str, dict[str, str]] = {}
    for raw in path.read_text().splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            header = stripped[1:].strip()
            if header.startswith("id\t"):
                fields = header.split("\t")
            continue
        if fields is None:
            raise ValueError(f"{path}: expected a commented TSV header")
        values = raw.split("\t")
        if len(values) != len(fields):
            raise ValueError(
                f"{path}: row has {len(values)} fields; expected {len(fields)}"
            )
        out[values[0]] = dict(zip(fields, values))
    return out


def _status_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for status, count in (row.get("games", {}).get("statuses", {}) or {}).items():
            counts[status] = counts.get(status, 0) + int(count)
    return counts


def summarize_run(
    run_dir: Path,
    *,
    prefix: str,
    config_meta: dict[str, dict[str, str]],
    skip_first: bool,
) -> dict:
    config = {}
    config_path = run_dir / "loop_config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            config = {}

    monitor = _jsonl(run_dir / "monitor.jsonl")
    measured = monitor[1:] if skip_first and len(monitor) > 1 else monitor
    statuses = _status_counts(monitor)
    bridge_errors = sum(
        count
        for status, count in statuses.items()
        if "bridgepoisonedexception" in status.lower() or "bridge" in status.lower()
    )
    crash_or_error = sum(
        count
        for status, count in statuses.items()
        if any(token in status.lower() for token in ("crash_or_hang", "exception", "error"))
    )
    non_won = sum(count for status, count in statuses.items() if status != "won")
    fallbacks = sum(
        int(row.get("census", {}).get("fallback", 0) or 0) for row in monitor
    )
    flags = sum(len(row.get("flags", []) or []) for row in monitor)
    guards = sum(len(row.get("guard", []) or []) for row in monitor)

    gen_s = sum(float(row.get("gen_s", 0) or 0) for row in measured)
    train_s = sum(float(row.get("train_s", 0) or 0) for row in measured)
    games = sum(
        int((row.get("games", {}) or {}).get("games", 0) or 0) for row in measured
    )
    if not games:
        games = len(measured) * int(config.get("games", 0) or 0)
    win_rates = [
        float(row["rl"]["win_per_s"])
        for row in measured
        if row.get("rl", {}).get("win_per_s") is not None
    ]

    state = {}
    state_path = run_dir / "loop_state.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except json.JSONDecodeError:
            state = {}
    completed = int(state.get("iteration", 0) or 0)
    requested = int(config.get("iterations", 0) or 0)
    config_id = run_dir.name[len(prefix) + 1 :] if run_dir.name.startswith(prefix + "-") else run_dir.name
    meta = config_meta.get(config_id, {})

    return {
        "run": run_dir.name,
        "config_id": config_id,
        "stage": meta.get("stage", ""),
        "complete": int(bool(requested and completed >= requested)),
        "iterations": completed,
        "measured_iterations": len(measured),
        "workers": config.get("workers", ""),
        "chunk": config.get("chunk", ""),
        "initial_delay_ms": config.get("launch_delay_ms", ""),
        "replacement_delay_ms": config.get("replacement_launch_delay_ms", ""),
        "learner_workers": config.get("rl_workers", ""),
        "seg": config.get("rl_seg", ""),
        "replay": config.get("replay", ""),
        "servers": config.get("servers", ""),
        "max_batch": config.get("max_batch", ""),
        "batch_window_ms": config.get("batch_window_ms", ""),
        "games": games,
        "gen_s": round(gen_s, 3),
        "train_s": round(train_s, 3),
        "total_s": round(gen_s + train_s, 3),
        "generation_games_per_hour": round(games * 3600 / gen_s, 2) if gen_s else "",
        "combined_games_per_hour": round(games * 3600 / (gen_s + train_s), 2)
        if gen_s + train_s
        else "",
        "train_windows_per_second": round(statistics.mean(win_rates), 2)
        if win_rates
        else "",
        "bridge_errors": bridge_errors,
        "crash_or_error_statuses": crash_or_error,
        "non_won_games": non_won,
        "fallbacks": fallbacks,
        "flags": flags,
        "guards": guards,
        "clean": int(
            bool(monitor)
            and not bridge_errors
            and not crash_or_error
            and not non_won
            and not guards
        ),
        "statuses": json.dumps(statuses, sort_keys=True),
    }


FIELDS = [
    "run",
    "config_id",
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
    "games",
    "gen_s",
    "train_s",
    "total_s",
    "generation_games_per_hour",
    "combined_games_per_hour",
    "train_windows_per_second",
    "bridge_errors",
    "crash_or_error_statuses",
    "non_won_games",
    "fallbacks",
    "flags",
    "guards",
    "clean",
    "statuses",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/training"))
    parser.add_argument("--prefix", required=True, help="run-name prefix before -<config-id>")
    parser.add_argument("--config-file", type=Path, default=None)
    parser.add_argument(
        "--skip-first",
        action="store_true",
        help="exclude iteration 0 from speed columns; errors still include it",
    )
    parser.add_argument("--out", type=Path, default=None, help="write TSV output here")
    args = parser.parse_args()

    config_meta = _config_index(args.config_file)
    run_dirs = sorted(
        path for path in args.root.glob(f"{args.prefix}-*") if path.is_dir()
    )
    if not run_dirs:
        parser.error(f"no run directories found under {args.root} with prefix {args.prefix!r}")

    rows = [
        summarize_run(
            path,
            prefix=args.prefix,
            config_meta=config_meta,
            skip_first=args.skip_first,
        )
        for path in run_dirs
    ]
    rows.sort(key=lambda row: (-int(row["clean"]), float(row["total_s"] or 1e99)))

    destination = args.out.open("w", newline="") if args.out else None
    try:
        writer = csv.DictWriter(
            destination or sys.stdout,
            fieldnames=FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if destination:
            destination.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
