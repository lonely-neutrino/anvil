"""Utilities for reporting a two-checkpoint model-vs-model match.

The bridge names registered players ``Anvil(1)`` and ``Anvil(2)`` in the
game-result JSON.  This module keeps the result accounting independent of
``arms_report.py``, whose model-win definition intentionally assumes that
only one seat is bridged and the other seat is heuristic.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

_WINNER_SEAT = re.compile(r"Anvil\((\d+)\)")


def winner_seat(winner: str | None) -> int | None:
    """Return the zero-based registered seat encoded in a Forge winner name."""
    match = _WINNER_SEAT.search(winner or "")
    if match is None:
        return None
    seat = int(match.group(1)) - 1
    return seat if seat >= 0 else None


def summarize_run(run_dir: str | Path, model_a_seat: int) -> dict:
    """Summarize one orientation of a model A/B match.

    ``model_a_seat`` identifies the seat occupied by A in this run. Games
    that do not finish with ``status == "won"`` are reported as non-decisive
    and are excluded from the decisive-only rate, while the all-games rate
    leaves them as non-wins. This makes crashes visible instead of silently
    dropping them from the headline.
    """
    path = Path(run_dir)
    rows = [json.loads(line) for line in (path / "games.jsonl").read_text().splitlines()]
    statuses = Counter(str(row.get("status", "")) for row in rows)
    decisive = [row for row in rows if row.get("status") == "won"]
    a_wins = 0
    b_wins = 0
    unattributed = 0
    for row in decisive:
        seat = winner_seat(row.get("winner"))
        if seat is None:
            unattributed += 1
        elif seat == model_a_seat:
            a_wins += 1
        elif seat == 1 - model_a_seat:
            b_wins += 1
        else:
            unattributed += 1

    return {
        "run": str(path),
        "model_a_seat": model_a_seat,
        "games": len(rows),
        "decisive": len(decisive),
        "nondecisive": len(rows) - len(decisive),
        "a_wins": a_wins,
        "b_wins": b_wins,
        "unattributed_decisive": unattributed,
        "crashes": sum(1 for row in rows if str(row.get("status", "")).startswith("crash")),
        "draw_clock": sum(1 for row in rows if row.get("draw_clock")),
        "statuses": dict(statuses),
    }


def _se(wins: int, n: int) -> float | None:
    if n <= 0:
        return None
    p = wins / n
    return math.sqrt(p * (1 - p) / n)


def aggregate(
    checkpoint_a: str,
    checkpoint_b: str,
    orientations: list[dict],
    seed_base: int,
    games_per_orientation: int,
) -> dict:
    """Aggregate the two mirrored seat orientations into a report."""
    games = sum(row["games"] for row in orientations)
    decisive = sum(row["decisive"] for row in orientations)
    a_wins = sum(row["a_wins"] for row in orientations)
    b_wins = sum(row["b_wins"] for row in orientations)
    nondecisive = sum(row["nondecisive"] for row in orientations)
    all_rate = a_wins / games if games else None
    decisive_rate = a_wins / decisive if decisive else None
    return {
        "checkpoint_a": checkpoint_a,
        "checkpoint_b": checkpoint_b,
        "seed_base": seed_base,
        "games_per_orientation": games_per_orientation,
        "games": games,
        "decisive": decisive,
        "nondecisive": nondecisive,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "unattributed_decisive": sum(row["unattributed_decisive"] for row in orientations),
        "crashes": sum(row["crashes"] for row in orientations),
        "draw_clock": sum(row["draw_clock"] for row in orientations),
        "a_winrate_all": all_rate,
        "a_se_all": _se(a_wins, games),
        "a_winrate_decisive": decisive_rate,
        "a_se_decisive": _se(a_wins, decisive),
        "orientations": orientations,
    }
