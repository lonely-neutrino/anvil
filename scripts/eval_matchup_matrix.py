#!/usr/bin/env python3
"""Print a deck matchup win-rate matrix for a completed evaluation.

The positional argument may be either:

* an iteration directory containing ``arms-report.json`` (the normal RL
  layout), or
* a single harness run directory containing ``run.json`` and ``games.jsonl``.

For model-vs-heuristic evaluations, rows are the model's deck and columns are
the opponent's deck.  Mirrored model-seat runs are combined automatically.

Examples:
    python scripts/eval_matchup_matrix.py \
        data/training/constructed-four-rl-from030-heur1/iter-009

    python scripts/eval_matchup_matrix.py \
        data/training/constructed-four-rl-from030-heur1/iter-009 \
        --basis decisive
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

Z95 = 1.959963984540054


def _path_arg(value: str) -> Path:
    """Accept normal POSIX paths and pasted WSL-style backslash paths."""
    return Path(value.replace("\\", "/")).expanduser()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise SystemExit(f"missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {path}: {exc}") from exc


def _resolve_run(path_text: str, eval_dir: Path, anvil_root: Path) -> Path:
    raw = _path_arg(path_text)
    candidates = [
        raw,
        eval_dir / raw,
        anvil_root / raw,
        Path.cwd() / raw,
    ]
    for path in candidates:
        if (path / "run.json").is_file():
            return path.resolve()
    tried = ", ".join(str(p) for p in candidates)
    raise SystemExit(f"could not resolve evaluation run {path_text!r}; tried: {tried}")


def _report_runs(eval_dir: Path, report_key: str | None, anvil_root: Path) -> list[Path]:
    report_path = eval_dir / "arms-report.json"
    if not report_path.is_file():
        return []

    report = _load_json(report_path)
    if isinstance(report, dict) and isinstance(report.get("runs"), list):
        sections = {"report": report}
    elif isinstance(report, dict):
        sections = {
            str(key): value
            for key, value in report.items()
            if isinstance(value, dict) and isinstance(value.get("runs"), list)
        }
    else:
        raise SystemExit(f"expected an object in {report_path}")

    if not sections:
        raise SystemExit(f"no run lists found in {report_path}")
    if report_key is not None:
        if report_key not in sections:
            raise SystemExit(
                f"report key {report_key!r} not found; available keys: "
                + ", ".join(sorted(sections))
            )
        sections = {report_key: sections[report_key]}
    elif len(sections) > 1:
        raise SystemExit(
            f"{report_path} contains multiple evaluations ({', '.join(sorted(sections))}); "
            "use --report-key to select one"
        )

    runs: list[Path] = []
    for section in sections.values():
        runs.extend(_resolve_run(str(p), eval_dir, anvil_root) for p in section["runs"])
    return runs


def _find_runs(eval_dir: Path, report_key: str | None, anvil_root: Path) -> list[Path]:
    from_report = _report_runs(eval_dir, report_key, anvil_root)
    if from_report:
        return from_report
    if (eval_dir / "run.json").is_file() and (
        (eval_dir / "games.jsonl").is_file() or (eval_dir / "workers").is_dir()
    ):
        return [eval_dir.resolve()]
    children = sorted(
        p for p in eval_dir.iterdir() if p.is_dir() and (p / "run.json").is_file()
    )
    if children:
        return children
    raise SystemExit(
        f"{eval_dir} is not an evaluation run or an iteration folder with arms-report.json"
    )


def _games(run_dir: Path) -> list[dict[str, Any]]:
    merged = run_dir / "games.jsonl"
    paths = [merged] if merged.is_file() else sorted((run_dir / "workers").glob("inv-*/games.jsonl"))
    if not paths:
        raise SystemExit(f"no games.jsonl found under {run_dir}")
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSON in {path}:{line_no}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _pair_fallback(run_dir: Path, game_index: int, manifest: dict[str, Any]) -> list[str] | None:
    pairs = run_dir / "pairs.txt"
    if not pairs.is_file():
        return None
    lines = [line.split() for line in pairs.read_text().splitlines() if line.split()]
    if not lines:
        return None
    gpp = int(manifest.get("games_per_pair") or 1)
    start = int(manifest.get("start_index") or 0)
    pair_index = (game_index - start) // gpp
    return lines[pair_index] if 0 <= pair_index < len(lines) else None


def _bridge_seat(manifest: dict[str, Any]) -> int | None:
    value = manifest.get("bridge_seats")
    if value is None:
        return None
    if isinstance(value, list):
        if len(value) != 1:
            raise SystemExit(f"expected one bridged seat, got {value!r}")
        value = value[0]
    text = str(value).strip()
    if text in {"0", "1"}:
        return int(text)
    raise SystemExit(f"could not interpret bridge_seats={value!r}")


def _deck_label(deck: str) -> str:
    name = Path(deck).stem
    if name.startswith("mono") and len(name) > 4:
        name = name[4:]
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return name[:1].upper() + name[1:]


def _wilson(wins: int, n: int) -> tuple[float, float, float]:
    if n <= 0:
        return math.nan, math.nan, math.nan
    p = wins / n
    z2 = Z95 * Z95
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = Z95 * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def _cell_text(wins: int, n: int) -> str:
    if n == 0:
        return "—"
    p, lo, hi = _wilson(wins, n)
    return f"{p:.1%} [{lo:.1%}, {hi:.1%}] ({wins}/{n})"


def aggregate(run_dirs: list[Path], basis: str) -> tuple[list[str], dict[tuple[str, str], dict[str, int]], dict[str, Any]]:
    cells: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"wins": 0, "games": 0, "decisive": 0, "nondecisive": 0}
    )
    decks: set[str] = set()
    total = {"games": 0, "decisive": 0, "nondecisive": 0, "wins": 0}

    for run_dir in run_dirs:
        manifest = _load_json(run_dir / "run.json")
        model_seat = _bridge_seat(manifest)
        if model_seat is None:
            # Useful for a heuristic-vs-heuristic or ordinary seat-0 read when
            # the harness did not bridge a model. The matrix is explicitly
            # seat-0's win rate in this case.
            model_seat = 0
        for row in _games(run_dir):
            pair = row.get("decks")
            if not isinstance(pair, list) or len(pair) != 2:
                pair = _pair_fallback(run_dir, int(row.get("i", 0)), manifest)
            if not isinstance(pair, list) or len(pair) != 2:
                raise SystemExit(f"game {row.get('i')} in {run_dir} has no two-deck matchup")
            pair = [str(pair[0]), str(pair[1])]
            if model_seat not in (0, 1):
                raise SystemExit(f"invalid scored seat {model_seat} in {run_dir}")
            row_deck, col_deck = pair[model_seat], pair[1 - model_seat]
            key = (row_deck, col_deck)
            decks.update(pair)
            status = str(row.get("status", ""))
            decisive = status == "won"
            winner = str(row.get("winner") or "")
            seat_won = f"({model_seat + 1})-" in winner
            win = bool(seat_won and decisive)
            c = cells[key]
            c["games"] += 1
            c["decisive"] += int(decisive)
            c["nondecisive"] += int(not decisive)
            c["wins"] += int(win)
            total["games"] += 1
            total["decisive"] += int(decisive)
            total["nondecisive"] += int(not decisive)
            total["wins"] += int(win)

    for c in cells.values():
        c["n"] = c["games"] if basis == "all" else c["decisive"]
    total["n"] = total["games"] if basis == "all" else total["decisive"]
    return sorted(decks), cells, total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_dir", type=_path_arg, help="iteration folder or harness run folder")
    ap.add_argument(
        "--basis",
        choices=("all", "decisive"),
        default="all",
        help="denominator for rates and CIs; default: all games (non-decisive is a non-win)",
    )
    ap.add_argument("--report-key", help="select a section if arms-report.json contains several evaluations")
    ap.add_argument(
        "--anvil-root",
        type=_path_arg,
        default=Path(__file__).resolve().parents[1],
        help="repository's anvil directory for resolving data/runs paths",
    )
    args = ap.parse_args()

    eval_dir = args.eval_dir.resolve()
    if not eval_dir.is_dir():
        raise SystemExit(f"not a directory: {eval_dir}")
    run_dirs = _find_runs(eval_dir, args.report_key, args.anvil_root.resolve())
    decks, cells, total = aggregate(run_dirs, args.basis)
    labels = {deck: _deck_label(deck) for deck in decks}

    print(f"Evaluation: {eval_dir}")
    print(f"Runs combined: {len(run_dirs)}")
    print(
        f"Basis: {args.basis} games; Wilson 95% CI; "
        f"total {total['wins']}/{total['n']} wins "
        f"({total['games']} games, {total['decisive']} decisive, {total['nondecisive']} non-decisive)"
    )
    print("Rows: scored/model deck; columns: opposing deck")
    print("Cell format: win rate [95% CI] (wins/denominator)\n")

    width = max(24, max((len(labels[d]) for d in decks), default=0) + 2)
    print("".ljust(width) + "".join(labels[d].rjust(36) for d in decks))
    for row_deck in decks:
        values = []
        for col_deck in decks:
            c = cells.get((row_deck, col_deck), {})
            n = c.get("n", 0)
            values.append(_cell_text(c.get("wins", 0), n).rjust(36))
        print(labels[row_deck].ljust(width) + "".join(values))

    print("\nPer-cell denominators:")
    for row_deck in decks:
        parts = []
        for col_deck in decks:
            c = cells.get((row_deck, col_deck), {})
            parts.append(f"{labels[row_deck]} vs {labels[col_deck]}: {c.get('n', 0)}")
        print("  " + "; ".join(parts))


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
