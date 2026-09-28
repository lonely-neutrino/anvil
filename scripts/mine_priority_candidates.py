#!/usr/bin/env python3
"""Mine generic priority-candidate drill points from observation stores.

The miner is deliberately outcome-blind: it only reads the decision window
and its legal options.  A candidate is identified by the same pair used by
the loader and serving path, ``(host entity id, normalized spell ability)``.

Example::

    uv run python scripts/mine_priority_candidates.py \
        --store data/trajectories/source-run \
        --model-seats 1 --sa-regex 'Brave the Elements' \
        --out brave-points.jsonl

The JSONL output is also accepted by ``grindstone plan --candidate-points``.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import re
from collections import Counter
from pathlib import Path

from anvil.store.castplan import ret_plans
from anvil.store.trajectories import open_store
from anvil.training.dataset import norm_sa


def _ints(raw: str | None) -> set[int] | None:
    if raw is None or not raw.strip():
        return None
    return {int(x) for x in raw.split(",") if x.strip()}


def _natural_candidate(dec: dict, opts: list[dict]) -> tuple[int, str] | None:
    """Return the logged natural candidate when it can be resolved exactly."""
    oi = dec.get("oi")
    if isinstance(oi, int) and 0 <= oi < len(opts):
        o = opts[oi]
        if o.get("e") is not None:
            return int(o["e"]), norm_sa(str(o.get("sa", "")))
    plans = ret_plans(dec.get("ret"))
    if not plans:
        return None
    plan = plans[0]
    if plan.get("e") is None:
        return None
    return int(plan["e"]), norm_sa(str(plan.get("sa", "")))


def _quiescent_priority(dec: dict) -> bool:
    """Keep only stack-empty windows when the structured obs is available.

    Older stores may not carry an observation payload; retain those rows for
    backwards-compatible mining and let replay guards decide whether they can
    be evaluated.  Current structured stores expose stack objects with
    ``z == "stack"``.
    """
    obs = dec.get("obs")
    if not isinstance(obs, dict):
        return True
    return not obs.get("stack") and not any(
        entity.get("z") == "stack" for entity in (obs.get("ents") or [])
    )


def _window_key(row: dict) -> tuple[str, int, int]:
    return str(row["store"]), int(row["g"]), int(row["window"])


def _window_score(key: tuple[str, int, int]) -> int:
    raw = "\x1f".join(map(str, key)).encode()
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")


def mine(
    stores: list[str | Path],
    out: str | Path,
    model_seats: set[int] | None = None,
    sa_pattern: str | None = None,
    limit: int = 0,
    store_labels: list[str] | None = None,
) -> dict:
    """Mine points and write JSONL, returning counters for tests/callers."""
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    regex = re.compile(sa_pattern, re.IGNORECASE) if sa_pattern else None
    if store_labels is not None and len(store_labels) != len(stores):
        raise ValueError("--store-label must be supplied once per --store")
    counts: Counter[str] = Counter()
    rows: list[dict] = []
    sampled: dict[tuple[str, int, int], list[dict]] = {}
    # A min-heap of retained scores.  The smallest retained score is the root,
    # so a larger score can replace it; the retained set is a deterministic
    # uniform-looking sample of windows across the complete source corpus
    # rather than a prefix sample.
    heap: list[tuple[int, tuple[str, int, int]]] = []

    def retain(row: dict) -> None:
        counts["candidate_points_seen"] += 1
        if not limit:
            rows.append(row)
            return
        key = _window_key(row)
        if key in sampled:
            sampled[key].append(row)
            return
        score = _window_score(key)
        entry = (score, key)
        if len(heap) < limit:
            heapq.heappush(heap, entry)
            sampled[key] = [row]
        elif entry > heap[0]:
            _, evicted = heapq.heapreplace(heap, entry)
            sampled.pop(evicted, None)
            sampled[key] = [row]

    for store_no, raw_path in enumerate(stores):
        path = Path(raw_path)
        label = store_labels[store_no] if store_labels else path.name
        store = open_store(path)
        for g in store.game_indices():
            try:
                traj = store.game(g)
            except Exception as exc:  # corrupt source frames are not points
                counts[f"skip_{type(exc).__name__}"] += 1
                continue
            for dec in traj.decisions:
                if dec.get("m") != "chooseSpellAbilityToPlay":
                    continue
                if not _quiescent_priority(dec):
                    counts["stack_filtered"] += 1
                    continue
                seat = int(dec.get("p", -1))
                if model_seats is not None and seat not in model_seats:
                    counts["seat_filtered"] += 1
                    continue
                opts = dec.get("opts") or []
                if not opts:
                    continue
                natural = _natural_candidate(dec, opts)
                seen: set[tuple[int, str]] = set()
                for ordinal, option in enumerate(opts):
                    entity = option.get("e")
                    sa = norm_sa(str(option.get("sa", "")))
                    if entity is None or not sa:
                        counts["malformed_option"] += 1
                        continue
                    candidate = (int(entity), sa)
                    if candidate in seen:
                        continue
                    seen.add(candidate)
                    if regex is not None and not regex.search(sa):
                        continue
                    window = int(dec.get("s", -1))
                    if window < 0:
                        counts["missing_window"] += 1
                        continue
                    row = {
                        "store": label,
                        "g": int(g),
                        # `s` is the stable decision/window id in an obs frame.
                        "window": window,
                        "turn": int(dec.get("t", 0)),
                        "phase": str(
                            dec.get(
                                "phase",
                                dec.get("ph", (dec.get("obs") or {}).get("phase", (dec.get("obs") or {}).get("ph", ""))),
                            )
                        ),
                        "seat": seat,
                        "candidate": {"entity": candidate[0], "sa": candidate[1]},
                        "natural": (
                            "pass"
                            if natural is None
                            else {"entity": natural[0], "sa": natural[1]}
                        ),
                        # Retain the original option ordinal as an audit aid;
                        # the planner assigns its own stable arm ordinal.
                        "option": ordinal,
                    }
                    retain(row)

    if limit:
        rows = [row for key in sorted(sampled) for row in sampled[key]]
    counts["points"] = len(rows)

    # Stable output is important for reproducible manifests and review.
    rows.sort(key=lambda r: (r["store"], r["g"], r["window"], r["seat"], r["candidate"]["entity"], r["candidate"]["sa"]))
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    counts["windows"] = len({(r["store"], r["g"], r["window"]) for r in rows})
    return dict(counts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", action="append", required=True, help="trajectory store (repeatable)")
    ap.add_argument("--store-label", action="append", default=None, help="stable planner label, one per --store")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model-seats", default=None, help="comma-separated zero-based model-controlled seats")
    ap.add_argument("--sa-regex", default=None, help="optional regular expression over normalized SA")
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="deterministically retain at most this many source windows; 0 means all",
    )
    args = ap.parse_args()
    try:
        stats = mine(args.store, args.out, _ints(args.model_seats), args.sa_regex, args.limit, args.store_label)
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"candidate mining failed: {exc}") from exc
    print(json.dumps(stats, sort_keys=True))


if __name__ == "__main__":
    main()
