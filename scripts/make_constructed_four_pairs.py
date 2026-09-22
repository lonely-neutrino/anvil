#!/usr/bin/env python3
"""Create a shuffled, balanced ordered-pair schedule for the four decks.

The schedule includes self-matches. Every ordered combination is represented
as evenly as possible, while the overall order is seeded and randomized so
the harness does not run one matchup in a large contiguous block.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

DEFAULT_DECKS = (
    "monoBlueTempo.dck",
    "monoGreenStompy.dck",
    "monoRedAggro.dck",
    "monoWhiteWeenie.dck",
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True, help="output tab-separated pairs file")
    ap.add_argument("--pairs", type=int, required=True, help="number of scheduled pair lines")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--decks", nargs="+", default=DEFAULT_DECKS)
    args = ap.parse_args()

    if len(args.decks) < 1 or len(set(args.decks)) != len(args.decks):
        ap.error("--decks must contain at least one unique deck filename")
    if args.pairs <= 0:
        ap.error("--pairs must be positive")

    ordered = [(seat0, seat1) for seat0 in args.decks for seat1 in args.decks]
    pairs = [ordered[i % len(ordered)] for i in range(args.pairs)]
    random.Random(args.seed).shuffle(pairs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(f"{seat0}\t{seat1}\n" for seat0, seat1 in pairs))
    print(
        f"wrote {len(pairs)} scheduled pairs ({len(ordered)} ordered combinations) "
        f"to {args.out}"
    )


if __name__ == "__main__":
    main()
