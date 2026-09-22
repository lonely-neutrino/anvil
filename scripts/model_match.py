#!/usr/bin/env python3
"""Run a mirrored model-vs-model evaluation.

Checkpoint A occupies seat 0 in the first half and seat 1 in the second half;
checkpoint B occupies the other seat. ``--games`` is per orientation, so the
default 1,000 means 2,000 total games.

Example (the mono-green RL checkpoints):

  ./.venv/bin/python scripts/model_match.py \
      --a data/training/mono-green-rl-4000/iter-024/train/last.pt \
      --b data/training/mono-green-rl-4000/iter-004/train/last.pt \
      --decks monoGreenStompy.dck monoGreenStompy.dck \
      --format Constructed --games 1000 --workers 4 --chunk 10 \
      --port 50071 --seed-base 20260908 --reask \
      --pool-version mono-green-stompy-v1 \
      --out data/runs/mono-green-rl024-vs-rl004.json

This uses the existing ROCm-aware Python environment through the interpreter
that launches the script. It never invokes uv.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import secrets
import socket
import sys
from pathlib import Path

from anvil.evals.model_match import aggregate, summarize_run
from anvil.training.selfplay import RUNS_DIR, _run, _start_server, _stop_server


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.25):
            return True
    except OSError:
        return False


def _matching_run_dirs(purpose: str) -> set[Path]:
    """Return run directories, excluding same-prefix server logs."""
    return {
        path
        for path in (Path(value) for value in glob.glob(str(RUNS_DIR / f"{purpose}-*")))
        if path.is_dir()
    }


def _pin_bridge_deadline(milliseconds: int) -> None:
    """Give cold-start inference enough time without changing other runs."""
    key = "-Danvil.bridge.deadline.ms="
    extra = [
        option
        for option in os.environ.get("ANVIL_EXTRA_JVM_OPTS", "").split()
        if not option.startswith(key)
    ]
    extra.append(f"{key}{milliseconds}")
    os.environ["ANVIL_EXTRA_JVM_OPTS"] = " ".join(extra)


def _harness_command(args: argparse.Namespace, purpose: str, seed_base: int) -> list[str]:
    cmd = [sys.executable, "-m", "anvil.bridge.harness", "launch"]
    if args.pairs_file:
        cmd += [
            "--pairs-file",
            str(args.pairs_file.resolve()),
            "--games-per-pair",
            str(args.games_per_pair),
        ]
    else:
        cmd += ["--decks", *args.decks]
    cmd += [
        "--format",
        args.format,
        "--games",
        str(args.games),
        "--workers",
        str(args.workers),
        "--chunk",
        str(args.chunk),
        "--bridge",
        f"grpc:localhost:{args.port}",
        "--bridge-seats",
        "0,1",
        "--obs",
        "--census",
        "--purpose",
        purpose,
        "--seed-base",
        str(seed_base),
    ]
    if args.pool_version:
        cmd += ["--pool-version", args.pool_version]
    if args.reask:
        cmd += ["--reask"]
    if args.calibrated:
        cmd += ["--calibrated"]
    return cmd


def _run_orientation(
    args: argparse.Namespace,
    label: str,
    checkpoint_seat0: Path,
    checkpoint_seat1: Path,
    seed_base: int,
    log_dir: Path,
) -> Path:
    purpose = f"{args.purpose}-{label}"
    before = _matching_run_dirs(purpose)
    if _port_open(args.port):
        raise SystemExit(
            f"port {args.port} is already in use; stop the existing server or choose another port"
        )
    log_dir.mkdir(parents=True, exist_ok=True)
    server = _start_server(
        str(checkpoint_seat0),
        args.port,
        log_dir / f"{purpose}-server.log",
        sample=False,
        ckpt_seat1=str(checkpoint_seat1),
        device=args.device,
    )
    try:
        _run(_harness_command(args, purpose, seed_base))
    finally:
        _stop_server(server)
    after = _matching_run_dirs(purpose)
    new = sorted(after - before)
    if len(new) != 1:
        raise RuntimeError(f"expected one run directory for {purpose}, got {new}")
    return Path(new[0])


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a mirrored two-checkpoint model match")
    ap.add_argument("--a", required=True, type=Path, help="checkpoint for model A")
    ap.add_argument("--b", required=True, type=Path, help="checkpoint for model B")
    decks = ap.add_mutually_exclusive_group(required=True)
    decks.add_argument("--decks", nargs=2, metavar=("SEAT0", "SEAT1"))
    decks.add_argument("--pairs-file", type=Path)
    ap.add_argument("--games", type=int, default=1000, help="games per seat orientation")
    ap.add_argument("--games-per-pair", type=int, default=5)
    ap.add_argument("--format", default="Commander")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--port", type=int, default=50071)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--bridge-deadline-ms",
        type=int,
        default=15000,
        help="Java bridge request deadline; higher default covers cold GPU inference",
    )
    ap.add_argument("--seed-base", type=int, default=None)
    ap.add_argument("--pool-version", default=None)
    ap.add_argument("--purpose", default="model-match")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--reask", action="store_true")
    ap.add_argument("--calibrated", action="store_true")
    args = ap.parse_args()

    for name, path in (("A", args.a), ("B", args.b)):
        if not path.is_file():
            ap.error(f"checkpoint {name} does not exist: {path}")
    if args.pairs_file and not args.pairs_file.is_file():
        ap.error(f"pairs file does not exist: {args.pairs_file}")
    if args.games <= 0 or args.games_per_pair <= 0 or args.bridge_deadline_ms <= 0:
        ap.error("--games, --games-per-pair, and --bridge-deadline-ms must be positive")
    if args.out.exists():
        ap.error(f"output already exists: {args.out}")
    if "/" in args.purpose or ".." in args.purpose:
        ap.error("--purpose must be a simple run-name component")

    seed_base = args.seed_base
    if seed_base is None:
        seed_base = secrets.randbelow(1 << 62)
        print(f"[model-match] generated seed base {seed_base}")
    _pin_bridge_deadline(args.bridge_deadline_ms)

    out = args.out.resolve()
    log_dir = out.parent
    orientations = []
    # A is seat 0 in the first run and seat 1 in the second. Both runs use
    # the same seed base and pair schedule so the seating correction is
    # explicit and reproducible.
    run_a0 = _run_orientation(args, "a0-b1", args.a.resolve(), args.b.resolve(), seed_base, log_dir)
    orientations.append(summarize_run(run_a0, model_a_seat=0))
    run_a1 = _run_orientation(args, "b0-a1", args.b.resolve(), args.a.resolve(), seed_base, log_dir)
    orientations.append(summarize_run(run_a1, model_a_seat=1))

    report = aggregate(
        str(args.a.resolve()),
        str(args.b.resolve()),
        orientations,
        seed_base,
        args.games,
    )
    report["bridge_deadline_ms"] = args.bridge_deadline_ms
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    def rate_text(rate: float | None, se: float | None) -> str:
        return "n/a" if rate is None else f"{rate:.4f} ± {se:.4f}"

    p_all = report["a_winrate_all"]
    p_dec = report["a_winrate_decisive"]
    print(
        f"[model-match] A wins {report['a_wins']}, B wins {report['b_wins']}, "
        f"non-decisive {report['nondecisive']} / {report['games']}"
    )
    print(
        f"[model-match] A winrate: {rate_text(p_dec, report['a_se_decisive'])} "
        f"(decisive), {rate_text(p_all, report['a_se_all'])} (all games)"
    )
    print(f"[model-match] report: {out}")


if __name__ == "__main__":
    main()
