#!/usr/bin/env python3
"""Run the Brave the Elements candidate-drill pipeline over a self-play corpus.

The driver is specific only in its default SA filter.  It discovers the
source stores from a self-play ``loop_state.json``, mines legal Brave options,
plans one candidate campaign per source-policy checkpoint, runs paired
natural/forced rollouts, and can pass all labels and natural stores to the
candidate auxiliary loss.

The source corpus is replayed with the checkpoint that generated each
iteration.  This is important for this corpus: ``run.json`` records the game
recipe, but not the self-play checkpoint, so the mapping is reconstructed as

    iteration 0 -> loop_config.json["ckpt"]
    iteration i -> iter-(i-1)/train/last.pt

Typical smoke test:

    ./.venv/bin/python scripts/run_brave_candidate_pipeline.py \
        --out /tmp/brave-candidate-smoke \
        --end-iteration 1 --k 2 --workers 1 --chunk 1

Full paired campaign, followed by candidate-loss training:

    ./.venv/bin/python scripts/run_brave_candidate_pipeline.py \
        --out data/candidate-drills/monowhite-oldBC-brave \
        --k 32 --train

The default loop directory is the 55-iteration monoWhite corpus named in the
experiment request.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from scripts.mine_priority_candidates import _natural_candidate, mine

REPO = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO / "data" / "runs"
DEFAULT_LOOP = REPO / "data" / "training" / "monowhiteweenie-rl-20260923-oldBC-color"
DEFAULT_BRAVE_REGEX = (
    r"(?:Brave the Elements|Choose a color\.\s*"
    r"White creatures you control gain protection from the chosen color)"
)


@dataclass(frozen=True)
class Source:
    iteration: int
    trajectory: Path
    run: Path
    checkpoint: Path
    model_seats: tuple[int, ...]


def repo_path(raw: str | Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = REPO / path
    return path.resolve()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def bridge_seats(run_json: Path) -> tuple[int, ...]:
    """Return the seats controlled by the model for one source run."""
    raw = read_json(run_json).get("bridge_seats")
    if raw is None or str(raw).strip() == "":
        # The mirror batch is bridged on both seats.
        return (0, 1)
    if isinstance(raw, list):
        return tuple(sorted({int(x) for x in raw}))
    return tuple(sorted({int(x) for x in str(raw).split(",") if x.strip()}))


def initial_checkpoint(loop_dir: Path, override: str | None) -> Path:
    raw = override
    if raw is None:
        config = loop_dir / "loop_config.json"
        if not config.exists():
            raise SystemExit(f"missing self-play config: {config}")
        raw = read_json(config).get("ckpt")
    if not raw:
        raise SystemExit(
            "could not determine iteration-0 checkpoint; pass --initial-ckpt"
        )
    path = repo_path(raw)
    if not path.exists():
        raise SystemExit(f"iteration-0 checkpoint does not exist: {path}")
    return path


def current_checkpoint(loop_dir: Path, override: str | None) -> Path:
    if override:
        path = repo_path(override)
    else:
        state = loop_dir / "loop_state.json"
        raw = read_json(state).get("ckpt")
        if not raw:
            raise SystemExit(f"loop state has no current checkpoint: {state}")
        path = repo_path(raw)
    if not path.exists():
        raise SystemExit(f"current/drill checkpoint does not exist: {path}")
    return path


def discover_sources(
    loop_dir: Path,
    start_iteration: int,
    end_iteration: int,
    initial_override: str | None,
) -> list[Source]:
    state_path = loop_dir / "loop_state.json"
    if not state_path.exists():
        raise SystemExit(f"missing loop state: {state_path}")
    state = read_json(state_path)
    groups = state.get("stores") or []
    if end_iteration >= len(groups):
        raise SystemExit(
            f"loop state contains {len(groups)} iterations, but "
            f"iteration {end_iteration} was requested"
        )
    first = initial_checkpoint(loop_dir, initial_override)
    sources: list[Source] = []
    for iteration in range(start_iteration, end_iteration + 1):
        if iteration == 0:
            ckpt = first
        else:
            ckpt = loop_dir / f"iter-{iteration - 1:03d}" / "train" / "last.pt"
        if not ckpt.exists():
            raise SystemExit(
                f"missing source checkpoint for iteration {iteration}: {ckpt}"
            )
        for raw_store in groups[iteration]:
            trajectory = repo_path(raw_store)
            run = RUNS_DIR / trajectory.name
            run_json = run / "run.json"
            if not trajectory.exists():
                raise SystemExit(f"missing trajectory store: {trajectory}")
            if not run_json.exists():
                raise SystemExit(f"missing source run manifest: {run_json}")
            sources.append(
                Source(
                    iteration=iteration,
                    trajectory=trajectory,
                    run=run,
                    checkpoint=ckpt.resolve(),
                    model_seats=bridge_seats(run_json),
                )
            )
    return sources


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mine_corpus(
    sources: list[Source],
    out_dir: Path,
    sa_regex: str,
    limit: int,
) -> tuple[list[dict], dict]:
    """Mine each bridged-seat class, then combine into one stable point file."""
    by_seats: dict[tuple[int, ...], list[Source]] = defaultdict(list)
    for source in sources:
        by_seats[source.model_seats].append(source)

    all_rows: list[dict] = []
    group_stats: dict[str, dict] = {}
    for seats, group in sorted(by_seats.items()):
        name = "-".join(map(str, seats))
        group_out = out_dir / "mined" / f"points-seats-{name}.jsonl"
        stats = mine(
            [source.trajectory for source in group],
            group_out,
            model_seats=set(seats),
            sa_pattern=sa_regex,
            limit=limit,
            store_labels=[source.run.name for source in group],
        )
        all_rows.extend(load_jsonl(group_out))
        group_stats["+".join(map(str, seats))] = {
            **stats,
            "stores": len(group),
        }

    all_rows.sort(
        key=lambda row: (
            str(row.get("store", "")),
            int(row.get("g", row.get("game", 0))),
            int(row.get("window", row.get("windowId", -1))),
            int((row.get("candidate") or {}).get("entity", -1)),
            str((row.get("candidate") or {}).get("sa", "")),
        )
    )
    mined_points = len(all_rows)
    mined_windows = len(
        {(row["store"], int(row["g"]), int(row["window"])) for row in all_rows}
    )
    all_rows = retain_balanced_windows(all_rows, sources, limit)
    points_path = out_dir / "points.jsonl"
    write_jsonl(points_path, all_rows)
    stats = {
        "sa_regex": sa_regex,
        "source_stores": len(sources),
        "source_iterations": len({source.iteration for source in sources}),
        "mined_points": mined_points,
        "mined_windows": mined_windows,
        "points": len(all_rows),
        "windows": len(
            {
                (row["store"], int(row["g"]), int(row["window"]))
                for row in all_rows
            }
        ),
        "groups": group_stats,
    }
    (out_dir / "mining-stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(
        f"[brave] mined {stats['points']} points in {stats['windows']} windows "
        f"from {stats['source_stores']} stores"
    )
    return all_rows, stats


def write_iteration_points(
    rows: list[dict], sources: list[Source], out_dir: Path
) -> dict[int, Path]:
    by_store = {source.run.name: source.iteration for source in sources}
    by_iteration: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        store = str(row.get("store", ""))
        if store not in by_store:
            raise SystemExit(f"candidate point references unknown source store: {store}")
        by_iteration[by_store[store]].append(row)
    paths: dict[int, Path] = {}
    for iteration, iteration_rows in sorted(by_iteration.items()):
        path = out_dir / "points" / f"iter-{iteration:03d}.jsonl"
        write_jsonl(path, iteration_rows)
        paths[iteration] = path
    return paths


def retain_balanced_windows(
    rows: list[dict], sources: list[Source], limit: int
) -> list[dict]:
    """Retain evenly spaced source windows while preserving all arms per window."""
    if not limit:
        return rows
    iteration_by_store = {source.run.name: source.iteration for source in sources}
    windows: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        key = (str(row["store"]), int(row["g"]), int(row["window"]))
        windows[key].append(row)
    ordered = sorted(
        windows,
        key=lambda key: (
            iteration_by_store.get(key[0], 10**9),
            key[0],
            key[1],
            key[2],
        ),
    )
    if len(ordered) <= limit:
        return rows
    # Evenly spaced selection gives the full corpus a chance to contribute,
    # rather than taking the first N windows from iteration zero.
    selected_keys = {
        ordered[(index * len(ordered)) // limit] for index in range(limit)
    }
    selected = [row for key in ordered if key in selected_keys for row in windows[key]]
    selected.sort(
        key=lambda row: (
            iteration_by_store.get(str(row["store"]), 10**9),
            str(row["store"]),
            int(row["g"]),
            int(row["window"]),
            int((row.get("candidate") or {}).get("entity", -1)),
            str((row.get("candidate") or {}).get("sa", "")),
        )
    )
    return selected


def run_command(command: list[str]) -> None:
    print("[brave] +", " ".join(command))
    subprocess.run(command, cwd=REPO, check=True)


def campaign_config(args: argparse.Namespace, sources: list[Source], drill_ckpt: Path) -> dict:
    return {
        "version": 3,
        "sa_regex": args.sa_regex,
        "start_iteration": args.start_iteration,
        "end_iteration": args.end_iteration,
        "limit": args.limit,
        "k": args.k,
        "tag_prefix": args.tag_prefix,
        "sample_mainline": not args.no_sample_mainline,
        "drill_checkpoint": str(drill_ckpt.resolve()),
        "source_stores": [str(source.trajectory.resolve()) for source in sources],
        "source_checkpoints": [str(source.checkpoint.resolve()) for source in sources],
    }


def validate_campaign(campaign: dict, expected: dict) -> None:
    actual = campaign.get("config")
    version = campaign.get("version")
    if version not in (2, 3) or not isinstance(actual, dict):
        raise SystemExit(
            "--resume found a legacy campaign manifest without a complete "
            "configuration; use a new --out directory or rerun without --resume"
        )
    actual = dict(actual)
    if version == 2:
        # v2 always used sampled mainline replay unless the new explicit
        # --no-sample-mainline flag was supplied; preserve that old default
        # while still rejecting an explicit mode conflict.
        actual["version"] = expected["version"]
        actual.setdefault("sample_mainline", True)
    mismatches = [
        (key, actual.get(key), value)
        for key, value in expected.items()
        if actual.get(key) != value
    ]
    if mismatches:
        key, old, new = mismatches[0]
        raise SystemExit(
            f"--resume configuration mismatch for {key}: stored={old!r}, "
            f"requested={new!r}; use a new --out directory"
        )


def reusable_plan(
    entry: dict, source: Source, points_path: Path, k: int
) -> bool:
    try:
        plan_dir = repo_path(entry["plan"])
        manifest = read_json(plan_dir / "manifest.json")
        return (
            bool(manifest.get("candidate_mode"))
            and int(manifest.get("k", -1)) == k
            and repo_path(manifest.get("ckpt", "")) == source.checkpoint.resolve()
            and repo_path(manifest.get("candidate_points", "")) == points_path.resolve()
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def plan_iteration(
    points: Path,
    source_checkpoint: Path,
    plan_dir: Path,
    k: int,
    tag: str,
) -> None:
    plan_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            sys.executable,
            "-m",
            "anvil.grindstone",
            "plan",
            "--candidate-points",
            str(points),
            "--out",
            str(plan_dir),
            "--ckpt",
            str(source_checkpoint),
            "--k",
            str(k),
            "--tag",
            tag,
        ]
    )


def generate_iteration(
    plan_dir: Path,
    tag: str,
    drill_checkpoint: Path,
    k: int,
    port: int,
    workers: int,
    chunk: int,
    sample_mainline: bool,
    server_device: str,
    max_batch: int,
    batch_window_ms: float,
) -> list[Path]:
    before = {path.resolve() for path in RUNS_DIR.glob(f"drill{tag}-*")}
    command = [
        sys.executable,
        "-m",
        "anvil.grindstone",
        "generate",
        "--manifest",
        str(plan_dir),
        "--force-candidate",
        "--drill-ckpt",
        str(drill_checkpoint),
        "--k",
        str(k),
        "--port",
        str(port),
        "--workers",
        str(workers),
        "--chunk",
        str(chunk),
        "--device",
        server_device,
        "--max-batch",
        str(max_batch),
        "--batch-window-ms",
        str(batch_window_ms),
    ]
    if sample_mainline:
        command.append("--sample-mainline")
    run_command(command)
    after = {path.resolve() for path in RUNS_DIR.glob(f"drill{tag}-*")}
    new_runs = sorted(after - before)
    if not new_runs:
        raise SystemExit(f"candidate generation produced no run dirs for tag {tag}")
    return new_runs


def label_files(run_dirs: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for run_dir in run_dirs:
        files.extend(sorted(run_dir.glob("workers/inv-*/labels.jsonl")))
    return files


def _candidate_label_key(row: dict) -> tuple | None:
    candidate = row.get("candidate") or {}
    try:
        return (
            int(row["i"]),
            int(row["window"]),
            int(candidate["entity"]),
            str(candidate["sa"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _candidate_target_keys(path: Path) -> set[tuple] | None:
    keys: set[tuple] = set()
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        if not line or line.startswith("#") or line.startswith("gameIdx\t"):
            continue
        fields = line.split("\t")
        if len(fields) != 8:
            return None
        try:
            keys.add((int(fields[0]), int(fields[1]), int(fields[6]), fields[7]))
        except ValueError:
            return None
    return keys


def _run_games_complete(run: Path) -> bool:
    try:
        config = read_json(run / "run.json")
        start = int(config.get("start_index", 0))
        total = int(config["games"])
    except (KeyError, OSError, TypeError, ValueError):
        return False
    expected = set(range(start, start + total))
    completed: set[int] = set()
    for path in sorted(run.glob("workers/inv-*/games.jsonl")):
        for line in path.read_text().splitlines():
            try:
                index = int(json.loads(line)["i"])
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if index in expected:
                completed.add(index)
    skip_path = run / "skips.json"
    if skip_path.exists():
        try:
            completed.update(
                int(index)
                for index in read_json(skip_path)["indices"]
                if int(index) in expected
            )
        except (KeyError, OSError, TypeError, ValueError):
            return False
    return bool(expected) and completed == expected


def _candidate_label_keys(run: Path) -> set[tuple]:
    keys: set[tuple] = set()
    for path in label_files([run]):
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("ev") == "candidate":
                key = _candidate_label_key(row)
                if key is not None:
                    keys.add(key)
    return keys


def rollout_runs_complete(run_dirs: Iterable[Path], plan_dir: Path) -> bool:
    """Require complete harness runs and one label row per planned arm."""
    try:
        manifest = read_json(repo_path(plan_dir) / "manifest.json")
        arms = manifest["arms"]
    except (KeyError, OSError, TypeError, ValueError):
        return False
    runs_by_purpose: dict[str, Path] = {}
    for raw_run in run_dirs:
        run = repo_path(raw_run)
        try:
            purpose = str(read_json(run / "run.json")["purpose"])
        except (KeyError, OSError, TypeError, ValueError):
            return False
        runs_by_purpose[purpose] = run
    if len(runs_by_purpose) != len(arms):
        return False
    for arm in arms:
        purpose = "drill{}-{}".format(manifest.get("tag", ""), arm["store"])
        run = runs_by_purpose.get(purpose)
        expected = _candidate_target_keys(repo_path(arm["candidate_file"]))
        if run is None or expected is None or not expected:
            return False
        if not _run_games_complete(run):
            return False
        if not expected.issubset(_candidate_label_keys(run)):
            return False
    return True


def summarize_labels(files: Iterable[Path]) -> dict:
    files = list(files)
    rows = []
    skip_counts: Counter[str] = Counter()
    for path in files:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("ev") != "candidate":
                continue
            rows.append(row)
            for reason, count in (row.get("skip_counts") or {}).items():
                skip_counts[str(reason)] += int(count)
    paired = sum(int(row.get("paired", 0)) for row in rows)
    forced_realized = sum(int(row.get("forced_realized", 0)) for row in rows)
    forced_total = sum(int(row.get("forced_total", 0)) for row in rows)
    return {
        "label_files": len(files),
        "candidate_rows": len(rows),
        "paired": paired,
        "forced_realized": forced_realized,
        "forced_total": forced_total,
        "realization_rate": forced_realized / forced_total if forced_total else None,
        "skip_counts": dict(skip_counts),
    }


def _policy_probability_mass(
    windows: list[tuple[dict, dict]],
    sa_regex: str,
    ckpt: Path | None,
    device: str,
) -> dict | None:
    """Score Brave and PASS probability mass on fresh eval windows."""
    if ckpt is None:
        return None
    if not windows:
        return {
            "windows": 0,
            "unmapped_windows": 0,
            "brave": None,
            "pass": None,
        }
    from anvil.bridge.server import ModelBackend

    regex = re.compile(sa_regex, re.IGNORECASE)
    backend = ModelBackend(str(ckpt), pass_delta=0.0, device=device, sample=False)
    brave_mass = 0.0
    pass_mass = 0.0
    scored = 0
    unmapped = 0
    for header, dec in windows:
        ex, aux = backend.feat.example(dec, header, "priority")
        out = backend.batcher.submit(ex, backend.pass_delta, noise=None)
        probs = out["choice_probs"][0].float().detach().cpu()
        aliases = aux.get("candidate_key_aliases") or []
        if not aliases:
            aliases = [
                [key] if key is not None else []
                for key in aux.get("candidate_keys", [])
            ]
        brave_indices = [
            index
            for index, keys in enumerate(aliases)
            if index > 0 and any(regex.search(str(key[1])) for key in keys)
        ]
        if not brave_indices or max(brave_indices) >= probs.shape[0]:
            unmapped += 1
            continue
        brave_mass += float(probs[brave_indices].sum())
        pass_mass += float(probs[0])
        scored += 1
    return {
        "windows": scored,
        "unmapped_windows": unmapped,
        "brave": brave_mass / scored if scored else None,
        "pass": pass_mass / scored if scored else None,
    }


def _fresh_eval_telemetry(
    run_dirs: list[tuple[Path, int]],
    sa_regex: str,
    ckpt: Path | None = None,
    device: str = "cuda:0",
) -> dict:
    """Summarize unforced observation frames and game outcomes from eval arms."""
    from anvil.store.trajectories import decode_frame
    from anvil.training.dataset import norm_sa

    regex = re.compile(sa_regex, re.IGNORECASE)
    available = 0
    selected = 0
    decode_failures = 0
    games = 0
    wins = 0
    draws = 0
    policy_windows: list[tuple[dict, dict]] = []
    for run_dir, seat in run_dirs:
        outcome_rows: dict[int, dict] = {}
        for path in sorted(run_dir.glob("workers/inv-*/games.jsonl")):
            for line in path.read_text().splitlines():
                try:
                    row = json.loads(line)
                    outcome_rows.setdefault(int(row["i"]), row)
                except (KeyError, json.JSONDecodeError, TypeError, ValueError):
                    continue
        games += len(outcome_rows)
        for row in outcome_rows.values():
            if row.get("status") == "won" and f"Anvil({seat + 1})" in (row.get("winner") or ""):
                wins += 1
            elif row.get("status") != "won":
                draws += 1

        seen_games: set[int] = set()
        for worker in sorted(run_dir.glob("workers/inv-*")):
            if not worker.is_dir():
                continue
            index_path = worker / "obs.idx.jsonl"
            frame_path = worker / "obs.zst"
            if not index_path.exists() or not frame_path.exists():
                continue
            with frame_path.open("rb") as stream:
                for line in index_path.read_text().splitlines():
                    try:
                        entry = json.loads(line)
                        game = int(entry["g"])
                    except (KeyError, json.JSONDecodeError, TypeError, ValueError):
                        continue
                    if game in seen_games:
                        continue
                    seen_games.add(game)
                    try:
                        stream.seek(int(entry["off"]))
                        header, decisions, _end, _marks = decode_frame(
                            stream.read(int(entry["clen"]))
                        )
                    except Exception:  # noqa: BLE001
                        decode_failures += 1
                        continue
                    # Keep the source frame for offline policy-mass scoring.
                    for dec in decisions:
                        if dec.get("m") != "chooseSpellAbilityToPlay":
                            continue
                        opts = dec.get("opts") or []
                        brave = [
                            option
                            for option in opts
                            if option.get("e") is not None
                            and regex.search(norm_sa(str(option.get("sa", ""))))
                        ]
                        if not brave:
                            continue
                        available += 1
                        natural = _natural_candidate(dec, opts)
                        if natural is not None and regex.search(natural[1]):
                            selected += 1
                        policy_windows.append((header, dec))
    policy_mass = _policy_probability_mass(policy_windows, sa_regex, ckpt, device)
    return {
        "games": games,
        "model_wins": wins,
        "draws_or_nonwins": draws,
        "model_win_rate": wins / games if games else None,
        "brave_available_windows": available,
        "brave_selected_windows": selected,
        "brave_selection_rate_conditional": selected / available if available else None,
        "decode_failures": decode_failures,
        # This is the pre-intervention categorical mass on the same fresh
        # unforced windows; it is separate from the argmax action telemetry.
        "policy_probability_mass_brave_vs_pass": policy_mass,
    }


def evaluate_fresh(
    sources: list[Source], ckpt: Path, args: argparse.Namespace, out_dir: Path
) -> dict:
    """Run fresh, unforced monoWhite-style arms for both model seats."""
    source_cfg = read_json(sources[0].run / "run.json")
    decks = list(args.eval_decks or source_cfg.get("decks") or [])
    if len(decks) != 2:
        raise SystemExit(
            "fresh evaluation needs exactly two fixed decks; pass --eval-decks D1 D2"
        )
    game_format = args.eval_format or source_cfg.get("format", "Constructed")
    pool_version = args.eval_pool_version or source_cfg.get("pool_version")
    server_log = out_dir / "eval-server.log"
    from anvil.training.selfplay import _start_server, _stop_server

    server = _start_server(
        str(ckpt),
        args.eval_port,
        server_log,
        sample=False,
        device=args.server_device,
        max_batch=args.max_batch,
        batch_window_ms=args.batch_window_ms,
    )
    runs: list[tuple[Path, int]] = []
    try:
        for seat in (0, 1):
            purpose = f"{args.tag_prefix}-eval-s{seat}"
            before = {path.resolve() for path in RUNS_DIR.glob(f"{purpose}-*")}
            command = [
                sys.executable,
                "-m",
                "anvil.bridge.harness",
                "launch",
                "--decks",
                *decks,
                "--format",
                game_format,
                "--games",
                str(args.eval_games),
                "--workers",
                str(args.eval_workers),
                "--chunk",
                str(args.eval_chunk),
                "--calibrated",
                "--bridge",
                f"grpc:localhost:{args.eval_port}",
                "--bridge-seats",
                str(seat),
                "--obs",
                "--census",
                "--purpose",
                purpose,
                "--seed-base",
                str(args.eval_seed_base),
            ]
            if pool_version:
                command += ["--pool-version", str(pool_version)]
            if source_cfg.get("reask"):
                command.append("--reask")
            run_command(command)
            matches = sorted(
                path.resolve()
                for path in RUNS_DIR.glob(f"{purpose}-*")
                if path.resolve() not in before
            )
            if len(matches) != 1:
                raise SystemExit(
                    f"fresh evaluation expected one run for {purpose}, found {matches}"
                )
            runs.append((matches[0], seat))
    finally:
        _stop_server(server)

    report = {
        "checkpoint": str(ckpt),
        "games_per_seat": args.eval_games,
        "runs": [str(path) for path, _seat in runs],
        "metrics": _fresh_eval_telemetry(
            runs, args.sa_regex, ckpt=ckpt, device=args.server_device
        ),
    }
    path = out_dir / "evaluation.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[brave] fresh evaluation: {json.dumps(report['metrics'], sort_keys=True)}")
    print(f"[brave] evaluation report: {path}")
    return report


def train_candidate_loss(
    sources: list[Source],
    rollout_runs: list[Path],
    drill_checkpoint: Path,
    args: argparse.Namespace,
    out_dir: Path,
) -> None:
    files = label_files(rollout_runs)
    if not files:
        raise SystemExit("cannot train: no candidate labels were produced")
    stores = [str(source.trajectory) for source in sources]
    command = [
        sys.executable,
        "-m",
        "anvil.training.rl",
        "--store",
        ",".join(stores),
        "--ckpt",
        str(drill_checkpoint),
        "--out",
        str(out_dir / "train"),
        "--epochs",
        str(args.epochs),
        "--workers",
        str(args.rl_workers),
        "--candidate-labels",
        ",".join(str(run_dir) for run_dir in rollout_runs),
        "--candidate-stores",
        ",".join(stores),
        "--candidate-frac",
        str(args.candidate_frac),
        "--candidate-calib-steps",
        str(args.candidate_calib_steps),
        "--candidate-margin",
        str(args.candidate_margin),
        "--device",
        args.device,
    ]
    if args.candidate_w is not None:
        command += ["--candidate-w", str(args.candidate_w)]
    if args.max_traj:
        command += ["--max-traj", str(args.max_traj)]
    run_command(command)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--loop-dir", type=Path, default=DEFAULT_LOOP)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start-iteration", type=int, default=0)
    ap.add_argument("--end-iteration", type=int, default=54)
    ap.add_argument("--initial-ckpt", default=None)
    ap.add_argument(
        "--drill-ckpt",
        default=None,
        help="current checkpoint for forced/natural candidate completions; "
        "defaults to the loop's final checkpoint",
    )
    ap.add_argument("--sa-regex", default=DEFAULT_BRAVE_REGEX)
    ap.add_argument(
        "--limit",
        type=int,
        default=50,
        help="retain at most this many source windows globally; 0 means all",
    )
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--port", type=int, default=50123)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=1)
    ap.add_argument(
        "--no-sample-mainline",
        action="store_true",
        help="use argmax source replay; sampled self-play sources use the default",
    )
    ap.add_argument("--mine-only", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--skip-rollouts", action="store_true")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="reuse points, plans, and completed rollout runs recorded in --out",
    )
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--rl-workers", type=int, default=1)
    ap.add_argument("--max-traj", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--server-device",
        default="cuda:0",
        help="PyTorch device for the rollout model server; use cpu for a CPU smoke test",
    )
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument(
        "--batch-window-ms",
        type=float,
        default=12.0,
        help="match the source self-play server's batching window (milliseconds)",
    )
    ap.add_argument("--candidate-frac", type=float, default=0.1)
    ap.add_argument("--candidate-w", type=float, default=None)
    ap.add_argument("--candidate-calib-steps", type=int, default=50)
    ap.add_argument("--candidate-margin", type=float, default=6.0)
    ap.add_argument(
        "--eval-games",
        type=int,
        default=0,
        help="run this many fresh unforced games per model seat after the "
        "campaign; 0 disables evaluation",
    )
    ap.add_argument("--eval-workers", type=int, default=4)
    ap.add_argument("--eval-chunk", type=int, default=10)
    ap.add_argument("--eval-port", type=int, default=50150)
    ap.add_argument("--eval-seed-base", type=int, default=20260927)
    ap.add_argument("--eval-ckpt", default=None)
    ap.add_argument("--eval-decks", nargs=2, default=None)
    ap.add_argument("--eval-format", default=None)
    ap.add_argument("--eval-pool-version", default=None)
    ap.add_argument("--tag-prefix", default="brave")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_iteration < 0 or args.end_iteration < args.start_iteration:
        raise SystemExit("invalid iteration range")
    if args.k <= 0:
        raise SystemExit("--k must be positive")
    if not re.fullmatch(r"[A-Za-z0-9]+", args.tag_prefix):
        raise SystemExit("--tag-prefix must contain only letters and digits")
    if args.mine_only and (args.plan_only or args.train):
        raise SystemExit("--mine-only cannot be combined with --plan-only or --train")
    if args.plan_only and args.train:
        raise SystemExit("--plan-only cannot be combined with --train")
    if args.eval_games < 0:
        raise SystemExit("--eval-games must be nonnegative")
    if args.eval_games and (args.eval_workers <= 0 or args.eval_chunk <= 0):
        raise SystemExit("--eval-workers and --eval-chunk must be positive")

    loop_dir = repo_path(args.loop_dir)
    out_dir = repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = discover_sources(
        loop_dir,
        args.start_iteration,
        args.end_iteration,
        args.initial_ckpt,
    )
    drill_ckpt = current_checkpoint(loop_dir, args.drill_ckpt)
    print(
        f"[brave] corpus: iterations {args.start_iteration}-{args.end_iteration}, "
        f"{len(sources)} stores; drill ckpt={drill_ckpt}"
    )

    campaign_path = out_dir / "campaign.json"
    expected_config = campaign_config(args, sources, drill_ckpt)
    if not args.mine_only:
        if args.resume:
            if not campaign_path.exists():
                raise SystemExit(
                    "--resume requires campaign.json with a complete configuration"
                )
            campaign = read_json(campaign_path)
            validate_campaign(campaign, expected_config)
            if campaign.get("version") == 2:
                campaign = dict(campaign)
                campaign["version"] = 3
                campaign["config"] = expected_config
        else:
            campaign = {"version": 3, "config": expected_config, "plans": []}
        campaign_path.write_text(json.dumps(campaign, indent=2) + "\n")

    points_path = out_dir / "points.jsonl"
    if args.resume and points_path.exists():
        points = load_jsonl(points_path)
        mining_stats = (
            read_json(out_dir / "mining-stats.json")
            if (out_dir / "mining-stats.json").exists()
            else {"points": len(points)}
        )
        print(f"[brave] reusing {len(points)} mined points from {points_path}")
    else:
        points, mining_stats = mine_corpus(
            sources,
            out_dir,
            args.sa_regex,
            args.limit,
        )
    if args.mine_only:
        return
    if not points:
        raise SystemExit(
            "no Brave the Elements candidate points were found; "
            "inspect mining-stats.json or run with --sa-regex '.*'"
        )

    iteration_points = write_iteration_points(points, sources, out_dir)
    source_by_iteration = {
        iteration: next(source for source in sources if source.iteration == iteration)
        for iteration in iteration_points
    }
    all_rollout_runs: list[Path] = []
    by_iteration = {
        int(entry["iteration"]): entry for entry in campaign.get("plans", [])
    }

    for iteration, points_path in sorted(iteration_points.items()):
        tag = f"{args.tag_prefix}i{iteration:03d}"
        plan_dir = out_dir / "plans" / f"iter-{iteration:03d}"
        plan_entry = by_iteration.get(iteration)
        if not (
            args.resume
            and plan_entry
            and reusable_plan(
                plan_entry,
                source_by_iteration[iteration],
                points_path,
                args.k,
            )
        ):
            plan_iteration(
                points_path,
                source_by_iteration[iteration].checkpoint,
                plan_dir,
                args.k,
                tag,
            )
            plan_entry = {
                "iteration": iteration,
                "tag": tag,
                "plan": str(plan_dir),
                "source_checkpoint": str(source_by_iteration[iteration].checkpoint),
                "points": str(points_path),
            }
            by_iteration[iteration] = plan_entry
            campaign["plans"] = [
                by_iteration[key] for key in sorted(by_iteration)
            ]
        if not args.plan_only and not args.skip_rollouts:
            saved_runs = [
                repo_path(path) for path in (plan_entry.get("rollout_runs") or [])
            ]
            if args.resume and rollout_runs_complete(saved_runs, plan_dir):
                runs = saved_runs
                print(f"[brave] reusing rollout runs for iteration {iteration}")
            else:
                runs = generate_iteration(
                    plan_dir,
                    tag,
                    drill_ckpt,
                    args.k,
                    args.port,
                    args.workers,
                    args.chunk,
                    not args.no_sample_mainline,
                    args.server_device,
                    args.max_batch,
                    args.batch_window_ms,
                )
            all_rollout_runs.extend(runs)
            plan_entry["rollout_runs"] = [str(run) for run in runs]
            campaign["plans"] = [by_iteration[key] for key in sorted(by_iteration)]
        campaign_path.write_text(json.dumps(campaign, indent=2) + "\n")

    if args.plan_only:
        print(f"[brave] plans written under {out_dir / 'plans'}")
        return
    if args.skip_rollouts:
        print("[brave] rollout generation skipped")
        return

    files = label_files(all_rollout_runs)
    label_summary = summarize_labels(files)
    (out_dir / "label-summary.json").write_text(
        json.dumps(label_summary, indent=2) + "\n"
    )
    (out_dir / "rollout-runs.json").write_text(
        json.dumps([str(run) for run in all_rollout_runs], indent=2) + "\n"
    )
    print(f"[brave] labels: {json.dumps(label_summary, sort_keys=True)}")
    if not files:
        raise SystemExit("rollouts completed without candidate label files")

    if args.train:
        if label_summary["paired"] <= 0:
            raise SystemExit(
                "cannot train: no valid paired candidate outcomes were produced; "
                "inspect label-summary.json before retrying"
            )
        train_candidate_loss(sources, all_rollout_runs, drill_ckpt, args, out_dir)
        print(f"[brave] candidate training checkpoint: {out_dir / 'train' / 'last.pt'}")

    if args.eval_games:
        eval_ckpt = repo_path(args.eval_ckpt) if args.eval_ckpt else (
            out_dir / "train" / "last.pt" if args.train else drill_ckpt
        )
        if not eval_ckpt.exists():
            raise SystemExit(f"fresh evaluation checkpoint does not exist: {eval_ckpt}")
        evaluate_fresh(sources, eval_ckpt, args, out_dir)


if __name__ == "__main__":
    main()
