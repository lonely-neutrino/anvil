"""Counterfactual priority-candidate labels and natural-window joins.

Candidate labels are intentionally separate from :mod:`seqlabels`.  The
forced arm is not behavior-policy data and must never be treated as a normal
V-trace trajectory.  A label row is keyed by ``(source store, game,
priority window)`` and carries one generic ``(entity, normalized SA)`` arm.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch


def _files(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            out.extend(sorted(p.glob("workers/inv-*/labels.jsonl")))
            out.extend(sorted(p.glob("labels.jsonl")))
        else:
            out.append(p)
    return list(dict.fromkeys(out))


def _candidate(r: dict) -> tuple[int, str] | None:
    from anvil.training.dataset import norm_sa

    c = r.get("candidate") or {}
    entity = c.get("entity", r.get("entityId", r.get("entity")))
    sa = c.get("sa", r.get("normalizedSA", r.get("sa")))
    if entity is None or sa is None:
        return None
    return int(entity), norm_sa(str(sa))


def _count(r: dict, *names: str) -> int:
    for name in names:
        if r.get(name) is not None:
            return int(r[name])
    return 0


def load_rows(paths: list[str], clip: float = 0.25) -> list[dict]:
    """Load valid paired candidate outcomes and compute signed advantages.

    The parser accepts both the descriptive names used by the implementation
    (``natural_wins``, ``forced_wins``, ``paired``) and compact aliases so
    labels from an older Forge jar can be audited without a conversion step.
    Rows with no valid paired completions are retained as ``skip`` counters
    in ``last_stats`` but are not training examples.
    """
    rows: list[dict] = []
    stats = {
        "files": 0,
        "rows": 0,
        "valid": 0,
        "skipped": 0,
        "realized": 0,
        "skip_reasons": {},
    }

    def add_skip_reason(reason: str, count: int = 1) -> None:
        reasons = stats["skip_reasons"]
        reasons[reason] = reasons.get(reason, 0) + count

    for path in _files(paths):
        stats["files"] += 1
        for line in path.open():
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                stats["skipped"] += 1
                continue
            stats["rows"] += 1
            for reason, count in (raw.get("skip_counts") or {}).items():
                add_skip_reason(str(reason), int(count))
            status = raw.get("status")
            if raw.get("skip") or status in {
                "SKIP",
                "NO_MATCH",
                "SEAT_MISMATCH",
                "NO_ONESHOT",
            }:
                stats["skipped"] += 1
                add_skip_reason(str(status or raw.get("skip") or "SKIP"))
                continue
            cand = _candidate(raw)
            if cand is None:
                stats["skipped"] += 1
                continue
            paired = _count(raw, "paired", "valid_paired", "n", "triples")
            if paired <= 0:
                stats["skipped"] += 1
                if not (raw.get("skip_counts") or {}):
                    add_skip_reason(str(status or "NO_PAIRED"))
                continue
            natural_wins = _count(raw, "natural_wins", "w_natural", "w_nat", "natural")
            forced_wins = _count(raw, "forced_wins", "w_forced", "w_candidate", "w_act")
            adv = (forced_wins - natural_wins) / paired
            adv = max(-clip, min(clip, adv))
            realized = _count(raw, "forced_realized", "realized", "casts", "cast")
            stats["valid"] += 1
            stats["realized"] += realized
            source = raw.get("source_store", raw.get("store", raw.get("source")))
            game = raw.get("game", raw.get("i", raw.get("g")))
            window = raw.get("window", raw.get("windowId", raw.get("fp", raw.get("s"))))
            if source is None or game is None or window is None:
                stats["skipped"] += 1
                stats["valid"] -= 1
                continue
            rows.append(
                {
                    "key": (str(source), int(game), int(window)),
                    "source": str(source),
                    "game": int(game),
                    "window": int(window),
                    "seat": int(raw.get("seat", -1)),
                    "entity": cand[0],
                    "sa": cand[1],
                    "adv": adv,
                    "natural_wr": natural_wins / paired,
                    "forced_wr": forced_wins / paired,
                    "paired": paired,
                    "realized": realized,
                    "forced_total": _count(raw, "forced_total", "attempts", "k") or paired,
                    "valid": True,
                }
            )
    # Last parsed statistics are useful to the CLI and tests without making
    # the row contract carry mutable state.
    load_rows.last_stats = stats  # type: ignore[attr-defined]
    return rows


def _first_window(store, feat, game: int, seat: int, window: int, full_vis: bool = False):
    from anvil.training.rl import game_trajectories

    trajs, _skip = game_trajectories(store, feat, game, full_vis=full_vis)
    for p, examples, _reward, _rej, examples_fv in trajs:
        if p != seat:
            continue
        for j, (ex, rec) in enumerate(examples):
            if int(rec.get("s", -1)) == window:
                return ex, examples_fv[j] if full_vis else None
    return None


def _source_window(store, feat, game: int, seat: int, window: int, full_vis: bool = False):
    """Build a candidate window directly from its natural source store.

    Candidate campaigns are labels-only and therefore do not need to emit a
    second V-trace store for the natural arm.  The source store already
    contains the exact structured priority decision and its serve-equivalent
    history can be rebuilt here.
    """
    from anvil.bridge.featurize import store_wire_hist

    try:
        traj = store.game(game)
    except Exception:
        return None
    prior = []
    for dec in traj.decisions:
        if int(dec.get("s", -1)) == window and int(dec.get("p", -1)) == seat:
            wire = dict(dec)
            if "hist" not in wire:
                wire["hist"] = store_wire_hist(prior, dec.get("_pos", len(prior)))
            ex, aux = feat.example(wire, traj.header, "priority")
            ex["_candidate_keys"] = aux.get("candidate_keys", [])
            ex["_candidate_key_aliases"] = aux.get("candidate_key_aliases", [])
            ex_fv = feat.example(wire, traj.header, "priority", full_vis=True)[0] if full_vis else None
            return ex, ex_fv
        prior.append(dec)
    return None


def _source_matches(source: str, path: Path) -> bool:
    """Accept either the miner's stable basename or its original path."""
    source = str(source)
    return source in {str(path), path.name} or Path(source).name == path.name


def candidate_inverse_frequency_weights(rows: list[dict]) -> list[float]:
    """Return mean-one inverse-frequency weights for candidate identities."""
    frequencies = Counter((int(row["entity"]), row["sa"]) for row in rows)
    if not rows:
        return []
    norm = len(rows) / max(len(frequencies), 1)
    return [
        norm / frequencies[(int(row["entity"]), row["sa"])]
        for row in rows
    ]


def build_candidate_batch(
    label_paths: list[str],
    store_paths: list[str],
    stem: str,
    methods: list[str],
    seg: int = 256,
    clip: float = 0.25,
    full_vis: bool = False,
) -> dict | None:
    """Join labels to natural windows and return collated segments.

    Fork stores are preferred when present.  Source-store joins are the
    labels-only fallback and intentionally do not require behavior-policy
    ``mu`` records, so forced trajectories can never enter ordinary V-trace.
    """
    from anvil.bridge.featurize import Featurizer
    from anvil.store.trajectories import open_store
    from anvil.training.dataset import collate

    rows = load_rows(label_paths, clip=clip)
    if not rows:
        return None
    by_window: dict[tuple, list[dict]] = {}
    for row in rows:
        # Several candidate arms intentionally share one source window.
        by_window.setdefault(row["key"], []).append(row)
    feat = Featurizer(stem, methods)
    picked: dict[tuple, tuple] = {}
    for raw_path in store_paths:
        path = Path(raw_path)
        store = open_store(path)
        source_default = path.name
        for g in store.game_indices():
            try:
                header = store.game(g).header
            except Exception:
                continue
            fk = header.get("fork") or {}
            source = str(fk.get("source", fk.get("store", source_default)))
            parent = int(fk.get("pg", fk.get("game", -1)))
            window = int(fk.get("window", fk.get("windowId", fk.get("fp", -1))))
            # Candidate stores from the first implementation carry source/g/w;
            # old fork stores have only pg/fp and are still accepted when the
            # label source is the path basename.
            candidates = by_window.get((source, parent, window), [])
            if not candidates and source == source_default:
                candidates = by_window.get((source_default, parent, window), [])
            for row in candidates:
                key = (row["key"], row["entity"], row["sa"])
                if key in picked:
                    continue
                got = _first_window(store, feat, g, row["seat"], row["window"], full_vis=full_vis)
                if got is not None:
                    picked[key] = (*got, row)

        # Candidate labels have one row per candidate arm, while all arms at
        # a window share the same natural decision.  A labels-only campaign
        # need not create a second behavior-policy store, so join directly to
        # the source store when no explicit fork header matched above.
        for row in rows:
            if not _source_matches(row["source"], path):
                continue
            key = (row["key"], row["entity"], row["sa"])
            if key in picked:
                continue
            got = _source_window(store, feat, row["game"], row["seat"], row["window"], full_vis)
            if got is not None:
                picked[key] = (*got, row)

    plain: list[dict] = []
    plain_fv: list[dict | None] = []
    joined: list[tuple[dict, dict | None, dict, int]] = []
    misses = 0
    for ex, ex_fv, row in picked.values():
        from anvil.training.dataset import norm_sa

        target = (int(row["entity"]), norm_sa(row["sa"]))
        aliases = ex.get("_candidate_key_aliases")
        if aliases:
            hits = [j for j, keys in enumerate(aliases) if j > 0 and target in keys]
        else:
            keys = list(ex.get("_candidate_keys") or [])
            hits = [j for j, key in enumerate(keys) if j > 0 and key == target]
        if len(hits) != 1:
            misses += 1
            continue
        joined.append((ex, ex_fv, row, hits[0]))

    # Equalize candidate arms by identity so a common action cannot dominate
    # the auxiliary phase merely because it appears in more source windows.
    weights = candidate_inverse_frequency_weights([row for _ex, _fv, row, _j in joined])
    meta: list[tuple[float, int, float, float, float]] = []
    for (ex, ex_fv, row, target_idx), weight in zip(joined, weights, strict=True):
        plain.append(ex)
        plain_fv.append(ex_fv)
        meta.append(
            (
                row["adv"],
                target_idx,
                row["natural_wr"],
                row["realized"] / max(row["forced_total"], 1),
                weight,
            )
        )

    if not plain:
        return None

    def collated(examples, metas):
        out = []
        for i in range(0, len(examples), seg):
            batch = collate(examples[i : i + seg])
            ms = metas[i : i + seg]
            tm = torch.zeros(len(ms), batch["cand_mask"].shape[1], dtype=torch.bool)
            for b, (_, j, _wr, _rr, _weight) in enumerate(ms):
                if j >= tm.shape[1]:
                    raise ValueError("candidate target index exceeds collated candidate width")
                tm[b, j] = True
            batch["candidate_adv"] = torch.tensor([m[0] for m in ms], dtype=torch.float32)
            batch["candidate_tmask"] = tm
            batch["candidate_wr_nat"] = torch.tensor([m[2] for m in ms], dtype=torch.float32)
            batch["candidate_realization"] = torch.tensor([m[3] for m in ms], dtype=torch.float32)
            batch["candidate_weight"] = torch.tensor([m[4] for m in ms], dtype=torch.float32)
            out.append(batch)
        return out

    result = {
        "segs": collated(plain, meta),
        "n": len(plain),
        "n_labels": len(rows),
        "n_joined": len(picked),
        "n_miss_target": misses,
        "positive_adv": sum(m[0] > 0 for m in meta),
        "negative_adv": sum(m[0] < 0 for m in meta),
        "mean_abs_adv": sum(abs(m[0]) for m in meta) / len(meta),
        "realization_rate": sum(m[3] for m in meta) / len(meta),
        "label_stats": getattr(load_rows, "last_stats", {}),
    }
    if full_vis:
        result["segs_fv"] = collated(plain_fv, meta)
    return result
