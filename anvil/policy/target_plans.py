"""Additive schema-v3 legal target-plan parsing.

Forge emits plans per wire option.  The policy still deduplicates options by
(entity row, normalized SA), so this module keeps both identities: canonical
candidate indices for masking and exact wire indices/concrete refs for the
CastPlan response.
"""

from __future__ import annotations

from typing import Any

import torch

from anvil.encoder.transform import player_target_position


def target_plan_fields(
    opts: list[dict],
    wire_to_candidate: list[int],
    n_cands: int,
    row_of: dict[int, int],
    perspective: int,
    n_players: int,
    t_max: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Convert concrete per-wire plans to item-canonical target classes.

    Tokens are stored as kind/index until collate knows the padded entity
    width.  Every sequence includes STOP.  A wire advertised as complete is
    downgraded to fallback if one of its references cannot join the visible
    observation or exceeds the model's target capacity.
    """

    plans: list[dict[str, Any]] = []
    wire_complete = [False] * (len(opts) + 1)  # one-based, 0 = PASS
    if not any("tc" in opt for opt in opts):
        return {}, {"target_plans": plans, "target_wire_complete": wire_complete}
    wires_by_cand: list[list[int]] = [[] for _ in range(n_cands)]

    for oi, opt in enumerate(opts):
        wire = oi + 1
        cand = wire_to_candidate[wire] if wire < len(wire_to_candidate) else -1
        if cand <= 0 or cand >= n_cands:
            continue
        wires_by_cand[cand].append(wire)
        if int(opt.get("tc", 0)) != 1:
            continue
        parsed: list[dict[str, Any]] = []
        ok = True
        for raw in opt.get("tp") or []:
            refs = raw.get("r") or []
            if len(refs) > t_max:
                ok = False
                break
            kinds: list[int] = []
            idxs: list[int] = []
            concrete: list[dict[str, int]] = []
            for ref in refs:
                if "p" in ref:
                    pi = int(ref["p"])
                    if not 0 <= pi < n_players:
                        ok = False
                        break
                    kinds.append(1)
                    idxs.append(player_target_position(pi, perspective, n_players))
                    concrete.append({"p": pi})
                elif "e" in ref and int(ref["e"]) in row_of:
                    eid = int(ref["e"])
                    kinds.append(0)
                    idxs.append(row_of[eid])
                    out_ref = {"e": eid}
                    if int(ref.get("ns", 0)) == 1:
                        out_ref["ns"] = 1
                    concrete.append(out_ref)
                else:
                    ok = False
                    break
            if not ok:
                break
            kinds.append(2)
            idxs.append(0)
            parsed.append(
                {
                    "candidate": cand,
                    "wire": wire,
                    "kinds": kinds,
                    "idxs": idxs,
                    "xmask": int(raw.get("xm", 0)),
                    "refs": concrete,
                }
            )
        if ok:
            wire_complete[wire] = True
            plans.extend(parsed)

    # A canonical candidate is authoritative only when every wire alias that
    # collapsed into it is authoritative.  Otherwise an omitted alias could
    # carry legal actions that a union mask would incorrectly remove.
    enforce = [False] * n_cands
    allow = [True] * n_cands
    for cand in range(1, n_cands):
        wires = wires_by_cand[cand]
        enforce[cand] = bool(wires) and all(wire_complete[w] for w in wires)
        if enforce[cand]:
            allow[cand] = any(p["candidate"] == cand for p in plans)

    width = t_max + 1
    fields = {
        "tp_cand": torch.tensor([p["candidate"] for p in plans], dtype=torch.int64),
        "tp_wire": torch.tensor([p["wire"] for p in plans], dtype=torch.int64),
        "tp_kind": torch.tensor(
            [p["kinds"] + [-1] * (width - len(p["kinds"])) for p in plans],
            dtype=torch.int64,
        ).reshape(-1, width),
        "tp_idx": torch.tensor(
            [p["idxs"] + [-1] * (width - len(p["idxs"])) for p in plans],
            dtype=torch.int64,
        ).reshape(-1, width),
        "tp_xmask": torch.tensor([p["xmask"] for p in plans], dtype=torch.int64),
        "tp_enforce": torch.tensor(enforce, dtype=torch.bool),
        "tp_cand_allow": torch.tensor(allow, dtype=torch.bool),
        "tp_forced_wire": torch.tensor(-1, dtype=torch.int64),
    }
    aux = {"target_plans": plans, "target_wire_complete": wire_complete}
    return fields, aux
