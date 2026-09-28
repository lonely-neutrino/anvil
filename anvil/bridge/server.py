"""Anvil decision server (M0 echo/random + M1 model): answers DecisionBridge sessions.

Modes:
- echo   -- echo the worker's pre-drawn answer back (bridge-tax instrument:
            gRPC-arm games are bit-identical to local-arm games, so the
            throughput delta isolates serialization + transport).
- random -- answer uniformly at random server-side, seeded per game from
            GameStart.seed (deterministic per seed; the M1-shaped mode).
- model  -- M1 D8: featurize the wire observation (same code path as the
            training loader), run AnvilNet.act, answer CastPlans + one-field
            tags. --ckpt required; --pass-delta is the calibration arm knob
            (pass_calibration.json "delta"). mtg.mulligan_tuck stays
            heuristic-fallback at D8 (SELECT_K answer mapping deferred).

Run: uv run python -m anvil.bridge.server [--port 50051] [--mode echo]
     [--tags mtg.priority,mtg.mulligan_keep,...]
     [--ckpt data/training/d7-ep3/last.pt --pass-delta 0.0]
     [--ckpt-seat1 data/training/other/last.pt]

With --ckpt-seat1, --ckpt serves registered seat 0 and the second checkpoint
serves registered seat 1. Routing uses the observation's perspective field.
This is an argmax evaluation mode; sampled dual-policy serving is intentionally
not enabled because a single --mu-out file cannot safely represent two policies.

One bidirectional stream per worker; one outstanding request per stream by
construction (the worker's game thread blocks), so the servicer is a plain
loop. Model inference is batch-1 behind a lock at first light — micro-batching
across streams is the known lever if the w=16 arms want it. Stats print on
Ctrl-C.
"""

from __future__ import annotations

import argparse
import json
import random
import signal
import threading
import time
from collections import Counter
from concurrent import futures
from pathlib import Path

import grpc

from anvil.bridge.pb import anvil_bridge_pb2 as pb
from anvil.bridge.pb import anvil_bridge_pb2_grpc as pb_grpc

PROTOCOL_VERSION = 0
DEFAULT_TAGS = "mtg.priority,mtg.mulligan_keep,mtg.mulligan_tuck,mtg.trigger,mtg.binary,mtg.number,mtg.choose_color"
MODEL_TAGS = "mtg.priority,mtg.mulligan_keep,mtg.trigger,mtg.binary,mtg.number,mtg.choose_color"
# advertised only when the checkpoint carries TRAINED combat heads —
# load_compat fresh-inits them for pre-D5 checkpoints, which must never serve
COMBAT_TAGS = "mtg.attack,mtg.block"
# advertised only when the checkpoint carries the pay_ params (M9 rung 3) —
# pre-M9 checkpoints must decline so the worker's echo answers AUTO
PAY_TAGS = "mtg.pay_mana_class"


def is_drill_game_id(game_id: str) -> bool:
    """Return whether a wire-only rollout should use the drill policy.

    ``.f`` is the legacy forced-branch marker. Candidate drills use ``.w``
    because their stable identity is keyed by source window rather than the
    transient fork counter. Both are labels-only wire sessions and must be
    routed to ``--drill-ckpt``; otherwise candidate arms accidentally run the
    pinned source checkpoint.
    """
    return ".f" in game_id or ".w" in game_id


class _Batcher:
    """GPU micro-batching (D6 groundwork): worker streams featurize in
    parallel and submit examples here; one thread drains up to max_batch
    items inside window_ms, collates, runs a single act(), and hands each
    caller a per-item view (batch dim kept, so answer translation indexes
    [0] unchanged). Measured motivation: batch-1 tops out at 59 rps — below
    the ~81 rps both-seats-bridged self-play needs at w=8. pass_delta rides
    per item as a (B,1) tensor (mixed priority/other batches). The batcher
    thread is the sole GPU user; the old per-request lock is gone."""

    def __init__(
        self,
        net,
        torch_mod,
        device: str,
        counts: Counter,
        max_batch: int = 16,
        window_ms: float = 3.0,
        temperature: float = 1.0,
    ):
        import queue

        self.net = net
        self.torch = torch_mod
        self.device = device
        self.counts = counts
        self.max_batch = max_batch
        self.window_ms = window_ms
        self.temperature = temperature
        self.q: "queue.Queue[dict]" = queue.Queue()
        self._queue_mod = queue
        # M10 v2: run the greedy emission decode in act() (set by the
        # backend when the ckpt carries sched params; per-window use is the
        # SchedServe's decision — the decode itself is cheap pointer steps)
        self.sched_decode = False
        threading.Thread(target=self._loop, daemon=True, name="gpu-batcher").start()

    def submit(
        self,
        ex: dict,
        pass_delta: float,
        noise: "dict | None" = None,
        forced_choice: "int | None" = None,
    ) -> dict:
        slot = {
            "ex": ex,
            "pd": pass_delta,
            "nz": noise,
            "forced": -1 if forced_choice is None else int(forced_choice),
            "ev": threading.Event(),
        }
        self.q.put(slot)
        slot["ev"].wait()
        if "err" in slot:
            raise slot["err"]
        return slot["out"]

    def _loop(self) -> None:
        from anvil.policy.sampling import pad_noise
        from anvil.training.dataset import collate

        queue = self._queue_mod
        while True:
            slots = [self.q.get()]
            deadline = time.monotonic() + self.window_ms / 1000
            while len(slots) < self.max_batch:
                t = deadline - time.monotonic()
                if t <= 0:
                    break
                try:
                    slots.append(self.q.get(timeout=t))
                except queue.Empty:
                    break
            self.counts[f"gpu_batch_{min(len(slots), 16)}"] += 1
            try:
                batch = {k: v.to(self.device) for k, v in collate([s["ex"] for s in slots]).items()}
                pd = self.torch.tensor(
                    [[s["pd"]] for s in slots], device=self.device, dtype=self.torch.float32
                )
                forced = self.torch.tensor(
                    [s["forced"] for s in slots], device=self.device, dtype=self.torch.long
                )
                # sampling is server-wide: slots carry noise all-or-none
                nz = (
                    pad_noise([s["nz"] for s in slots], batch, self.device)
                    if slots[0]["nz"] is not None
                    else None
                )
                with self.torch.autocast(self.device, dtype=self.torch.bfloat16):
                    out = self.net.act(
                        batch,
                        pass_delta=pd,
                        noise=nz,
                        temperature=self.temperature,
                        sched_decode=self.sched_decode,
                        forced_choice=forced,
                    )
                for i, s in enumerate(slots):
                    # per-item views keep the batch dim; scalars are shared
                    # (n_ent/stop_idx are batch-padded dims by construction)
                    s["out"] = {
                        k: (v[i : i + 1] if self.torch.is_tensor(v) else v) for k, v in out.items()
                    }
            except Exception as e:
                for s in slots:
                    s["err"] = e
            finally:
                for s in slots:
                    s["ev"].set()


class ModelBackend:
    """Loads a D7 checkpoint and answers decisions. Import of torch/model
    machinery is deferred to here so echo/random sessions stay lightweight."""

    def __init__(
        self,
        ckpt_path: str,
        pass_delta: float,
        device: str = "cuda:0",
        sample: bool = False,
        temperature: float = 1.0,
        mu_path: "str | None" = None,
        instrument: bool = False,
        max_batch: int = 16,
        batch_window_ms: float = 3.0,
    ):
        import torch

        from anvil.bridge.featurize import Featurizer
        from anvil.encoder.transform import require_player_target_convention
        from anvil.training.dataset import default_methods
        from anvil.training.train import build_net

        self.torch = torch
        # Deserialize on CPU first. This avoids ROCm failures while PyTorch
        # restores checkpoint storages directly onto the accelerator; the
        # network is moved to `device` below before the weights are loaded.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        require_player_target_convention(cfg, f"checkpoint {ckpt_path}")
        # sa_vocab_size absent = pre-D2 host-level checkpoint: the model has
        # no SA descriptor and answers host_level=True (Java runs the full
        # disambiguation ladder). D2+ checkpoints name the SA themselves.
        self.n_sa = cfg.get("sa_vocab_size", 0)
        # Candidate-specific intervention is only meaningful for the v1
        # SA-level policy.  Host-only checkpoints cannot identify a requested
        # (entity, SA) arm and must advertise a hard capability miss.
        self.supports_forced_priority = bool(self.n_sa)
        # trained combat heads present? (D5 checkpoints; pre-D5 ones get
        # fresh-init heads from load_compat and must not serve combat tags)
        self.has_combat = any(k.startswith(("atk_", "blk_", "cmb_")) for k in ckpt["model"])
        # pay_ params present? (M9 rung 3; same never-serve-fresh-init rule —
        # except pay_bias's +2.0 init is BY DESIGN safe, so the gate is about
        # the untrained pointer keys, not the bias)
        self.has_pay = any(k.startswith("pay_") for k in ckpt["model"])
        # plan-carry params present? (M9 D6; the graft ckpt saves them even at
        # zero-init, so the carry activates exactly when the ckpt was built
        # for it — the has_pay/never-serve-fresh-init convention)
        self.carry_plan = any(
            k.startswith(("plan_", "assemble.plan_proj")) for k in ckpt["model"]
        )
        # M10 v2 schedule-carry params present? (m10-build-spec §3; the same
        # graft-saves-them-at-init convention as carry_plan)
        self.carry_sched = any(
            k.startswith(("sched_", "assemble.sched_")) for k in ckpt["model"]
        )
        self.net = build_net(
            cfg["embed"], cfg["pool_manifest"], len(default_methods()), n_sa=self.n_sa
        ).to(device)
        self.net.load_compat(ckpt["model"])
        self.net.eval()
        self.feat = Featurizer(cfg["embed"], default_methods())
        if self.n_sa and self.n_sa != len(self.feat.sa_vocab):
            raise ValueError(
                f"checkpoint sa_vocab_size {self.n_sa} != pinned sa_vocab "
                f"{len(self.feat.sa_vocab)} — serve/train vocab skew"
            )
        self.pass_delta = pass_delta
        self.device = device
        self.counts: Counter[str] = Counter()
        self.batcher = _Batcher(
            self.net,
            torch,
            device,
            self.counts,
            max_batch=max_batch,
            window_ms=batch_window_ms,
            temperature=temperature,
        )
        # sampling mode (M2 D6): Gumbel-max instead of argmax, behavior-policy
        # record per answered decision -> mu.jsonl, joined at ingest on (g, s)
        self.sample = sample
        self.temperature = temperature
        # M7 forced-branch instrument mode: wire-only fork sessions (g=-1)
        # may be SAMPLED without mu records — forced-branch completions are
        # measurement, never training data (m7-plan D2 pin 3). Off, the
        # standing guard below raises as before.
        self.instrument = instrument
        # D6 plan carry (m9-d6-plan-latent-spec §3): {(g, seat): (turn, vec)}.
        # Emission = the first request of a (g, seat, turn) group (has_plan=0,
        # the emitted out["plan"] is cached); later same-turn requests feed it
        # back (has_plan=1). Wire-only fork headers (g<0) never carry — their
        # g collides across streams. Capped FIFO: finished games just age out.
        self.plan_carry: dict[tuple, tuple] = {}
        self.plan_lock = threading.Lock()
        self._plan_cap = 4096
        # M10 v2 discrete schedule carry (revise-on-trigger; sched_serve.py)
        self.sched_serve = None
        if self.carry_sched:
            from anvil.bridge.sched_serve import SchedServe

            self.sched_serve = SchedServe(self.feat)
            self.batcher.sched_decode = True
        self.mu_file = None
        self.mu_lock = threading.Lock()
        if sample:
            if not mu_path:
                raise ValueError("--sample requires --mu-out")
            self.mu_file = open(mu_path, "a", buffering=1)
            self.mu_file.write(
                json.dumps(
                    {
                        "k": "meta",
                        "ckpt": str(ckpt_path),
                        "step": ckpt.get("step"),
                        "pass_delta": pass_delta,
                        "temperature": temperature,
                    }
                )
                + "\n"
            )
        print(
            f"[server] model {ckpt_path} step={ckpt.get('step')} "
            f"pass_delta={pass_delta} device={device} "
            f"sample={sample} temperature={temperature} "
            f"micro-batch<= {self.batcher.max_batch} window {self.batcher.window_ms}ms"
        )

    def answer(
        self, req: pb.DecisionRequest, header: dict | None, game_seed: int | None = None
    ) -> pb.DecisionResponse | None:
        """None = decline (worker falls back, tagged). Any exception is the
        caller's to turn into a loud decline — silence would poison an arm."""
        from anvil.bridge.featurize import TAG_TASK

        task = TAG_TASK.get(req.decision_tag)
        if task is None or not req.observation or header is None:
            return None
        dec = json.loads(req.observation)
        if req.retry_of:
            # Re-ask after a realizer veto (d6-vtrace-loop §6b). Telemetry
            # only: the re-asked dec carries a fresh s and reduced opts, so
            # the mu record and answer path need nothing special.
            self.counts["reask"] += 1
        ex, aux = self.feat.example(dec, header, task)
        forced_choice = None
        forced_wire_option = None
        forced_request = bool(req.force_option)
        if forced_request:
            if task != "priority":
                self.counts["force_nonpriority"] += 1
                return None
            if not self.supports_forced_priority:
                self.counts["force_unsupported_host_checkpoint"] += 1
                return None
            # CastPlan and DecisionRequest use the engine's one-based option
            # convention: 0 is PASS, option 1 names req.options[0].
            wire = int(req.forced_option)
            if wire <= 0 or wire > len(req.options):
                self.counts["force_invalid_index"] += 1
                return None
            wire_to_candidate = aux.get("wire_to_candidate") or []
            pos = wire  # aux[0] is PASS, aux[1] is req.options[0]
            if pos >= len(wire_to_candidate):
                self.counts["force_invalid_index"] += 1
                return None
            forced_choice = int(wire_to_candidate[pos])
            if forced_choice <= 0:
                self.counts["force_unmapped"] += 1
                return None
            # Keep the requested wire option separate from the canonical
            # model choice. Several identical visible entities can share one
            # dedup row; returning cand_first_opt here would silently rewrite
            # a request for the second copy into the first copy.
            forced_wire_option = wire
            self.counts["force_requested"] += 1
        plan_key, plan_emit = self._plan_inject(ex, header, dec)
        sched_ctx = None
        if self.sched_serve is not None and header.get("g", -1) >= 0:
            if req.retry_of:
                dec["retry_of"] = True  # trigger-1 detector (m10-build-spec §3)
            sched_ctx = self.sched_serve.inject(ex, aux, dec, header, task)
        delta = self.pass_delta if task == "priority" else 0.0
        if req.forbid_decline and task == "priority":
            # M7 forced-branch act ask: mask the pass logit so the sampled/
            # argmax pick must be a cast. -1e9 dominates any real logit in
            # both modes; the calibration delta is irrelevant under the mask.
            delta -= 1e9
            self.counts["forbid_decline"] += 1
        wire_fork = header.get("g", -1) < 0
        noise = None
        if self.sample:
            if wire_fork and not self.instrument:
                # A wire-only fork header (g=-1): every completion would share
                # (g, s) mu keys AND the parent's noise seed. Sampled drill
                # serving requires -forkobs (synthetic unique g, per-completion
                # announced seed) — or instrument mode, where the worker
                # announces per-completion seeds and mu is skipped below.
                # Raising -> loud decline -> heuristic fallback with NO mu
                # record, so nothing poisoned can train.
                raise ValueError(
                    "sampled serving needs a store-indexed header "
                    "(fork sessions require -forkobs or "
                    "--fork-instrument)"
                )
            from anvil.policy.sampling import make_noise, noise_seed

            noise = make_noise(
                ex, task, self.temperature, seed=noise_seed(game_seed or 0, dec["s"])
            )
        out = self.batcher.submit(ex, delta, noise, forced_choice=forced_choice)
        if plan_emit and plan_key is not None:
            self._plan_store(plan_key, dec.get("t", 0), out["plan"][0].float().cpu())
        sched_row = None
        if sched_ctx is not None:
            sched_row = self.sched_serve.after(sched_ctx, out, aux, dec)
        if self.sample and not wire_fork and not forced_request:
            self._write_mu(header["g"], dec, task, ex, aux, out, sched=sched_row)
        resp = pb.DecisionResponse(decision_seq=req.decision_seq)
        if task == "priority":
            resp.construct.cast_plan.CopyFrom(
                self._castplan(out, aux, forced_option=forced_wire_option)
            )
        elif task == "attack":
            resp.construct.attack_map.CopyFrom(self._attackmap(out, aux))
        elif task == "block":
            resp.construct.block_map.CopyFrom(self._blockmap(out, aux))
        elif task == "pay_class":
            # SELECT_ONE over {auto} ∪ goal options: choice 0 = auto = wire
            # index 0; goal candidates are positional (cand_first_opt)
            c = int(out["choice"][0])
            resp.index = 0 if c == 0 else aux["cand_first_opt"][c]
        elif task in ("mull_keep", "trigger", "binary"):
            resp.flag = bool(out["bool"][0])
        elif task == "number":
            n = int(out["num"][0])
            if req.shape == pb.SELECT_ONE:
                # list-variant chooseNumber: labels are the values; nearest wins
                try:
                    vals = [int(o.label) for o in req.options]
                except ValueError:
                    return None
                resp.index = min(range(len(vals)), key=lambda i: abs(vals[i] - n))
            else:
                c = req.constraints
                v = max(int(c.min), min(n, int(c.max))) if c.max > c.min else int(c.min)
                if v != n:
                    self.counts["num_clamped"] += 1
                resp.value = v
        elif task == "choose_color":
            # The model predicts a canonical WUBRG class; the worker expects
            # the index in the callback's legal-option list.  Reject any
            # malformed observation or mismatched request so GrpcBridge uses
            # its deterministic local echo instead of returning an illegal
            # color.
            c = int(out["color"][0])
            first = aux.get("color_first_opt", [])
            if not aux.get("color_options_valid") or not (0 <= c < len(first)):
                self.counts["color_invalid"] += 1
                return None
            option = first[c]
            if option < 0 or option >= len(req.options):
                self.counts["color_invalid"] += 1
                return None
            from anvil.training.dataset import color_class

            label_class = color_class(req.options[option].label)
            if label_class is None:
                self.counts["color_invalid"] += 1
                return None
            if label_class != c:
                self.counts["color_invalid"] += 1
                return None
            resp.index = option
        return resp

    def _plan_inject(self, ex: dict, header: dict, dec: dict) -> tuple:
        """D6 carry lookup (m9-d6-plan-latent-spec §3): fills ex's
        plan_vec/has_plan and returns (key, emit). Emission = no cached
        vector for this (g, seat, turn); wire-fork headers (g<0) never
        carry (their g collides across streams)."""
        if not self.carry_plan or header.get("g", -1) < 0:
            return None, False
        key = (header["g"], dec.get("p", -1))
        turn = dec.get("t", 0)
        with self.plan_lock:
            st = self.plan_carry.get(key)
        if st is not None and st[0] == turn:
            ex["plan_vec"] = st[1]
            ex["has_plan"] = 1.0
            return key, False
        ex["plan_vec"] = self.torch.zeros(self.net.assemble.plan_tok.shape[-1])
        ex["has_plan"] = 0.0
        return key, True

    def _plan_store(self, key: tuple, turn: int, vec) -> None:
        with self.plan_lock:
            self.plan_carry[key] = (turn, vec)
            while len(self.plan_carry) > self._plan_cap:
                self.plan_carry.pop(next(iter(self.plan_carry)))

    def _write_mu(
        self, g: int, dec: dict, task: str, ex: dict, aux: dict, out: dict,
        sched: "dict | None" = None,
    ) -> None:
        """One behavior-policy record (M2 D6) -> mu.jsonl, joined at ingest
        on (g, s). Record construction lives in sampling.mu_record. The M10
        `sched` field is the discrete carry's verbatim serialization
        (m10-build-spec §1) — the loader reconstructs, never derives."""
        from anvil.policy.sampling import mu_record

        rec = mu_record(g, dec["s"], task, ex, aux, out)
        if sched is not None:
            rec["sched"] = sched
        with self.mu_lock:
            self.mu_file.write(json.dumps(rec) + "\n")

    def _castplan(
        self, out: dict, aux: dict, forced_option: int | None = None
    ) -> pb.CastPlan:
        cp = pb.CastPlan()
        choice = int(out["choice"][0])
        if choice == 0:
            if forced_option is not None:
                raise ValueError("forced candidate inference returned PASS")
            self.counts["pass"] += 1
            return cp  # spell_option 0 = pass (label-space convention)
        if forced_option is None:
            cp.spell_option = aux["cand_first_opt"][choice] + 1
        else:
            cp.spell_option = int(forced_option)
        # SA-level model (D2+): the option index IS the chosen SA — the Java
        # ladder skips its kind/order rungs (shape->pay only). Host-level
        # checkpoints keep the full ladder.
        cp.host_level = self.n_sa == 0
        n_ent, stop = int(out["n_ent"]), int(out["stop_idx"])
        for t in range(out["tgt_picks"].shape[1]):
            pick = int(out["tgt_picks"][0, t])
            if pick == stop:
                break
            ref = cp.target_refs.add()
            if pick < n_ent:
                # dedup-group row -> deterministic representative (lowest id)
                eid = aux["row_min_id"].get(pick, -1)
                ref.entity = eid
                if eid in aux["stack_ids"]:
                    ref.ns = 1
            else:
                player_pos = pick - n_ent
                seats = aux.get("seats")
                if not isinstance(seats, list) or not 0 <= player_pos < len(seats):
                    raise ValueError(f"invalid self-first player target position {player_pos}")
                ref.player = seats[player_pos]
        x = int(out["x_cls"][0])
        cp.has_x = True
        # class 17 = ">16" overflow bucket; clamp + count (decision 2026-07-10)
        cp.x_value = min(x, 16)
        if x >= 17:
            self.counts["x_overflow_clamped"] += 1
        self.counts["cast"] += 1
        return cp

    def _attackmap(self, out: dict, aux: dict) -> "pb.AttackMap":
        """Per-row picks -> entity-ref assignments. Dedup rows expand to the
        count head's k first-fit members; player positions (self-first, the
        combat-head convention) map back to registered indices via seats."""
        am = pb.AttackMap()
        n_ent = int(out["n_ent"])
        for i, row in enumerate(aux["cmb_rows"]):
            if not bool(out["atk_yes"][0, i]):
                continue
            ids = aux["cmb_members"][row]
            k = 1 if len(ids) == 1 else min(int(out["cmb_count"][0, i]), len(ids))
            tgt = int(out["atk_tgt"][0, i])
            for eid in ids[:k]:
                a = am.assignments.add()
                a.attacker.entity = eid
                if tgt < n_ent:
                    a.defender.entity = aux["row_min_id"].get(tgt, -1)
                else:
                    a.defender.player = aux["seats"][tgt - n_ent]
            self.counts["attack_rows"] += 1
        if not am.assignments:
            self.counts["attack_empty"] += 1
        return am

    def _blockmap(self, out: dict, aux: dict) -> "pb.BlockMap":
        """blk_pick slot M (the batch none column) = no block; otherwise the
        slot names an attacker row — first-fit member is the engine-side tie
        (multiset semantics, same as the labels). Group blocks expand to the
        count head's j members."""
        bm = pb.BlockMap()
        slots = aux["blk_atk_rows"]
        for i, row in enumerate(aux["cmb_rows"]):
            s = int(out["blk_pick"][0, i])
            if s >= len(slots):  # the none slot (index M) or a padded column
                continue
            ids = aux["cmb_members"][row]
            j = 1 if len(ids) == 1 else min(int(out["cmb_count"][0, i]), len(ids))
            # attacker rows are the opponent's — resolve via the all-rows map
            atk_id = aux["row_min_id"].get(slots[s], -1)
            for eid in ids[:j]:
                a = bm.assignments.add()
                a.blocker.entity = eid
                a.attacker.entity = atk_id
            self.counts["block_rows"] += 1
        return bm


class DecisionServicer(pb_grpc.DecisionBridgeServicer):
    def __init__(
        self,
        mode: str,
        bridged_tags: list[str],
        deadline_ms: int = 5000,
        backend: ModelBackend | None = None,
        drill_backend: ModelBackend | None = None,
        seat_backends: dict[int, ModelBackend] | None = None,
    ):
        self.mode = mode
        self.bridged_tags = bridged_tags
        self.deadline_ms = deadline_ms
        self.backend = backend
        # Dual-policy drill serving (M4 D2.4): fork wire sessions (wid
        # contains ".f" or ".w", e.g. "g42.f0r3" or
        # "g42.w4.r0.n") are answered by drill_backend while
        # the mainline replay stays on the pinned backend — per-checkpoint
        # drill evals need the replay policy frozen to reach the fork at all.
        self.drill_backend = drill_backend
        # Model-match mode: the bridge is shared by both registered seats, so
        # select the policy from the observation perspective. This is kept
        # separate from drill_backend: drill routing is keyed by fork session,
        # while model-match routing is keyed by seat.
        self.seat_backends = seat_backends
        self.drill_requests = 0
        self.requests_by_tag: Counter[str] = Counter()
        self.fallbacks: Counter[str] = Counter()
        self.games = 0
        self.t0 = time.monotonic()
        backends = [b for b in (backend, drill_backend) if b is not None]
        if seat_backends:
            backends.extend(seat_backends.values())
        self.forced_priority_option = bool(
            mode == "model"
            and backends
            and all(getattr(b, "supports_forced_priority", False) for b in backends)
        )

    def _backend_for(self, req: pb.DecisionRequest) -> ModelBackend | None:
        """Choose a seat-specific policy for a normal model request.

        The Java bridge carries the registered player's perspective in every
        model observation as ``p``. Missing or malformed observations retain
        the legacy primary-backend behavior; ModelBackend will then decline
        the request and the existing fallback accounting remains in charge.
        """
        if not self.seat_backends:
            return self.backend
        try:
            dec = json.loads(req.observation) if req.observation else None
            seat = int(dec.get("p", -1)) if isinstance(dec, dict) else -1
        except (TypeError, ValueError, json.JSONDecodeError):
            seat = -1
        return self.seat_backends.get(seat, self.backend)

    def _answer(self, req: pb.DecisionRequest, rng: random.Random) -> pb.DecisionResponse:
        if self.mode == "echo" and req.HasField("echo_answer"):
            resp = pb.DecisionResponse()
            resp.CopyFrom(req.echo_answer)
            resp.decision_seq = req.decision_seq
            return resp
        resp = pb.DecisionResponse(decision_seq=req.decision_seq)
        n = len(req.options)
        c = req.constraints
        if req.shape == pb.SELECT_ONE:
            resp.index = rng.randrange(n) if n > 1 else 0
        elif req.shape == pb.SELECT_K:
            k = min(c.k or c.min, n)
            resp.indices.indices.extend(sorted(rng.sample(range(n), int(k))))
        elif req.shape == pb.INT_IN_RANGE:
            resp.value = rng.randint(c.min, c.max) if c.max > c.min else c.min
        elif req.shape == pb.BOOL:
            resp.flag = rng.random() < 0.5
        elif req.shape == pb.ORDER_N:
            order = list(range(n))
            rng.shuffle(order)
            resp.ordering.indices.extend(order)
        else:
            resp.fallback = True  # CONSTRUCT not answered at M0
        return resp

    def Session(self, request_iterator, context):
        rng = random.Random(0)
        worker = "?"
        header: dict | None = None
        game_seed = 0
        use_drill = False
        for msg in request_iterator:
            kind = msg.WhichOneof("msg")
            if kind == "hello":
                worker = msg.hello.worker_id
                yield pb.ServerMsg(
                    hello=pb.ServerHello(
                        protocol_version=PROTOCOL_VERSION,
                        bridged_tags=self.bridged_tags,
                        default_deadline_ms=self.deadline_ms,
                        one_shot_cast=self.mode == "model",
                        forced_priority_option=self.forced_priority_option,
                    )
                )
            elif kind == "game_start":
                self.games += 1
                game_seed = msg.game_start.seed
                rng = random.Random(game_seed)
                # Requests belong to the stream's last-announced game; the
                # worker re-announces the mainline after each fork block.
                use_drill = self.drill_backend is not None and is_drill_game_id(
                    msg.game_start.game_id
                )
                header = None
                if msg.game_start.header:
                    try:
                        header = json.loads(msg.game_start.header)
                    except ValueError:
                        print(f"[server] worker={worker}: unparseable game header")
            elif kind == "request":
                self.requests_by_tag[msg.request.decision_tag] += 1
                if use_drill:
                    self.drill_requests += 1
                if self.mode == "model":
                    yield pb.ServerMsg(
                        response=self._model_answer(
                            msg.request,
                            header,
                            game_seed,
                            self.drill_backend
                            if use_drill
                            else self._backend_for(msg.request),
                        )
                    )
                else:
                    yield pb.ServerMsg(response=self._answer(msg.request, rng))
            elif kind == "game_end":
                pass  # worker-side logs are authoritative at M0
            elif kind == "ping":
                yield pb.ServerMsg(ping=msg.ping)
        print(f"[server] stream closed: worker={worker}")

    def _model_answer(
        self,
        req: pb.DecisionRequest,
        header: dict | None,
        game_seed: int = 0,
        backend: ModelBackend | None = None,
    ) -> pb.DecisionResponse:
        if backend is None:
            backend = self.backend
        try:
            resp = backend.answer(req, header, game_seed)
        except Exception as e:  # loud decline; a silent wrong answer poisons the arm
            print(f"[server] MODEL ERROR on {req.decision_tag} seq={req.decision_seq}: {e!r}")
            resp = None
        if resp is None:
            self.fallbacks[req.decision_tag] += 1
            return pb.DecisionResponse(decision_seq=req.decision_seq, fallback=True)
        return resp

    def stats(self) -> str:
        dt = time.monotonic() - self.t0
        total = sum(self.requests_by_tag.values())
        lines = [f"{self.games} games, {total} requests in {dt:.0f}s ({total / dt:.0f} rps)"]
        if self.drill_backend is not None:
            lines.append(f"  drill-policy requests: {self.drill_requests}")
        lines += [f"  {t}: {n}" for t, n in self.requests_by_tag.most_common()]
        if self.fallbacks:
            lines += [f"  FALLBACK {t}: {n}" for t, n in self.fallbacks.most_common()]
        if self.seat_backends:
            for seat, backend in sorted(self.seat_backends.items()):
                lines += [f"  model seat{seat} {k}: {n}" for k, n in backend.counts.most_common()]
        elif self.backend is not None:
            lines += [f"  model {k}: {n}" for k, n in self.backend.counts.most_common()]
        return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=50051)
    ap.add_argument("--mode", choices=["echo", "random", "model"], default="echo")
    ap.add_argument(
        "--tags",
        default=None,
        help=f"default: {DEFAULT_TAGS} (echo/random) or {MODEL_TAGS} (model)",
    )
    ap.add_argument("--ckpt", default="data/training/d7-ep3/last.pt")
    ap.add_argument(
        "--ckpt-seat1",
        default=None,
        help="model-match mode: checkpoint used for registered seat 1; "
        "--ckpt remains the checkpoint for seat 0",
    )
    ap.add_argument(
        "--pass-delta",
        type=float,
        default=0.0,
        help="PASS-logit offset (pass_calibration.json delta; arm knob)",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--sample",
        action="store_true",
        help="Gumbel-max sampling instead of argmax (D6 actors); "
        "writes behavior-policy records to --mu-out",
    )
    ap.add_argument(
        "--temperature", type=float, default=1.0, help="sampling temperature (with --sample)"
    )
    ap.add_argument(
        "--max-batch",
        type=int,
        default=16,
        help="maximum GPU micro-batch size (default: 16)",
    )
    ap.add_argument(
        "--batch-window-ms",
        type=float,
        default=3.0,
        help="GPU micro-batch collection window in milliseconds (default: 3)",
    )
    ap.add_argument(
        "--mu-out", default=None, help="behavior-policy mu.jsonl path (required with --sample)"
    )
    ap.add_argument(
        "--drill-ckpt",
        default=None,
        help="dual-policy drill serving (M4 D2.4): fork wire "
        "sessions (wid contains '.f' or '.w') are answered by this "
        "checkpoint; the mainline replay stays on --ckpt. "
        "Model mode only. Argmax unless --drill-sample.",
    )
    ap.add_argument(
        "--drill-sample",
        action="store_true",
        help="sample the drill backend (M4 D3 training "
        "generation: fork completions sampled with mu "
        "records at --temperature, mainline stays argmax); "
        "requires --drill-ckpt and --drill-mu-out, and the "
        "run must use -forkobs",
    )
    ap.add_argument(
        "--drill-mu-out",
        default=None,
        help="drill backend's behavior-policy mu.jsonl (required with --drill-sample)",
    )
    ap.add_argument(
        "--fork-instrument",
        action="store_true",
        help="M7 forced-branch instrument mode: sampled serving "
        "for wire-only fork sessions (no -forkobs) with NO "
        "mu records — the worker announces per-completion "
        "seeds; forced-branch completions never train",
    )
    args = ap.parse_args()

    if args.ckpt_seat1 and args.mode != "model":
        ap.error("--ckpt-seat1 requires --mode model")
    if args.ckpt_seat1 and (args.sample or args.drill_sample):
        ap.error("--ckpt-seat1 is argmax-only; omit --sample/--drill-sample")

    backend = None
    drill_backend = None
    seat_backends = None
    if args.mode == "model":
        backend = ModelBackend(
            args.ckpt,
            args.pass_delta,
            args.device,
            sample=args.sample,
            temperature=args.temperature,
            mu_path=args.mu_out,
            instrument=args.fork_instrument,
            max_batch=args.max_batch,
            batch_window_ms=args.batch_window_ms,
        )
        if args.ckpt_seat1:
            seat_backends = {0: backend}
            seat_backends[1] = ModelBackend(
                args.ckpt_seat1,
                args.pass_delta,
                args.device,
                max_batch=args.max_batch,
                batch_window_ms=args.batch_window_ms,
                instrument=args.fork_instrument,
            )
        if args.drill_ckpt:
            drill_backend = ModelBackend(
                args.drill_ckpt,
                args.pass_delta,
                args.device,
                sample=args.drill_sample,
                temperature=args.temperature,
                mu_path=args.drill_mu_out,
                instrument=args.fork_instrument,
                max_batch=args.max_batch,
                batch_window_ms=args.batch_window_ms,
            )
    model_backends = list(seat_backends.values()) if seat_backends else [backend]
    tags = (
        args.tags
        if args.tags is not None
        else (
            (
                MODEL_TAGS
                # The hello advertises one global tag set. In a dual-policy
                # match, advertise optional heads only when every seat's
                # checkpoint has them; otherwise one seat could receive a
                # freshly initialized head rather than a trained policy.
                + ("," + COMBAT_TAGS if all(b.has_combat for b in model_backends) else "")
                + ("," + PAY_TAGS if all(b.has_pay for b in model_backends) else "")
            )
            if args.mode == "model"
            else DEFAULT_TAGS
        )
    )
    servicer = DecisionServicer(
        args.mode,
        tags.split(","),
        backend=backend,
        drill_backend=drill_backend,
        seat_backends=seat_backends,
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=32))
    pb_grpc.add_DecisionBridgeServicer_to_server(servicer, server)
    server.add_insecure_port(f"127.0.0.1:{args.port}")
    server.start()
    print(f"[server] mode={args.mode} port={args.port} tags={servicer.bridged_tags}")

    # SIGTERM behaves like SIGINT (background servers ignore SIGINT under
    # non-interactive shells — the standing lesson): both converge on the
    # stats + counts-dump path.
    def _term(_sig, _frm):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _term)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n[server] " + servicer.stats())
        # M10 v2: machine-readable counts beside the mu file — the driver's
        # telemetry pickup (m10-build-spec §5 families 2/3 ride the
        # SchedServe counters; log parsing is not an interface)
        if backend is not None and args.mu_out:
            dump = dict(backend.counts)
            if backend.sched_serve is not None:
                dump.update(backend.sched_serve.counts)
            Path(str(args.mu_out) + ".counts.json").write_text(
                json.dumps(dump, indent=1) + "\n"
            )
        server.stop(grace=1)


if __name__ == "__main__":
    main()
