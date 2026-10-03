"""Seat-specific model routing for model-vs-model evaluation."""

from collections import Counter
from types import SimpleNamespace

from anvil.bridge.pb import anvil_bridge_pb2 as pb
from anvil.bridge.server import DecisionServicer


def _backend(name: str):
    return SimpleNamespace(name=name, counts=Counter())


def test_model_match_routes_by_observation_perspective():
    seat0 = _backend("seat0")
    seat1 = _backend("seat1")
    servicer = DecisionServicer(
        "model",
        [],
        backend=seat0,
        seat_backends={0: seat0, 1: seat1},
    )

    req0 = pb.DecisionRequest(observation=b'{"p":0}')
    req1 = pb.DecisionRequest(observation=b'{"p":1}')

    assert servicer._backend_for(req0) is seat0
    assert servicer._backend_for(req1) is seat1


def test_model_match_keeps_primary_backend_for_missing_observation():
    seat0 = _backend("seat0")
    seat1 = _backend("seat1")
    servicer = DecisionServicer(
        "model",
        [],
        backend=seat0,
        seat_backends={0: seat0, 1: seat1},
    )

    assert servicer._backend_for(pb.DecisionRequest()) is seat0
    assert servicer._backend_for(pb.DecisionRequest(observation=b"not-json")) is seat0


def test_candidate_window_routes_to_drill_backend():
    class Backend:
        supports_forced_priority = True

        def __init__(self, name):
            self.name = name
            self.calls = 0
            self.counts = Counter()

        def answer(self, req, _header, _seed, greedy=False):
            self.calls += 1
            return pb.DecisionResponse(decision_seq=req.decision_seq)

    primary = Backend("primary")
    drill = Backend("drill")
    servicer = DecisionServicer(
        "model",
        [],
        backend=primary,
        drill_backend=drill,
    )

    messages = [
        pb.WorkerMsg(hello=pb.WorkerHello(worker_id="worker")),
        pb.WorkerMsg(
            game_start=pb.GameStart(
                game_id="g17.w4.r0.n",
                seed=17,
            )
        ),
        pb.WorkerMsg(
            request=pb.DecisionRequest(
                decision_seq=9,
                decision_tag="mtg.priority",
            )
        ),
    ]

    responses = list(servicer.Session(iter(messages), None))

    assert drill.calls == 1
    assert primary.calls == 0
    assert responses[-1].response.decision_seq == 9
