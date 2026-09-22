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
