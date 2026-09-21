import pytest

from decide import AsyncClient, Choice, Client, Gate, Noul
from decide.errors import AllBackendsFailed, BackendError, ConfigError
from tests.conftest import FakeBackend

Q = {"team": Choice("Which team?", {"billing": None, "eng": None}), "refund": Noul("Refund?")}


def test_single_backend_ok_route():
    c = Client([FakeBackend("a")])
    r = c.decide("charged twice", Q)
    assert r.choices["team"].choice == "billing"
    assert r.meta.route == ["a:ok"] and r.meta.backend == "a"


def test_error_falls_through_to_next():
    bad = FakeBackend("a", fail=BackendError("a", "boom"))
    c = Client([bad, FakeBackend("b")])
    r = c.decide("s", Q)
    assert r.meta.route == ["a:error", "b:ok"] and r.meta.backend == "b"


def test_on_error_raise():
    c = Client(
        [FakeBackend("a", fail=BackendError("a", "boom")), FakeBackend("b")],
        policy=Gate(on_error="raise"),
    )
    with pytest.raises(BackendError):
        c.decide("s", Q)


def test_low_confidence_escalates_then_ok():
    c = Client(
        [FakeBackend("a", confidence=0.5), FakeBackend("b", confidence=0.95)],
        policy=Gate(min_confidence=0.75),
    )
    r = c.decide("s", Q)
    assert r.meta.backend == "b"
    assert r.meta.route[0].startswith("a:low_confidence:") and r.meta.route[1] == "b:ok"


def test_all_low_confidence_returns_best():
    c = Client(
        [FakeBackend("a", confidence=0.5), FakeBackend("b", confidence=0.6)],
        policy=Gate(min_confidence=0.9),
    )
    r = c.decide("s", Q)
    assert r.meta.backend == "b" and r.meta.route[-1] == "b:accepted_low_confidence"


def test_all_failed_raises():
    c = Client(
        [
            FakeBackend("a", fail=BackendError("a", "x")),
            FakeBackend("b", fail=BackendError("b", "y")),
        ]
    )
    with pytest.raises(AllBackendsFailed) as ei:
        c.decide("s", Q)
    assert ei.value.route == ["a:error", "b:error"] and len(ei.value.errors) == 2


def test_batch_preserves_order_and_per_state_routes():
    a = FakeBackend("a", confidence=0.5, batch=True)
    b = FakeBackend("b", confidence=0.95)
    c = Client([a, b], policy=Gate(min_confidence=0.75))
    rs = c.decide_batch(["s1", "s2", "s3"], Q)
    assert [r.meta.backend for r in rs] == ["b", "b", "b"]
    assert len(a.calls) == 3 and all(r.meta.route[0].startswith("a:low_confidence") for r in rs)


async def test_async_client_same_semantics():
    c = AsyncClient([FakeBackend("a", fail=BackendError("a", "x")), FakeBackend("b")])
    r = await c.decide("s", Q)
    assert r.meta.route == ["a:error", "b:ok"]


def test_from_env_nothing_configured():
    with pytest.raises(ConfigError, match="TYPESAFE_API_KEY"):
        Client.from_env(env={})


def test_from_env_orders_backends(monkeypatch):
    import decide.client as mod

    made = []
    monkeypatch.setattr(
        mod, "load_backend", lambda name, **kw: made.append((name, kw)) or FakeBackend(name)
    )
    c = Client.from_env(
        env={"TYPESAFE_API_KEY": "k", "OPENROUTER_API_KEY": "o", "DECIDE_MIN_CONFIDENCE": "0.8"}
    )
    assert [n for n, _ in made] == ["typesafe", "openrouter"]
    assert c.policy.min_confidence == 0.8


def test_from_env_explicit_order(monkeypatch):
    import decide.client as mod

    monkeypatch.setattr(mod, "load_backend", lambda name, **kw: FakeBackend(name))
    c = Client.from_env(
        env={"DECIDE_BACKENDS": "llm,typesafe", "TYPESAFE_API_KEY": "k", "OPENAI_API_KEY": "z"}
    )
    assert [b.name for b in c.backends] == ["llm", "typesafe"]
