import pytest

from decide import AsyncClient, Choice, ChoiceAnswer, Client, Gate, Noul, NoulAnswer
from decide.errors import AllBackendsFailed, BackendError, ConfigError
from tests.conftest import FakeBackend

Q = {"team": Choice("Which team?", {"billing": None, "eng": None}), "refund": Noul("Refund?")}


def _capturing_loader(captured):
    def _load(name, **kw):
        captured[name] = kw
        return FakeBackend(name)

    return _load


def _by_state_answers(conf):
    """Answers that echo the request's state as the choice, at a fixed confidence.

    Lets a test assert a result belongs to a particular input state instead of
    just its backend name, so reordering bugs are caught.
    """

    def _answers(request):
        return {
            "team": ChoiceAnswer(request.state, {request.state: conf, "other": 1 - conf}),
            "refund": NoulAnswer(conf),
        }

    return _answers


def _fail_for_state(backend_name, state):
    def _answers(request):
        if request.state == state:
            raise BackendError(backend_name, "boom")
        return {
            "team": ChoiceAnswer("billing", {"billing": 0.95, "eng": 0.05}),
            "refund": NoulAnswer(0.95),
        }

    return _answers


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
    assert ei.value.partial == {} and ei.value.failed == {}


async def test_async_client_same_semantics():
    c = AsyncClient([FakeBackend("a", fail=BackendError("a", "x")), FakeBackend("b")])
    r = await c.decide("s", Q)
    assert r.meta.route == ["a:error", "b:ok"]


# --- decide_batch: ordering, routing, mixed sub-batches ---------------------------------


def test_batch_preserves_order_and_per_state_routes():
    a = FakeBackend("a", answers=_by_state_answers(0.5), batch=True)
    b = FakeBackend("b", answers=_by_state_answers(0.95))
    c = Client([a, b], policy=Gate(min_confidence=0.75))
    rs = c.decide_batch(["s1", "s2", "s3"], Q)

    assert [r.meta.backend for r in rs] == ["b", "b", "b"]
    assert [r.choices["team"].choice for r in rs] == ["s1", "s2", "s3"]
    assert len(a.calls) == 3 and all(r.meta.route[0].startswith("a:low_confidence") for r in rs)


def test_batch_mixed_confidence_partitions_sub_batch():
    def a_answers(request):
        conf = 0.95 if request.state in ("s1", "s3") else 0.5
        return {
            "team": ChoiceAnswer("billing", {"billing": conf, "eng": 1 - conf}),
            "refund": NoulAnswer(conf),
        }

    a = FakeBackend("a", answers=a_answers, batch=True)
    b = FakeBackend("b", confidence=0.95)
    c = Client([a, b], policy=Gate(min_confidence=0.75))
    rs = c.decide_batch(["s1", "s2", "s3"], Q)

    assert [r.meta.backend for r in rs] == ["a", "b", "a"]
    assert len(a.calls) == 3
    assert len(b.calls) == 1
    assert b.calls[0].state == "s2"


def test_batch_call_error_fails_whole_sub_batch():
    a = FakeBackend("a", batch=True, fail=BackendError("a", "boom"))
    b = FakeBackend("b")
    c = Client([a, b])
    rs = c.decide_batch(["s1", "s2"], Q)

    assert [r.meta.route[0] for r in rs] == ["a:error", "a:error"]
    assert [r.meta.backend for r in rs] == ["b", "b"]


def test_batch_on_error_raise_stops_sending_early():
    a = FakeBackend("a", fail=BackendError("a", "boom"))
    b = FakeBackend("b")
    c = Client([a, b], policy=Gate(on_error="raise"))

    with pytest.raises(BackendError):
        c.decide_batch(["s1", "s2", "s3"], Q)
    assert len(a.calls) == 1


def test_batch_on_error_raise_with_batch_capable_backend():
    a = FakeBackend("a", batch=True, fail=BackendError("a", "boom"))
    b = FakeBackend("b")
    c = Client([a, b], policy=Gate(on_error="raise"))

    with pytest.raises(BackendError):
        c.decide_batch(["s1", "s2"], Q)


def test_batch_partial_failure_preserves_completed_siblings():
    a = FakeBackend("a", answers=_fail_for_state("a", "s2"))
    b = FakeBackend("b", answers=_fail_for_state("b", "s2"))
    c = Client([a, b], policy=Gate(min_confidence=0.75))

    with pytest.raises(AllBackendsFailed) as ei:
        c.decide_batch(["s1", "s2", "s3"], Q)

    exc = ei.value
    assert set(exc.partial) == {0, 2}
    assert exc.partial[0].meta.backend == "a" and exc.partial[0].meta.route == ["a:ok"]
    assert exc.partial[2].meta.backend == "a" and exc.partial[2].meta.route == ["a:ok"]
    assert list(exc.failed) == [1]
    assert exc.failed[1] == ["a:error", "b:error"]
    assert exc.route == ["a:error", "b:error"]
    assert len(exc.errors) == 2


async def test_async_client_decide_batch():
    a = FakeBackend("a", answers=_by_state_answers_varied({"s1", "s3"}, high=0.95, low=0.5))
    b = FakeBackend("b", answers=_by_state_answers(0.95))
    c = AsyncClient([a, b], policy=Gate(min_confidence=0.75))
    rs = await c.decide_batch(["s1", "s2", "s3"], Q)

    assert [r.meta.backend for r in rs] == ["a", "b", "a"]
    assert [r.choices["team"].choice for r in rs] == ["s1", "s2", "s3"]


def _by_state_answers_varied(high_confidence_states, *, high, low):
    def _answers(request):
        conf = high if request.state in high_confidence_states else low
        return {
            "team": ChoiceAnswer(request.state, {request.state: conf, "other": 1 - conf}),
            "refund": NoulAnswer(conf),
        }

    return _answers


async def test_async_client_decide_batch_on_error_raise():
    a = FakeBackend("a", fail=BackendError("a", "boom"))
    b = FakeBackend("b")
    c = AsyncClient([a, b], policy=Gate(on_error="raise"))

    with pytest.raises(BackendError):
        await c.decide_batch(["s1", "s2", "s3"], Q)


async def test_async_client_decide_batch_all_failed_raises_with_partial():
    a = FakeBackend("a", answers=_fail_for_state("a", "s2"))
    b = FakeBackend("b", answers=_fail_for_state("b", "s2"))
    c = AsyncClient([a, b], policy=Gate(min_confidence=0.75))

    with pytest.raises(AllBackendsFailed) as ei:
        await c.decide_batch(["s1", "s2", "s3"], Q)

    exc = ei.value
    assert set(exc.partial) == {0, 2}
    assert list(exc.failed) == [1]


# --- context managers --------------------------------------------------------------------


class _ClosingBackend(FakeBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.closed = False

    def close(self):
        self.closed = True


class _AsyncClosingBackend(FakeBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aclosed = False

    async def aclose(self):
        self.aclosed = True


def test_client_context_manager_closes_backends():
    backend = _ClosingBackend("a")
    with Client([backend]) as c:
        assert c.backends == [backend]
    assert backend.closed


async def test_async_client_context_manager_calls_aclose():
    backend = _AsyncClosingBackend("a")
    async with AsyncClient([backend]) as c:
        assert c.backends == [backend]
    assert backend.aclosed


async def test_async_client_context_manager_falls_back_to_close():
    backend = _ClosingBackend("a")
    async with AsyncClient([backend]):
        pass
    assert backend.closed


# --- from_env ------------------------------------------------------------------------


def test_from_env_nothing_configured():
    with pytest.raises(ConfigError, match="TYPESAFE_API_KEY"):
        Client.from_env(env={})


def test_from_env_orders_backends(monkeypatch):
    import decide.client as mod

    made = []

    def _load(name, **kw):
        made.append((name, kw))
        return FakeBackend(name)

    monkeypatch.setattr(mod, "load_backend", _load)
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


def test_from_env_blank_decide_backends_falls_back_to_auto_detect(monkeypatch):
    import decide.client as mod

    captured = {}
    monkeypatch.setattr(mod, "load_backend", _capturing_loader(captured))
    c = Client.from_env(env={"DECIDE_BACKENDS": "   ", "TYPESAFE_API_KEY": "k"})
    assert [b.name for b in c.backends] == ["typesafe"]


def test_from_env_explicit_backend_missing_env_raises(monkeypatch):
    import decide.client as mod

    monkeypatch.setattr(mod, "load_backend", _capturing_loader({}))
    with pytest.raises(ConfigError, match="TYPESAFE_API_KEY"):
        Client.from_env(env={"DECIDE_BACKENDS": "typesafe"})


def test_from_env_crossencoder_cannot_be_configured():
    with pytest.raises(ConfigError, match="cannot be configured from the environment"):
        Client.from_env(env={"DECIDE_BACKENDS": "crossencoder"})


def test_from_env_bad_min_confidence_raises_config_error():
    with pytest.raises(ConfigError):
        Client.from_env(env={"TYPESAFE_API_KEY": "k", "DECIDE_MIN_CONFIDENCE": "not-a-float"})


def test_from_env_typesafe_kwargs_without_base_url(monkeypatch):
    import decide.client as mod

    captured = {}
    monkeypatch.setattr(mod, "load_backend", _capturing_loader(captured))
    Client.from_env(env={"TYPESAFE_API_KEY": "k"})
    assert captured["typesafe"] == {"api_key": "k"}


def test_from_env_typesafe_kwargs_with_base_url(monkeypatch):
    import decide.client as mod

    captured = {}
    monkeypatch.setattr(mod, "load_backend", _capturing_loader(captured))
    Client.from_env(env={"TYPESAFE_API_KEY": "k", "TYPESAFE_BASE_URL": "http://x"})
    assert captured["typesafe"] == {"api_key": "k", "base_url": "http://x"}


def test_from_env_openrouter_kwargs(monkeypatch):
    import decide.client as mod

    captured = {}
    monkeypatch.setattr(mod, "load_backend", _capturing_loader(captured))
    Client.from_env(env={"OPENROUTER_API_KEY": "o"})
    assert captured["openrouter"] == {"api_key": "o"}


def test_from_env_llm_kwargs_omits_none_and_defaults_model(monkeypatch):
    import decide.client as mod

    captured = {}
    monkeypatch.setattr(mod, "load_backend", _capturing_loader(captured))
    Client.from_env(env={"DECIDE_LLM_BASE_URL": "http://y"})
    assert captured["llm"] == {"base_url": "http://y", "model": "gpt-4o-mini"}


def test_from_env_llm_kwargs_with_api_key_and_model(monkeypatch):
    import decide.client as mod

    captured = {}
    monkeypatch.setattr(mod, "load_backend", _capturing_loader(captured))
    Client.from_env(env={"OPENAI_API_KEY": "z", "DECIDE_LLM_MODEL": "gpt-4o"})
    assert captured["llm"] == {"api_key": "z", "model": "gpt-4o"}


def test_from_env_laya_kwargs(monkeypatch):
    import decide.client as mod

    captured = {}
    monkeypatch.setattr(mod, "load_backend", _capturing_loader(captured))
    monkeypatch.setattr(mod, "available", lambda: {"laya": True, "laya_mlx": False})
    c = Client.from_env(env={"DECIDE_LOCAL_MODEL": "m1"})
    assert captured["laya"] == {"model": "m1"}
    assert [b.name for b in c.backends] == ["laya"]


def test_from_env_laya_mlx_preferred_when_both_available(monkeypatch):
    import decide.client as mod

    captured = {}
    monkeypatch.setattr(mod, "load_backend", _capturing_loader(captured))
    monkeypatch.setattr(mod, "available", lambda: {"laya": True, "laya_mlx": True})
    c = Client.from_env(env={"DECIDE_LOCAL_MODEL": "m1"})
    assert captured == {"laya_mlx": {"model": "m1"}}
    assert [b.name for b in c.backends] == ["laya_mlx"]
