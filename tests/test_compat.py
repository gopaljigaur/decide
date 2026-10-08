import builtins
import json
import subprocess
import sys

import httpx
import pytest

import decide
from decide.errors import ConfigError
from decide.gate import Gate
from decide.types import Choice, Noul, Score
from tests.conftest import FakeBackend

sdk = pytest.importorskip("typesafe_sdk")

from decide.compat import AsyncTypeSafeClient, DecideResponse, TypeSafeClient  # noqa: E402

OK = {
    "model": "jev-latest",
    "usage": {"input_tokens": 7, "output_tokens": 3},
    "answers": {
        "team": {
            "type": "choice",
            "choice": "billing",
            "confidence": 0.9,
            "probabilities": {"billing": 0.9, "eng": 0.1},
        },
        "sev": {
            "type": "score",
            "score": 0.3,
            "confidence": 0.7,
            "legend": {"0": "minor", "1": "blocked"},
            "probabilities": {"0": 0.7, "1": 0.3},
        },
        "refund": {"type": "noul", "noul": 0.8},
    },
}

VENDOR_QUESTIONS = {
    "team": sdk.Choice(instructions="Which team?", criteria={"billing": "money", "eng": None}),
    "sev": sdk.Score(instructions="How bad?", criteria=["minor", "blocked"]),
    "refund": sdk.Noul(instructions="Refund?", criteria={"true": "wants one"}),
}


def _ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=OK)


def _down(request: httpx.Request) -> httpx.Response:
    return httpx.Response(503, text="down")


def _client(handler=_ok, **kw):
    return TypeSafeClient(api_key="sk-test", transport=httpx.MockTransport(handler), **kw)


def test_vendor_questions_round_trip_to_answers_and_wire():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=OK)

    r = _client(handler).system_one("charged twice", VENDOR_QUESTIONS)
    qs = seen["body"]["questions"]
    assert qs["team"] == {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "money", "eng": None},
    }
    assert qs["sev"] == {
        "type": "score",
        "instructions": "How bad?",
        "criteria": ["minor", "blocked"],
    }
    assert qs["refund"] == {
        "type": "noul",
        "instructions": "Refund?",
        "criteria": {"true": "wants one"},
    }
    assert r.choices["team"].choice == "billing"
    assert r.scores["sev"].probabilities[0] == pytest.approx(0.7)
    assert r.nouls["refund"].noul == pytest.approx(0.8)


def test_noul_without_criteria_and_raw_dict_questions():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=OK)

    questions = {
        "team": {
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {"billing": None, "eng": None},
        },
        "sev": sdk.Score(instructions="How bad?", criteria=["minor", "blocked"]),
        "refund": sdk.Noul(instructions="Refund?"),
    }
    _client(handler).system_one("x", questions)
    assert seen["body"]["questions"]["refund"] == {"type": "noul", "instructions": "Refund?"}
    assert seen["body"]["questions"]["team"]["type"] == "choice"


def test_hosted_only_matches_vendor_client():
    httpx2 = pytest.importorskip("httpx2")
    vendor = sdk.TypeSafeClient(
        api_key="sk-test",
        transport=httpx2.MockTransport(lambda r: httpx2.Response(200, json=OK)),
    )
    expected = vendor.system_one("charged twice", VENDOR_QUESTIONS)
    got = _client().system_one("charged twice", VENDOR_QUESTIONS)
    assert got.model_dump() == expected.model_dump()
    assert got.decide.backend == "typesafe"
    assert got.decide.route == ["typesafe:ok"]


def test_hosted_failure_falls_through_to_local_backend():
    local = FakeBackend(name="local")
    r = _client(_down, fallback=[local]).system_one("charged twice", VENDOR_QUESTIONS)
    assert r.decide.backend == "local"
    assert r.decide.route == ["typesafe:error", "local:ok"]
    assert r.decide.latency_ms >= 0
    assert r.choices["team"].choice == "billing"
    assert len(local.calls) == 1


def test_gate_sends_low_confidence_hosted_answer_to_fallback():
    local = FakeBackend(name="local", confidence=0.99)
    only_team = {**OK, "answers": {"team": OK["answers"]["team"]}}
    c = _client(
        lambda r: httpx.Response(200, json=only_team),
        fallback=[local],
        gate=Gate(min_confidence=0.95),
    )
    r = c.system_one("x", {"team": VENDOR_QUESTIONS["team"]})
    assert r.decide.backend == "local"
    assert r.decide.route[0].startswith("typesafe:low_confidence")


def test_string_fallback_resolves_through_load(monkeypatch):
    loaded = []
    local = FakeBackend(name="local")

    def fake_load(name, **kw):
        loaded.append(name)
        return local

    monkeypatch.setattr("decide.compat.load_backend", fake_load)
    r = _client(_down, fallback=["laya_mlx"]).system_one("x", VENDOR_QUESTIONS)
    assert loaded == ["laya_mlx"] and r.decide.backend == "local"


def test_response_is_vendor_model_and_validates():
    r = _client().system_one("x", VENDOR_QUESTIONS)
    assert isinstance(r, DecideResponse)
    assert isinstance(r, sdk.SystemOneResponse)
    assert (r.model, r.usage.input_tokens, r.usage.output_tokens) == ("jev-latest", 7, 3)
    assert set(r.answers) == {"team", "sev", "refund"}
    assert sdk.SystemOneResponse.model_validate(
        r.model_dump(), strict=False
    ) == sdk.SystemOneResponse(model=r.model, usage=r.usage, answers=r.answers)
    assert "decide" not in r.model_dump()


def test_fallback_response_validates_as_vendor_model():
    r = _client(_down, fallback=[FakeBackend(name="local")]).system_one("x", VENDOR_QUESTIONS)
    assert isinstance(r, sdk.SystemOneResponse)
    assert r.model == "local"
    assert (r.usage.input_tokens, r.usage.output_tokens) == (0, 0)
    assert r.scores["sev"].legend == {0: "minor", 1: "blocked"}


def test_mixed_vendor_and_native_questions():
    seen = {}

    def handler(request):
        seen["q"] = json.loads(request.content)["questions"]
        return httpx.Response(200, json=OK)

    questions = {
        "team": Choice("Which team?", ["billing", "eng"]),
        "sev": sdk.Score(instructions="How bad?", criteria=["minor", "blocked"]),
        "refund": Noul("Refund?"),
    }
    r = _client(handler).system_one("x", questions)
    assert [seen["q"][k]["type"] for k in ("team", "sev", "refund")] == ["choice", "score", "noul"]
    assert r.choices["team"].choice == "billing"
    assert Score("q", ["a", "b"])  # native Score is still a valid question object


def test_response_model_is_not_supported():
    with pytest.raises(NotImplementedError, match="response_model"):
        _client().system_one("x", VENDOR_QUESTIONS, response_model=dict)


@pytest.mark.parametrize(
    "kw", [{"retry": object()}, {"extra_headers": {"a": "b"}}, {"extra_body": {"a": 1}}]
)
def test_unsupported_per_call_options_are_loud(kw):
    with pytest.raises(NotImplementedError, match=next(iter(kw))):
        _client().system_one("x", VENDOR_QUESTIONS, **kw)


def test_http_client_constructor_option_is_loud():
    with pytest.raises(NotImplementedError, match="http_client"):
        TypeSafeClient(api_key="k", http_client=object())


def test_constructor_headers_model_base_url_and_env_key(monkeypatch):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["h"] = request.headers
        seen["model"] = json.loads(request.content)["model"]
        return httpx.Response(200, json=OK)

    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-env")
    c = TypeSafeClient(
        model="jev-1",
        base_url="https://example.test",
        headers={"X-Team": "a"},
        transport=httpx.MockTransport(handler),
    )
    c.system_one("x", VENDOR_QUESTIONS)
    assert seen["url"] == "https://example.test/v1/systemone"
    assert seen["h"]["authorization"] == "Bearer sk-env" and seen["h"]["x-team"] == "a"
    assert seen["model"] == "jev-1"


def test_missing_api_key_is_config_error(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="TYPESAFE_API_KEY"):
        TypeSafeClient()


def test_context_manager_closes_backends():
    with _client() as c:
        c.system_one("x", VENDOR_QUESTIONS)
    assert c._client.backends[0]._client is None


async def test_async_client_falls_through_to_local():
    local = FakeBackend(name="local")
    async with AsyncTypeSafeClient(
        api_key="sk-test", transport=httpx.MockTransport(_down), fallback=[local]
    ) as c:
        r = await c.system_one("x", VENDOR_QUESTIONS)
    assert isinstance(r, sdk.SystemOneResponse)
    assert r.decide.route == ["typesafe:error", "local:ok"]


async def test_async_hosted_only_and_response_model():
    c = AsyncTypeSafeClient(api_key="sk-test", transport=httpx.MockTransport(_ok))
    r = await c.system_one("x", VENDOR_QUESTIONS)
    assert r.decide.route == ["typesafe:ok"]
    with pytest.raises(NotImplementedError, match="response_model"):
        await c.system_one("x", VENDOR_QUESTIONS, response_model=dict)
    await c.aclose()


def _block_sdk(monkeypatch):
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "typesafe_sdk" or name.startswith("typesafe_sdk."):
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    for mod in [m for m in sys.modules if m.startswith("typesafe_sdk")]:
        monkeypatch.delitem(sys.modules, mod)


def test_missing_sdk_is_config_error_naming_install_command(monkeypatch):
    _block_sdk(monkeypatch)
    with pytest.raises(ConfigError, match="pip install typesafe-sdk"):
        TypeSafeClient(api_key="k")
    with pytest.raises(ConfigError, match="pip install typesafe-sdk"):
        AsyncTypeSafeClient(api_key="k")


def test_import_decide_does_not_need_the_sdk():
    code = (
        "import sys, builtins\n"
        "real = builtins.__import__\n"
        "def fake(name, *a, **k):\n"
        "    if name.split('.')[0] == 'typesafe_sdk':\n"
        "        raise ImportError(name)\n"
        "    return real(name, *a, **k)\n"
        "builtins.__import__ = fake\n"
        "import decide\n"
        "assert decide.TypeSafeClient and decide.AsyncTypeSafeClient\n"
        "import decide.compat\n"
        "assert 'typesafe_sdk' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_exported_from_package():
    assert decide.TypeSafeClient is TypeSafeClient
    assert decide.AsyncTypeSafeClient is AsyncTypeSafeClient
    assert {"TypeSafeClient", "AsyncTypeSafeClient"} <= set(decide.__all__)


@pytest.mark.live
def test_live_fallback_engages_with_a_bad_key():
    c = TypeSafeClient(api_key="sk-invalid", fallback=["laya_mlx"])
    r = c.system_one("charged twice", {"refund": sdk.Noul(instructions="Refund?")})
    assert r.decide.route[0] == "typesafe:error"


def test_transport_failure_is_routed_to_fallback():
    def boom(request):
        raise httpx.ConnectError("no route", request=request)

    r = _client(boom, fallback=[FakeBackend(name="local")]).system_one("x", VENDOR_QUESTIONS)
    assert r.decide.route == ["typesafe:error", "local:ok"]


def _status(code, body=None, headers=None):
    return lambda request: httpx.Response(code, json=body or {"error": "nope"}, headers=headers)


@pytest.mark.parametrize(
    ("code", "error"),
    [
        (503, "TypeSafeInternalServerError"),
        (500, "TypeSafeInternalServerError"),
        (401, "TypeSafeAuthenticationError"),
        (403, "TypeSafePermissionDeniedError"),
        (429, "TypeSafeRateLimitError"),
        (400, "TypeSafeBadRequestError"),
        (404, "TypeSafeNotFoundError"),
        (422, "TypeSafeUnprocessableEntityError"),
        (418, "TypeSafeAPIError"),
    ],
)
def test_hosted_only_http_failure_raises_vendor_error(code, error):
    with pytest.raises(getattr(sdk, error)) as exc:
        _client(_status(code, headers={"x-typesafe-request-id": "r1"})).system_one(
            "x", VENDOR_QUESTIONS
        )
    assert type(exc.value) is getattr(sdk, error)
    assert isinstance(exc.value, sdk.TypeSafeAPIError | sdk.TypeSafeError)
    assert exc.value.status == code
    assert exc.value.body == {"error": "nope"}
    assert exc.value.request_id == "r1"
    assert isinstance(exc.value.__cause__, decide.BackendError)


def test_hosted_only_transport_failure_raises_connection_error():
    def boom(request):
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(sdk.TypeSafeAPIConnectionError) as exc:
        _client(boom).system_one("x", VENDOR_QUESTIONS)
    assert type(exc.value) is sdk.TypeSafeAPIConnectionError
    assert isinstance(exc.value, sdk.TypeSafeError)


def test_hosted_only_timeout_raises_timeout_error():
    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(sdk.TypeSafeAPITimeoutError):
        _client(slow, timeout=3).system_one("x", VENDOR_QUESTIONS)


def test_hosted_only_bad_payload_raises_vendor_error():
    with pytest.raises(sdk.TypeSafeError):
        _client(lambda r: httpx.Response(200, json={"nope": 1})).system_one("x", VENDOR_QUESTIONS)


def test_all_tiers_failing_raises_both_vendor_and_decide_error():
    local = FakeBackend(name="local", fail=decide.BackendError("local", "broken"))
    with pytest.raises(sdk.TypeSafeError) as exc:
        _client(_down, fallback=[local]).system_one("x", VENDOR_QUESTIONS)
    err = exc.value
    assert isinstance(err, decide.AllBackendsFailed)
    assert isinstance(err, sdk.TypeSafeInternalServerError)
    assert err.route == ["typesafe:error", "local:error"]
    assert len(err.errors) == 2 and err.partial == {} and err.failed == {}
    assert isinstance(err.__cause__, sdk.TypeSafeInternalServerError)
    assert err.__cause__.status == 503


def test_all_tiers_failing_is_caught_by_either_handler():
    local = FakeBackend(name="local", fail=decide.BackendError("local", "broken"))
    for kind in (sdk.TypeSafeError, sdk.TypeSafeAPIError, decide.AllBackendsFailed):
        with pytest.raises(kind):
            _client(_down, fallback=[local]).system_one("x", VENDOR_QUESTIONS)


def test_fallback_that_succeeds_does_not_raise():
    r = _client(_down, fallback=[FakeBackend(name="local")]).system_one("x", VENDOR_QUESTIONS)
    assert r.decide.backend == "local"


async def test_async_hosted_only_failure_raises_vendor_error():
    c = AsyncTypeSafeClient(api_key="sk-test", transport=httpx.MockTransport(_down))
    with pytest.raises(sdk.TypeSafeInternalServerError):
        await c.system_one("x", VENDOR_QUESTIONS)
    await c.aclose()


async def test_async_all_tiers_failing_raises_both():
    local = FakeBackend(name="local", fail=decide.BackendError("local", "broken"))
    c = AsyncTypeSafeClient(
        api_key="sk-test", transport=httpx.MockTransport(_down), fallback=[local]
    )
    with pytest.raises(sdk.TypeSafeError) as exc:
        await c.system_one("x", VENDOR_QUESTIONS)
    assert isinstance(exc.value, decide.AllBackendsFailed)
    assert exc.value.route == ["typesafe:error", "local:error"]
    await c.aclose()


def test_backend_error_keeps_its_constructor_and_status_attributes():
    err = decide.BackendError("b", "m")
    assert (err.status, err.body, err.headers) == (None, None, None)
    assert str(err) == "b: m"


@pytest.fixture
def sleeps(monkeypatch):
    calls: list[float] = []
    monkeypatch.setattr("decide.compat.time.sleep", calls.append)

    async def fake(delay):
        calls.append(delay)

    monkeypatch.setattr("decide.compat.asyncio.sleep", fake)
    return calls


def _flaky(failures, code=503):
    state = {"n": 0}

    def handler(request):
        state["n"] += 1
        if state["n"] <= failures:
            return httpx.Response(code, text="down")
        return httpx.Response(200, json=OK)

    handler.state = state
    return handler


def test_retry_recovers_before_the_chain_moves_on(sleeps):
    handler = _flaky(2)
    local = FakeBackend(name="local")
    policy = sdk.RetryPolicy(max_retries=2, backoff_jitter=0)
    r = _client(handler, retry=policy, fallback=[local]).system_one("x", VENDOR_QUESTIONS)
    assert r.decide.route == ["typesafe:ok"]
    assert handler.state["n"] == 3 and local.calls == []
    assert sleeps == [0.5, 1.0]


def test_retry_exhausted_then_raises_last_error(sleeps):
    handler = _flaky(99)
    policy = sdk.RetryPolicy(max_retries=3, backoff_initial=1.0, backoff_max=3.0, backoff_jitter=0)
    with pytest.raises(sdk.TypeSafeInternalServerError):
        _client(handler, retry=policy).system_one("x", VENDOR_QUESTIONS)
    assert handler.state["n"] == 4
    assert sleeps == [1.0, 2.0, 3.0]


def test_retry_exhausted_moves_on_to_fallback(sleeps):
    handler = _flaky(99)
    local = FakeBackend(name="local")
    policy = sdk.RetryPolicy(max_retries=1, backoff_jitter=0)
    r = _client(handler, retry=policy, fallback=[local]).system_one("x", VENDOR_QUESTIONS)
    assert r.decide.route == ["typesafe:error", "local:ok"]
    assert handler.state["n"] == 2


def test_retry_only_covers_http_statuses_in_the_policy(sleeps):
    handler = _flaky(99, code=401)
    with pytest.raises(sdk.TypeSafeAuthenticationError):
        _client(handler, retry=sdk.RetryPolicy(max_retries=3)).system_one("x", VENDOR_QUESTIONS)
    assert handler.state["n"] == 1 and sleeps == []
    handler = _flaky(99, code=418)
    policy = sdk.RetryPolicy(max_retries=1, http_statuses={418}, backoff_jitter=0)
    with pytest.raises(sdk.TypeSafeAPIError):
        _client(handler, retry=policy).system_one("x", VENDOR_QUESTIONS)
    assert handler.state["n"] == 2


def test_retry_covers_transport_failures(sleeps):
    state = {"n": 0}

    def handler(request):
        state["n"] += 1
        if state["n"] == 1:
            raise httpx.ConnectError("no route", request=request)
        return httpx.Response(200, json=OK)

    policy = sdk.RetryPolicy(max_retries=1, backoff_jitter=0)
    assert _client(handler, retry=policy).system_one("x", VENDOR_QUESTIONS).decide.route == [
        "typesafe:ok"
    ]
    off = sdk.RetryPolicy(max_retries=1, api_connection_error=False)
    state["n"] = 0
    with pytest.raises(sdk.TypeSafeAPIConnectionError):
        _client(handler, retry=off).system_one("x", VENDOR_QUESTIONS)
    assert state["n"] == 1


def test_retry_jitter_only_shortens_the_delay(sleeps):
    handler = _flaky(1)
    policy = sdk.RetryPolicy(max_retries=1, backoff_initial=1.0, backoff_jitter=0.5)
    _client(handler, retry=policy).system_one("x", VENDOR_QUESTIONS)
    assert 0.5 <= sleeps[0] <= 1.0


def test_no_retry_policy_means_one_attempt(sleeps):
    handler = _flaky(99)
    with pytest.raises(sdk.TypeSafeInternalServerError):
        _client(handler).system_one("x", VENDOR_QUESTIONS)
    assert handler.state["n"] == 1 and sleeps == []


def test_retry_with_max_retries_zero_is_one_attempt(sleeps):
    handler = _flaky(99)
    with pytest.raises(sdk.TypeSafeInternalServerError):
        _client(handler, retry=sdk.RetryPolicy(max_retries=0)).system_one("x", VENDOR_QUESTIONS)
    assert handler.state["n"] == 1


async def test_async_retry_uses_async_sleep(sleeps):
    handler = _flaky(1)
    policy = sdk.RetryPolicy(max_retries=1, backoff_jitter=0)
    c = AsyncTypeSafeClient(api_key="sk-test", transport=httpx.MockTransport(handler), retry=policy)
    r = await c.system_one("x", VENDOR_QUESTIONS)
    assert r.decide.route == ["typesafe:ok"] and sleeps == [0.5]
    await c.aclose()


def test_retry_wrapper_keeps_close_working():
    policy = sdk.RetryPolicy(max_retries=1)
    with _client(retry=policy) as c:
        c.system_one("x", VENDOR_QUESTIONS)
    assert c._client.backends[0]._client is None
