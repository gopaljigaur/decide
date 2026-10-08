import json
import os

import httpx
import pytest

from decide import backends
from decide.backends.openai_decisions import OpenAIDecisionsBackend
from decide.errors import (
    AuthError,
    BackendConnectionError,
    BackendError,
    BadResponseError,
    RateLimitError,
)
from decide.types import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Request,
    Score,
    ScoreAnswer,
)

REQ = Request(
    "charged twice",
    {
        "team": Choice("Which team?", {"billing": "money", "eng": None}),
        "sev": Score("How bad?", ["minor", {"k": "blocked"}]),
        "refund": Noul("Refund?"),
    },
)
OK = {
    "model": "gpt-6-luna",
    "answers": [
        {
            "name": "team",
            "type": "choice",
            "choice": "billing",
            "confidence": 0.9,
            "probabilities": [
                {"value": "billing", "probability": 0.9},
                {"value": "eng", "probability": 0.1},
            ],
        },
        {
            "name": "sev",
            "type": "score",
            "score": 0.3,
            "confidence": 0.7,
            "probabilities": [
                {"value": 0, "label": "minor", "probability": 0.7},
                {"value": 1, "label": '{"k":"blocked"}', "probability": 0.3},
            ],
        },
        {"name": "refund", "type": "predicate", "probability": 0.8},
    ],
}


def _backend(handler, **kw):
    return OpenAIDecisionsBackend(api_key="sk-test", transport=httpx.MockTransport(handler), **kw)


def _json(payload):
    return lambda request: httpx.Response(200, json=payload)


def _with(index, **changes):
    answers = [dict(a) for a in OK["answers"]]
    answers[index] = {**answers[index], **changes}
    return {**OK, "answers": answers}


def test_request_shape_and_headers():
    seen = {}

    def handler(r: httpx.Request):
        seen["url"] = str(r.url)
        seen["headers"] = r.headers
        seen["body"] = json.loads(r.content)
        return httpx.Response(200, json=OK)

    _backend(handler).decide(REQ)
    assert seen["url"] == "https://api.openai.com/v1/decisions"
    assert seen["headers"]["authorization"] == "Bearer sk-test"
    assert seen["headers"]["content-type"] == "application/json"
    assert seen["body"] == {
        "model": "gpt-6-luna",
        "input": "charged twice",
        "questions": [
            {
                "name": "team",
                "type": "choice",
                "instructions": "Which team?",
                "choices": [{"value": "billing", "description": "money"}, {"value": "eng"}],
            },
            {
                "name": "sev",
                "type": "score",
                "instructions": "How bad?",
                "levels": [{"label": "minor"}, {"label": '{"k":"blocked"}'}],
            },
            {"name": "refund", "type": "predicate", "instructions": "Refund?"},
        ],
    }


def test_structured_state_and_instructions_are_rendered_and_noul_criteria_are_kept():
    seen = {}

    def handler(r):
        seen["body"] = json.loads(r.content)
        return httpx.Response(200, json={"answers": [OK["answers"][2]]})

    req = Request(
        {"b": 1, "a": 2},
        {"refund": Noul({"q": "Refund?"}, {"true": "wants one", "false": "does not"})},
    )
    _backend(handler).decide(req)
    assert seen["body"]["input"] == '{"a":2,"b":1}'
    assert seen["body"]["questions"] == [
        {
            "name": "refund",
            "type": "predicate",
            "instructions": '{"q":"Refund?"}\n\nIf true: wants one\nIf false: does not',
        }
    ]


def test_parses_all_three_answer_types():
    resp = _backend(_json(OK)).decide(REQ)
    assert resp.answers == {
        "team": ChoiceAnswer("billing", {"billing": 0.9, "eng": 0.1}),
        "sev": ScoreAnswer(0.3, [0.7, 0.3], ["minor", '{"k":"blocked"}']),
        "refund": NoulAnswer(0.8),
    }
    assert resp.meta.backend == "openai_decisions" and resp.meta.raw == OK


def test_answers_are_matched_by_name_not_position():
    shuffled = {**OK, "answers": list(reversed(OK["answers"]))}
    assert _backend(_json(shuffled)).decide(REQ).answers == _backend(_json(OK)).decide(REQ).answers


def test_score_probabilities_are_ordered_by_level_index():
    probs = list(reversed(OK["answers"][1]["probabilities"]))
    resp = _backend(_json(_with(1, probabilities=probs))).decide(REQ)
    assert resp.scores["sev"].probabilities == [0.7, 0.3]


def test_model_precedence_and_reported_model():
    seen = {}

    def handler(r):
        seen["m"] = json.loads(r.content)["model"]
        return httpx.Response(200, json={k: v for k, v in OK.items() if k != "model"})

    r1 = _backend(handler).decide(REQ)
    assert seen["m"] == "gpt-6-luna" and r1.meta.model == "gpt-6-luna"
    r2 = _backend(handler, model="gpt-6-sol").decide(REQ)
    assert seen["m"] == "gpt-6-sol" and r2.meta.model == "gpt-6-sol"
    r3 = _backend(handler, model="gpt-6-sol").decide(Request(REQ.state, REQ.questions, model="x"))
    assert seen["m"] == "x" and r3.meta.model == "x"


def test_model_in_response_wins():
    r = _backend(_json({**OK, "model": "gpt-6-luna-2026-10-06"}), model="alias").decide(REQ)
    assert r.meta.model == "gpt-6-luna-2026-10-06"


def test_base_url_override():
    seen = {}

    def handler(r):
        seen["url"] = str(r.url)
        return httpx.Response(200, json=OK)

    _backend(handler, base_url="https://proxy.test/v1/").decide(REQ)
    assert seen["url"] == "https://proxy.test/v1/decisions"


@pytest.mark.parametrize(
    ("status", "error"),
    [(401, AuthError), (403, AuthError), (429, RateLimitError), (500, BackendError)],
)
def test_status_mapping(status, error):
    b = _backend(lambda r: httpx.Response(status, json={"error": "x"}))
    with pytest.raises(error) as exc:
        b.decide(REQ)
    assert exc.value.status == status and exc.value.body == {"error": "x"}


def test_transport_failure_is_a_connection_error():
    def boom(r):
        raise httpx.ConnectError("no route", request=r)

    with pytest.raises(BackendConnectionError):
        _backend(boom).decide(REQ)


def test_refusal_raises_naming_the_question():
    payload = _with(2, type="refusal")
    payload["answers"][2] = {"name": "refund", "type": "refusal"}
    with pytest.raises(BadResponseError, match=r"refund.*refus"):
        _backend(_json(payload)).decide(REQ)


def _drop(index):
    return {**OK, "answers": [a for i, a in enumerate(OK["answers"]) if i != index]}


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ("not json", "JSON"),
        ([], "answers"),
        ({"answers": {}}, "answers"),
        ({"answers": ["x"]}, "answer"),
        (_drop(1), "sev"),
        ({**OK, "answers": [*OK["answers"], {"name": "extra", "type": "predicate"}]}, "extra"),
        ({**OK, "answers": [*OK["answers"], OK["answers"][2]]}, "refund"),
        (_with(2, type="choice"), "refund"),
        (_with(0, type="predicate"), "team"),
        (_with(2, probability=None), "refund"),
        (_with(2, probability="0.8"), "refund"),
        (_with(2, probability=True), "refund"),
        (_with(0, choice=None), "team"),
        (_with(0, choice=3), "team"),
        (_with(0, probabilities=None), "team"),
        (_with(0, probabilities=[]), "team"),
        (_with(0, probabilities=[{"value": "billing"}]), "team"),
        (_with(0, probabilities=[{"value": "billing", "probability": "x"}]), "team"),
        (_with(1, score=None), "sev"),
        (_with(1, score="0.3"), "sev"),
        (_with(1, probabilities=[]), "sev"),
        (_with(1, probabilities=[{"value": 0, "probability": 1.0}]), "sev"),
        (
            _with(
                1, probabilities=[{"value": 0, "probability": 1}, {"value": 5, "probability": 0}]
            ),
            "sev",
        ),
        (
            _with(
                1, probabilities=[{"value": 0, "probability": 1}, {"value": 0, "probability": 0}]
            ),
            "sev",
        ),
    ],
)
def test_malformed_responses_are_rejected(payload, match):
    if payload == "not json":
        handler = lambda r: httpx.Response(200, text="nope")  # noqa: E731
    else:
        handler = _json(payload)
    with pytest.raises(BadResponseError, match=match):
        _backend(handler).decide(REQ)


async def test_async_path():
    seen = {}

    async def handler(r):
        seen["body"] = json.loads(r.content)
        return httpx.Response(200, json=OK)

    b = OpenAIDecisionsBackend(api_key="k", transport=httpx.MockTransport(handler))
    resp = await b.adecide(REQ)
    assert resp.nouls["refund"].noul == 0.8 and resp.meta.model == "gpt-6-luna"
    assert seen["body"]["questions"][0]["name"] == "team"
    await b.aclose()


async def test_async_errors_are_mapped():
    b = OpenAIDecisionsBackend(
        api_key="k", transport=httpx.MockTransport(lambda r: httpx.Response(429))
    )
    with pytest.raises(RateLimitError):
        await b.adecide(REQ)
    await b.aclose()


def test_client_is_reused_and_close_closes_it():
    b = _backend(_json(OK))
    b.decide(REQ)
    client = b._client
    b.decide(REQ)
    assert b._client is client and client.is_closed is False
    b.close()
    assert client.is_closed is True and b._client is None


async def test_aclose_closes_the_async_client():
    b = OpenAIDecisionsBackend(api_key="k", transport=httpx.MockTransport(_json(OK)))
    await b.adecide(REQ)
    aclient = b._aclient
    await b.aclose()
    assert aclient.is_closed is True and b._aclient is None


def test_capabilities():
    caps = _backend(_json(OK)).capabilities()
    assert (caps.batch, caps.local) == (False, False)


def test_registered_and_loadable():
    assert backends.REGISTRY["openai_decisions"].endswith(":OpenAIDecisionsBackend")
    assert backends.available()["openai_decisions"] is True
    assert isinstance(backends.load("openai_decisions", api_key="k"), OpenAIDecisionsBackend)


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY is not set")
def test_live_smoke():
    b = OpenAIDecisionsBackend(api_key=os.environ["OPENAI_API_KEY"])
    resp = b.decide(
        Request(
            "I was charged twice for the same order.",
            {
                "team": Choice("Which team owns this?", {"billing": None, "engineering": None}),
                "refund": Noul("Does the customer want a refund?"),
            },
        )
    )
    assert resp.choices["team"].choice in {"billing", "engineering"}
    assert 0.0 <= resp.nouls["refund"].noul <= 1.0
