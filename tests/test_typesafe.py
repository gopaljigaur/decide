import json

import httpx
import pytest

from decide.backends.typesafe import TypeSafeBackend
from decide.errors import AuthError, BackendConnectionError, BadResponseError, RateLimitError
from decide.types import Choice, Noul, Request

REQ = Request(
    "charged twice",
    {"team": Choice("Which team?", {"billing": None, "eng": None}), "refund": Noul("Refund?")},
)
OK = {
    "answers": {
        "team": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.9, "eng": 0.1},
        },
        "refund": {"type": "noul", "noul": 0.8},
    }
}


def _backend(handler, **kw):
    return TypeSafeBackend(api_key="sk-test", transport=httpx.MockTransport(handler), **kw)


def test_request_shape_and_headers():
    seen = {}

    def handler(r: httpx.Request):
        seen["url"] = str(r.url)
        seen["auth"] = r.headers["authorization"]
        seen["body"] = json.loads(r.content)
        return httpx.Response(200, json=OK)

    resp = _backend(handler).decide(REQ)
    assert seen["url"] == "https://api.typesafe.ai/v1/systemone"
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["model"] == "jev-latest" and seen["body"]["state"] == "charged twice"
    assert seen["body"]["questions"]["team"]["type"] == "choice"
    assert resp.choices["team"].choice == "billing"
    assert resp.meta.backend == "typesafe" and resp.meta.raw == OK


def test_model_precedence():
    seen = {}

    def handler(r):
        seen["m"] = json.loads(r.content)["model"]
        return httpx.Response(200, json=OK)

    _backend(handler, model="jev-1.13").decide(REQ)
    assert seen["m"] == "jev-1.13"
    _backend(handler).decide(Request(REQ.state, REQ.questions, model="jev-x"))
    assert seen["m"] == "jev-x"


@pytest.mark.parametrize("status,exc", [(401, AuthError), (403, AuthError), (429, RateLimitError)])
def test_http_errors_map(status, exc):
    with pytest.raises(exc):
        _backend(lambda r: httpx.Response(status, json={"error": "nope"})).decide(REQ)


def test_other_http_error_maps_to_backend_error_with_body():
    from decide.errors import BackendError

    body = {"error": "boom " + "x" * 300}

    def handler(r):
        return httpx.Response(500, json=body)

    with pytest.raises(BackendError) as exc_info:
        _backend(handler).decide(REQ)
    assert not isinstance(exc_info.value, (AuthError, RateLimitError))
    assert "boom" in exc_info.value.message


def test_transport_error():
    def handler(r):
        raise httpx.ConnectError("down")

    with pytest.raises(BackendConnectionError):
        _backend(handler).decide(REQ)


def test_bad_body():
    with pytest.raises(BadResponseError):
        _backend(lambda r: httpx.Response(200, json={"nope": 1})).decide(REQ)


async def test_async_path():
    b = _backend(lambda r: httpx.Response(200, json=OK))
    r = await b.adecide(REQ)
    assert r.nouls["refund"].noul == 0.8


@pytest.mark.live
def test_live_typesafe():
    import os

    b = TypeSafeBackend(api_key=os.environ["TYPESAFE_API_KEY"])
    r = b.decide(REQ)
    assert set(r.choices["team"].probabilities) == {"billing", "eng"}
