import pytest

from decide.client import Client
from decide.errors import AllBackendsFailed, BackendError
from decide.types import Choice, Noul, Score
from tests.conftest import FakeBackend

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from decide.server import create_app  # noqa: E402

QUESTIONS = {
    "team": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "money", "eng": None},
    },
    "sev": {
        "type": "score",
        "instructions": "How severe?",
        "criteria": ["minor", "blocked"],
    },
    "refund": {"type": "noul", "instructions": "Refund asked?"},
}


def _body(**overrides):
    body = {"state": "ticket text", "model": "jev-latest", "questions": QUESTIONS}
    body.update(overrides)
    return body


def test_systemone_round_trip_returns_answers_and_route():
    client = Client([FakeBackend(name="ok")])
    app = create_app(client)
    tc = TestClient(app)

    resp = tc.post("/v1/systemone", json=_body())

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["model"] == "ok"
    assert set(payload["answers"]) == {"team", "sev", "refund"}
    assert payload["answers"]["team"]["type"] == "choice"
    assert payload["answers"]["sev"]["type"] == "score"
    assert payload["answers"]["refund"]["type"] == "noul"
    assert payload["decide"]["route"] == ["ok:ok"]
    assert payload["decide"]["backend"] == "ok"
    assert isinstance(payload["decide"]["latency_ms"], float)
    assert payload["usage"] == {"input_tokens": 0, "output_tokens": 0}


def test_systemone_uses_backend_meta_model_when_present():
    backend = FakeBackend(name="ok")
    backend.model = "fake-model-v1"
    client = Client([backend])
    app = create_app(client)
    tc = TestClient(app)

    resp = tc.post("/v1/systemone", json=_body())

    assert resp.status_code == 200
    assert resp.json()["model"] == "fake-model-v1"


def test_health_lists_backend_names():
    client = Client([FakeBackend(name="one"), FakeBackend(name="two")])
    app = create_app(client)
    tc = TestClient(app)

    resp = tc.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "backends": ["one", "two"]}


def test_models_lists_backends_openai_style():
    client = Client([FakeBackend(name="one"), FakeBackend(name="two")])
    app = create_app(client)
    tc = TestClient(app)

    resp = tc.get("/v1/models")

    assert resp.status_code == 200
    assert resp.json() == {
        "object": "list",
        "data": [{"id": "one", "object": "model"}, {"id": "two", "object": "model"}],
    }


def test_systemone_400_when_questions_missing():
    client = Client([FakeBackend()])
    app = create_app(client)
    tc = TestClient(app)

    resp = tc.post("/v1/systemone", json={"state": "s"})

    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert "questions" in body["error"]["message"]


def test_systemone_400_on_invalid_json():
    client = Client([FakeBackend()])
    app = create_app(client)
    tc = TestClient(app)

    resp = tc.post(
        "/v1/systemone", content=b"{not json", headers={"content-type": "application/json"}
    )

    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"


def test_systemone_401_without_bearer_when_api_key_set():
    client = Client([FakeBackend()])
    app = create_app(client, api_key="t")
    tc = TestClient(app)

    resp = tc.post("/v1/systemone", json=_body())

    assert resp.status_code == 401
    assert resp.json()["error"]["type"] == "authentication_error"


def test_systemone_401_with_wrong_bearer_when_api_key_set():
    client = Client([FakeBackend()])
    app = create_app(client, api_key="t")
    tc = TestClient(app)

    resp = tc.post("/v1/systemone", json=_body(), headers={"Authorization": "Bearer wrong"})

    assert resp.status_code == 401
    assert resp.json()["error"]["type"] == "authentication_error"


def test_systemone_200_with_correct_bearer_when_api_key_set():
    client = Client([FakeBackend(name="ok")])
    app = create_app(client, api_key="t")
    tc = TestClient(app)

    resp = tc.post("/v1/systemone", json=_body(), headers={"Authorization": "Bearer t"})

    assert resp.status_code == 200


def test_systemone_502_when_only_backend_fails():
    failing = FakeBackend(name="down", fail=BackendError("down", "boom"))
    client = Client([failing])
    app = create_app(client)
    tc = TestClient(app)

    resp = tc.post("/v1/systemone", json=_body())

    assert resp.status_code == 502
    body = resp.json()
    assert body["error"]["type"] == "backend_error"
    assert body["error"]["route"] == ["down:error"]


def test_systemone_502_body_reports_all_backends_failed_route(monkeypatch):
    failing = FakeBackend(name="down", fail=BackendError("down", "boom"))
    client = Client([failing])
    app = create_app(client)
    tc = TestClient(app)

    resp = tc.post("/v1/systemone", json=_body())

    assert resp.status_code == 502
    # Sanity: matches what AllBackendsFailed.route would report directly.
    try:
        client.decide("ticket text", {"refund": Noul("Refund asked?")})
    except AllBackendsFailed as exc:
        assert resp.json()["error"]["route"] == exc.route


def test_create_app_config_error_without_fastapi(monkeypatch):
    import builtins

    from decide.errors import ConfigError

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "fastapi" or name.startswith("fastapi."):
            raise ImportError("no fastapi")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ConfigError, match="pydecide\\[server\\]"):
        create_app(Client([FakeBackend()]))


def test_response_validates_against_sdk_response_schema():
    import json
    import pathlib

    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        pathlib.Path(__file__).with_name("fixtures").joinpath("typesafe_schema.json").read_text()
    )

    client = Client([FakeBackend(name="ok")])
    app = create_app(client)
    tc = TestClient(app)

    resp = tc.post("/v1/systemone", json=_body())
    assert resp.status_code == 200
    jsonschema.validate(resp.json(), schema["response"])


def test_response_parses_with_sdk_pydantic_model():
    try:
        from typesafe_sdk._schemas.models import SystemOneResponse
    except ImportError:
        pytest.skip("typesafe_sdk._schemas.models.SystemOneResponse not available")

    client = Client([FakeBackend(name="ok")])
    app = create_app(client)
    tc = TestClient(app)

    resp = tc.post("/v1/systemone", json=_body())
    assert resp.status_code == 200
    SystemOneResponse.model_validate(resp.json())


def test_questions_round_trip_choice_score_noul_via_client_calls():
    backend = FakeBackend(name="ok")
    client = Client([backend])
    app = create_app(client)
    tc = TestClient(app)

    tc.post("/v1/systemone", json=_body())

    assert len(backend.calls) == 1
    call = backend.calls[0]
    assert isinstance(call.questions["team"], Choice)
    assert isinstance(call.questions["sev"], Score)
    assert isinstance(call.questions["refund"], Noul)
    assert call.state == "ticket text"
    assert call.model == "jev-latest"
