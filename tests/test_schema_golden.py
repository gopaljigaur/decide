import json
import pathlib

import pytest

from decide.types import (
    Choice,
    ChoiceAnswer,
    Meta,
    Noul,
    NoulAnswer,
    Request,
    Response,
    Score,
    ScoreAnswer,
)
from decide.wire import parse_wire_request, to_wire_answers, to_wire_request

jsonschema = pytest.importorskip("jsonschema")

SCHEMA = json.loads(
    pathlib.Path(__file__).with_name("fixtures").joinpath("typesafe_schema.json").read_text()
)


def test_our_request_validates_against_sdk_request_schema():
    req = Request(
        {"t": "x"},
        {
            "c": Choice("q", {"a": "d", "b": None}),
            "s": Score("q", ["lo", "hi"]),
            "n": Noul("q", {"true": "y"}),
        },
        model="jev-latest",
    )
    jsonschema.validate(to_wire_request(req), SCHEMA["request"])


def test_parse_wire_request_round_trips_a_schema_valid_body():
    body = {
        "state": {"t": "x"},
        "model": "jev-latest",
        "questions": {
            "c": {"type": "choice", "instructions": "q", "criteria": {"a": "d", "b": None}},
            "s": {"type": "score", "instructions": "q", "criteria": ["lo", "hi"]},
            "n": {"type": "noul", "instructions": "q", "criteria": {"true": "y"}},
        },
    }
    jsonschema.validate(body, SCHEMA["request"])
    parsed = parse_wire_request(body)
    assert to_wire_request(parsed) == body


def test_our_answers_validate_against_sdk_response_schema():
    resp = Response(
        {
            "c": ChoiceAnswer("a", {"a": 0.7, "b": 0.3}),
            "s": ScoreAnswer(0.6, [0.4, 0.6], ["lo", "hi"]),
            "n": NoulAnswer(0.2),
        },
        Meta("x", "jev-latest", 1.0),
    )
    # SystemOneResponse ("response" in the fixture) is the full API envelope: besides
    # `answers` (what `to_wire_answers` produces) its own `required` list demands a
    # top-level `model` and `usage`. Those two aren't part of the per-answer codec
    # `to_wire_answers` covers -- they come from the response envelope, not from a
    # `Response`'s answers -- so the golden payload fills them in here from the
    # fixture's required fields rather than weakening the assertion.
    assert set(SCHEMA["response"]["required"]) == {"answers", "model", "usage"}
    usage_required = SCHEMA["response"]["$defs"]["Usage"]["required"]
    assert set(usage_required) == {"input_tokens", "output_tokens"}
    payload = {
        "model": "jev-latest",
        "usage": dict.fromkeys(usage_required, 0),
        "answers": to_wire_answers(resp),
    }
    jsonschema.validate(payload, SCHEMA["response"])
