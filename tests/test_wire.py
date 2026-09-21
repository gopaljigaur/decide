import pytest

from decide.backends.base import Capabilities
from decide.errors import BadResponseError
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
from decide.wire import from_wire_answers, to_wire_answers, to_wire_request
from tests.conftest import FakeBackend

REQ = Request(
    state={"ticket": "charged twice"},
    questions={
        "team": Choice("Which team?", {"billing": "money", "eng": None}),
        "sev": Score("How severe?", ["minor", "blocked"]),
        "refund": Noul("Refund asked?"),
    },
    model="jev-latest",
)


def test_to_wire_request_matches_typesafe_shape():
    w = to_wire_request(REQ)
    assert w["state"] == {"ticket": "charged twice"}
    assert w["model"] == "jev-latest"
    assert w["questions"]["team"] == {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "money", "eng": None},
    }
    assert w["questions"]["sev"] == {
        "type": "score",
        "instructions": "How severe?",
        "criteria": ["minor", "blocked"],
    }
    assert w["questions"]["refund"] == {"type": "noul", "instructions": "Refund asked?"}
    assert "model" not in to_wire_request(Request("s", {"q": Noul("x")}))


def test_from_wire_answers_parses_all_types_and_fills_gaps():
    a = from_wire_answers(
        {
            "team": {"type": "choice", "probabilities": {"billing": 0.8, "eng": 0.2}},
            "sev": {"type": "score", "probabilities": [0.25, 0.75]},
            "refund": {"type": "noul", "noul": 0.9},
        },
        REQ,
    )
    assert a["team"] == ChoiceAnswer("billing", {"billing": 0.8, "eng": 0.2})
    assert a["sev"] == ScoreAnswer(0.75, [0.25, 0.75], ["minor", "blocked"])
    assert a["refund"] == NoulAnswer(0.9)


def test_from_wire_answers_rejects_missing_and_unknown():
    with pytest.raises(BadResponseError):
        from_wire_answers({"team": {"type": "choice", "probabilities": {"billing": 1.0}}}, REQ)
    with pytest.raises(BadResponseError):
        from_wire_answers(
            {
                "team": {"type": "choice", "probabilities": {"billing": 1.0}},
                "sev": {"type": "score", "probabilities": [1, 0]},
                "refund": {"type": "noul", "noul": 0.1},
                "ghost": {"type": "noul", "noul": 0.1},
            },
            REQ,
        )


def test_round_trip_answers():
    resp = Response(
        answers=from_wire_answers(
            {
                "team": {
                    "type": "choice",
                    "choice": "eng",
                    "probabilities": {"billing": 0.4, "eng": 0.6},
                },
                "sev": {"type": "score", "score": 1.0, "probabilities": [0.0, 1.0]},
                "refund": {"type": "noul", "noul": 0.2},
            },
            REQ,
        ),
        meta=Meta("fake", None, 0.0),
    )
    w = to_wire_answers(resp)
    assert w["team"] == {
        "type": "choice",
        "choice": "eng",
        "probabilities": {"billing": 0.4, "eng": 0.6},
    }
    assert w["sev"] == {"type": "score", "score": 1.0, "probabilities": [0.0, 1.0]}
    assert w["refund"] == {"type": "noul", "noul": 0.2}


def test_fake_backend_fabricates_confident_answers(fake_backend):
    r = fake_backend.decide(REQ)
    assert r.meta.backend == "fake" and r.meta.latency_ms >= 0
    assert r.choices["team"].choice == "billing"
    assert r.choices["team"].probabilities["billing"] == 0.9
    assert abs(sum(r.scores["sev"].probabilities) - 1) < 1e-9
    assert r.nouls["refund"].noul == 0.9


def test_fake_backend_batch_capability_flag():
    assert FakeBackend(batch=True).capabilities() == Capabilities(batch=True)


def test_from_wire_answers_rejects_type_mismatch():
    with pytest.raises(BadResponseError):
        from_wire_answers(
            {
                "team": {"type": "score", "probabilities": {"billing": 1.0}},
                "sev": {"type": "score", "probabilities": [1, 0]},
                "refund": {"type": "noul", "noul": 0.1},
            },
            REQ,
        )
    with pytest.raises(BadResponseError):
        from_wire_answers(
            {
                "team": {"type": "choice", "probabilities": {"billing": 1.0}},
                "sev": {"type": "choice", "probabilities": [1, 0]},
                "refund": {"type": "noul", "noul": 0.1},
            },
            REQ,
        )
    with pytest.raises(BadResponseError):
        from_wire_answers(
            {
                "team": {"type": "choice", "probabilities": {"billing": 1.0}},
                "sev": {"type": "score", "probabilities": [1, 0]},
                "refund": {"type": "choice", "noul": 0.1},
            },
            REQ,
        )


def test_from_wire_answers_rejects_choice_not_among_candidates():
    with pytest.raises(BadResponseError):
        from_wire_answers(
            {
                "team": {
                    "type": "choice",
                    "choice": "sales",
                    "probabilities": {"billing": 0.5, "eng": 0.5},
                },
                "sev": {"type": "score", "probabilities": [1, 0]},
                "refund": {"type": "noul", "noul": 0.1},
            },
            REQ,
        )


def test_from_wire_answers_rejects_missing_probabilities_field():
    with pytest.raises(BadResponseError):
        from_wire_answers(
            {
                "team": {"type": "choice", "choice": "billing"},
                "sev": {"type": "score", "probabilities": [1, 0]},
                "refund": {"type": "noul", "noul": 0.1},
            },
            REQ,
        )
    with pytest.raises(BadResponseError):
        from_wire_answers(
            {
                "team": {"type": "choice", "probabilities": {"billing": 1.0}},
                "sev": {"type": "score"},
                "refund": {"type": "noul", "noul": 0.1},
            },
            REQ,
        )


def test_from_wire_answers_rejects_missing_noul_field():
    with pytest.raises(BadResponseError):
        from_wire_answers(
            {
                "team": {"type": "choice", "probabilities": {"billing": 1.0}},
                "sev": {"type": "score", "probabilities": [1, 0]},
                "refund": {"type": "noul"},
            },
            REQ,
        )


def test_from_wire_answers_rejects_non_numeric_probabilities_and_noul():
    with pytest.raises(BadResponseError):
        from_wire_answers(
            {
                "team": {"type": "choice", "probabilities": {"billing": "high", "eng": 0.2}},
                "sev": {"type": "score", "probabilities": [1, 0]},
                "refund": {"type": "noul", "noul": 0.1},
            },
            REQ,
        )
    with pytest.raises(BadResponseError):
        from_wire_answers(
            {
                "team": {"type": "choice", "probabilities": {"billing": 1.0}},
                "sev": {"type": "score", "probabilities": ["low", "high"]},
                "refund": {"type": "noul", "noul": 0.1},
            },
            REQ,
        )
    with pytest.raises(BadResponseError):
        from_wire_answers(
            {
                "team": {"type": "choice", "probabilities": {"billing": 1.0}},
                "sev": {"type": "score", "probabilities": [1, 0]},
                "refund": {"type": "noul", "noul": "yes"},
            },
            REQ,
        )
