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
from decide.wire import from_wire_answers, parse_wire_request, to_wire_answers, to_wire_request
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
            "sev": {"type": "score", "probabilities": {"0": 0.25, "1": 0.75}},
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


def test_from_wire_answers_empty_choice_probabilities_raises_bad_response():
    with pytest.raises(BadResponseError):
        from_wire_answers(
            {
                "team": {"type": "choice", "probabilities": {}},
                "sev": {"type": "score", "probabilities": {"0": 0.25, "1": 0.75}},
                "refund": {"type": "noul", "noul": 0.9},
            },
            REQ,
        )


def test_from_wire_answers_empty_score_probabilities_raises_bad_response():
    with pytest.raises(BadResponseError):
        from_wire_answers(
            {
                "team": {"type": "choice", "probabilities": {"billing": 0.8, "eng": 0.2}},
                "sev": {"type": "score", "probabilities": {}},
                "refund": {"type": "noul", "noul": 0.9},
            },
            REQ,
        )
    with pytest.raises(BadResponseError):
        from_wire_answers(
            {
                "team": {"type": "choice", "probabilities": {"billing": 1.0}},
                "sev": {"type": "score", "probabilities": {"0": 1, "1": 0}},
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
                "sev": {"type": "score", "score": 1.0, "probabilities": {"0": 0.0, "1": 1.0}},
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
        "confidence": 0.6,
        "probabilities": {"billing": 0.4, "eng": 0.6},
    }
    assert w["sev"] == {
        "type": "score",
        "score": 1.0,
        "confidence": 1.0,
        "legend": {"0": "minor", "1": "blocked"},
        "probabilities": {"0": 0.0, "1": 1.0},
    }
    assert w["refund"] == {"type": "noul", "noul": 0.2}


def test_to_wire_answers_choice_confidence_is_probability_of_chosen_option():
    # `choice` can legitimately differ from the probability argmax (e.g. a caller
    # constructed the answer directly, or `from_wire_answers` accepted an explicit
    # `choice` field that wasn't the top-probability candidate). `confidence` must
    # track the chosen option's own probability, not just the highest one.
    resp = Response(
        answers={"team": ChoiceAnswer("billing", {"billing": 0.3, "eng": 0.7})},
        meta=Meta("fake", None, 0.0),
    )
    w = to_wire_answers(resp)
    assert w["team"]["confidence"] == 0.3


def test_to_wire_answers_rejects_choice_answer_with_no_probabilities():
    resp = Response(answers={"team": ChoiceAnswer("billing", {})}, meta=Meta("fake", None, 0.0))
    with pytest.raises(BadResponseError):
        to_wire_answers(resp)


def test_to_wire_answers_rejects_choice_not_in_its_own_probabilities():
    resp = Response(
        answers={"team": ChoiceAnswer("sales", {"billing": 0.5, "eng": 0.5})},
        meta=Meta("fake", None, 0.0),
    )
    with pytest.raises(BadResponseError):
        to_wire_answers(resp)


def test_to_wire_answers_rejects_score_answer_with_no_probabilities():
    resp = Response(
        answers={"sev": ScoreAnswer(0.0, [], ["minor", "blocked"])},
        meta=Meta("fake", None, 0.0),
    )
    with pytest.raises(BadResponseError):
        to_wire_answers(resp)


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
                "sev": {"type": "score", "probabilities": {"0": 1, "1": 0}},
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
                "sev": {"type": "score", "probabilities": {"0": 1, "1": 0}},
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
                "sev": {"type": "score", "probabilities": {"0": 1, "1": 0}},
                "refund": {"type": "noul", "noul": 0.1},
            },
            REQ,
        )


def test_from_wire_answers_rejects_missing_probabilities_field():
    with pytest.raises(BadResponseError):
        from_wire_answers(
            {
                "team": {"type": "choice", "choice": "billing"},
                "sev": {"type": "score", "probabilities": {"0": 1, "1": 0}},
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
                "sev": {"type": "score", "probabilities": {"0": 1, "1": 0}},
                "refund": {"type": "noul"},
            },
            REQ,
        )


def test_from_wire_answers_rejects_non_numeric_probabilities_and_noul():
    with pytest.raises(BadResponseError):
        from_wire_answers(
            {
                "team": {"type": "choice", "probabilities": {"billing": "high", "eng": 0.2}},
                "sev": {"type": "score", "probabilities": {"0": 1, "1": 0}},
                "refund": {"type": "noul", "noul": 0.1},
            },
            REQ,
        )
    with pytest.raises(BadResponseError):
        from_wire_answers(
            {
                "team": {"type": "choice", "probabilities": {"billing": 1.0}},
                "sev": {"type": "score", "probabilities": {"0": "low", "1": "high"}},
                "refund": {"type": "noul", "noul": 0.1},
            },
            REQ,
        )
    with pytest.raises(BadResponseError):
        from_wire_answers(
            {
                "team": {"type": "choice", "probabilities": {"billing": 1.0}},
                "sev": {"type": "score", "probabilities": {"0": 1, "1": 0}},
                "refund": {"type": "noul", "noul": "yes"},
            },
            REQ,
        )


def test_from_wire_answers_rejects_score_probabilities_that_are_not_a_mapping():
    # TypeSafe's SystemOneResponse represents Score probabilities as an object keyed
    # by stringified level index ("0", "1", ...), matching `legend`, not a JSON array.
    with pytest.raises(BadResponseError):
        from_wire_answers(
            {
                "team": {"type": "choice", "probabilities": {"billing": 1.0}},
                "sev": {"type": "score", "probabilities": [1, 0]},
                "refund": {"type": "noul", "noul": 0.1},
            },
            REQ,
        )


def test_parse_wire_request_inverts_to_wire_request():
    parsed = parse_wire_request(to_wire_request(REQ))
    assert parsed.state == REQ.state
    assert parsed.model == REQ.model
    assert parsed.questions == REQ.questions


def test_parse_wire_request_handles_omitted_model_and_noul_without_criteria():
    wire = to_wire_request(Request("s", {"q": Noul("x")}))
    parsed = parse_wire_request(wire)
    assert parsed.model is None
    assert parsed.questions["q"] == Noul("x")


def test_parse_wire_request_rejects_missing_questions():
    with pytest.raises(ValueError, match="questions"):
        parse_wire_request({"state": "s"})


def test_parse_wire_request_rejects_non_mapping_questions():
    with pytest.raises(ValueError, match="questions"):
        parse_wire_request({"state": "s", "questions": ["not", "a", "mapping"]})


def test_parse_wire_request_rejects_empty_questions():
    with pytest.raises(ValueError, match="questions"):
        parse_wire_request({"state": "s", "questions": {}})


def test_parse_wire_request_rejects_missing_state():
    with pytest.raises(ValueError, match="state"):
        parse_wire_request({"questions": {"q": {"type": "noul", "instructions": "x"}}})


@pytest.mark.parametrize("bad_state", [42, None, True, b"bytes"])
def test_parse_wire_request_rejects_wrong_state_type(bad_state):
    with pytest.raises(ValueError, match="state"):
        parse_wire_request(
            {
                "state": bad_state,
                "questions": {"q": {"type": "noul", "instructions": "x"}},
            }
        )


@pytest.mark.parametrize("good_state", ["text", {"a": 1}, [1, 2, 3]])
def test_parse_wire_request_accepts_str_mapping_and_sequence_state(good_state):
    parsed = parse_wire_request(
        {"state": good_state, "questions": {"q": {"type": "noul", "instructions": "x"}}}
    )
    assert parsed.state == good_state


def test_parse_wire_request_rejects_non_mapping_body():
    with pytest.raises(ValueError, match="mapping"):
        parse_wire_request(["not", "a", "mapping"])


def test_parse_wire_request_rejects_unknown_question_type():
    with pytest.raises(ValueError, match="type"):
        parse_wire_request(
            {"state": "s", "questions": {"q": {"type": "mystery", "instructions": "x"}}}
        )


def test_parse_wire_request_defaults_missing_or_null_instructions_to_empty_string():
    parsed = parse_wire_request(
        {
            "state": "s",
            "questions": {
                "noul_missing": {"type": "noul"},
                "noul_null": {"type": "noul", "instructions": None},
                "choice": {"type": "choice", "criteria": {"a": None}},
                "score": {"type": "score", "criteria": ["a", "b"]},
            },
        }
    )
    assert parsed.questions["noul_missing"] == Noul("")
    assert parsed.questions["noul_null"] == Noul("")
    assert parsed.questions["choice"] == Choice("", {"a": None})
    assert parsed.questions["score"] == Score("", ["a", "b"])

    # Round-trips: the empty default is a real instructions value, not an omission.
    wire = to_wire_request(parsed)
    assert wire["questions"]["noul_missing"]["instructions"] == ""
    assert wire["questions"]["choice"]["instructions"] == ""
    assert wire["questions"]["score"]["instructions"] == ""


def test_parse_wire_request_rejects_missing_criteria_for_choice_and_score():
    with pytest.raises(ValueError, match="criteria"):
        parse_wire_request(
            {"state": "s", "questions": {"q": {"type": "choice", "instructions": "x"}}}
        )
    with pytest.raises(ValueError, match="criteria"):
        parse_wire_request(
            {"state": "s", "questions": {"q": {"type": "score", "instructions": "x"}}}
        )


def test_parse_wire_request_rejects_wrong_criteria_container_per_type():
    with pytest.raises(ValueError, match="criteria"):
        parse_wire_request(
            {
                "state": "s",
                "questions": {"q": {"type": "choice", "instructions": "x", "criteria": ["a", "b"]}},
            }
        )
    with pytest.raises(ValueError, match="criteria"):
        parse_wire_request(
            {
                "state": "s",
                "questions": {"q": {"type": "score", "instructions": "x", "criteria": {"a": "b"}}},
            }
        )
    with pytest.raises(ValueError, match="criteria"):
        parse_wire_request(
            {
                "state": "s",
                "questions": {
                    "q": {"type": "noul", "instructions": "x", "criteria": ["true", "false"]}
                },
            }
        )


def test_from_wire_answers_tolerates_sparse_score_probabilities():
    # A missing index defaults to 0.0, and a key that isn't a valid index (e.g. a
    # stray non-integer key, or an out-of-range index) is ignored rather than
    # rejected.
    a = from_wire_answers(
        {
            "team": {"type": "choice", "probabilities": {"billing": 1.0}},
            "sev": {"type": "score", "probabilities": {"0": 0.4, "lo": 1, "9": 1}},
            "refund": {"type": "noul", "noul": 0.1},
        },
        REQ,
    )
    assert a["sev"] == ScoreAnswer(0.0, [0.4, 0.0], ["minor", "blocked"])
