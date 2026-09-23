import pytest

from decide.gate import Gate
from decide.types import ChoiceAnswer, Meta, NoulAnswer, Response, ScoreAnswer


def _resp(**answers):
    return Response(answers=answers, meta=Meta("fake", None, 0.0))


def test_confidence_per_question_type():
    r = _resp(
        c=ChoiceAnswer("a", {"a": 0.6, "b": 0.4}),
        n=NoulAnswer(0.2),
        s=ScoreAnswer(1, [0.5, 0.5], ["x", "y"]),
    )
    assert Gate().confidence(r) == {"c": 0.6, "n": 0.8}


def test_default_gate_passes_everything():
    result = Gate().passes(_resp(c=ChoiceAnswer("a", {"a": 0.34, "b": 0.33, "c": 0.33})))
    assert result == (True, "ok")


def test_global_threshold_and_reason():
    ok, why = Gate(min_confidence=0.75).passes(_resp(c=ChoiceAnswer("a", {"a": 0.6, "b": 0.4})))
    assert not ok and why.startswith("low_confidence:c=0.60<0.75")


def test_per_question_override_wins():
    g = Gate(min_confidence=0.9, per_question={"c": 0.5})
    assert g.passes(_resp(c=ChoiceAnswer("a", {"a": 0.6, "b": 0.4}))) == (True, "ok")
    assert not g.passes(_resp(c=ChoiceAnswer("a", {"a": 0.6, "b": 0.4}), n=NoulAnswer(0.5)))[0]


def test_confidence_defaults_to_zero_for_empty_choice_probabilities():
    r = _resp(c=ChoiceAnswer("a", {}))
    assert Gate().confidence(r) == {"c": 0.0}


def test_scores_are_not_gated():
    result = Gate(min_confidence=0.99).passes(_resp(s=ScoreAnswer(0.5, [0.5, 0.5], ["x", "y"])))
    assert result == (True, "ok")


def test_per_question_beats_per_type_beats_min_confidence():
    g = Gate(
        min_confidence=0.5,
        per_type={"choice": 0.7},
        per_question={"c": 0.1},
    )
    # per_question wins even though per_type and min_confidence would fail it.
    assert g.passes(_resp(c=ChoiceAnswer("a", {"a": 0.2, "b": 0.8}))) == (True, "ok")


def test_per_type_beats_min_confidence():
    g = Gate(min_confidence=0.1, per_type={"choice": 0.9})
    ok, why = g.passes(_resp(c=ChoiceAnswer("a", {"a": 0.6, "b": 0.4})))
    assert not ok and why == "low_confidence:c=0.60<0.90"


def test_per_type_noul_gates_while_lower_per_type_choice_passes():
    g = Gate(per_type={"choice": 0.5, "noul": 0.95})
    r = _resp(c=ChoiceAnswer("a", {"a": 0.6, "b": 0.4}), n=NoulAnswer(0.9))
    ok, why = g.passes(r)
    assert not ok and why == "low_confidence:n=0.90<0.95"


def test_per_type_falls_back_to_min_confidence_for_unlisted_type():
    g = Gate(min_confidence=0.8, per_type={"choice": 0.1})
    ok, why = g.passes(_resp(n=NoulAnswer(0.5)))
    assert not ok and why == "low_confidence:n=0.50<0.80"


def test_per_type_invalid_key_raises():
    with pytest.raises(ValueError, match="per_type"):
        Gate(per_type={"bogus": 0.5})


def test_per_type_score_key_is_accepted_but_scores_are_never_gated():
    g = Gate(per_type={"score": 0.99})
    result = g.passes(_resp(s=ScoreAnswer(0.5, [0.5, 0.5], ["x", "y"])))
    assert result == (True, "ok")
