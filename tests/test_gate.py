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


def test_scores_are_not_gated():
    result = Gate(min_confidence=0.99).passes(_resp(s=ScoreAnswer(0.5, [0.5, 0.5], ["x", "y"])))
    assert result == (True, "ok")
