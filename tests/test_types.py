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
    render_content,
)


def test_render_content_json_is_compact_and_sorted():
    assert render_content("x") == "x"
    assert render_content({"b": 1, "a": [1, 2]}) == '{"a":[1,2],"b":1}'


def test_choice_requires_candidates_and_unique_names():
    with pytest.raises(ValueError, match="criteria"):
        Choice("q", {})
    with pytest.raises(ValueError, match="criteria"):
        Choice("q", {"": None})
    c = Choice("q", {"a": None, "b": "desc"})
    assert list(c.criteria) == ["a", "b"]
    with pytest.raises(TypeError):
        c.criteria["z"] = None  # immutable


def test_score_requires_two_levels():
    with pytest.raises(ValueError, match="criteria"):
        Score("q", ["only"])
    assert Score("q", ["low", "high"]).criteria == ("low", "high")


def test_noul_criteria_keys_restricted():
    Noul("q")
    Noul("q", {"true": "yes"})
    with pytest.raises(ValueError, match="criteria"):
        Noul("q", {"maybe": "x"})


def test_request_needs_questions_with_names():
    with pytest.raises(ValueError, match="questions"):
        Request("s", {})
    with pytest.raises(ValueError, match="questions"):
        Request("s", {"": Noul("q")})


def test_response_views_and_getitem():
    r = Response(
        answers={
            "t": ChoiceAnswer("a", {"a": 0.7, "b": 0.3}),
            "s": ScoreAnswer(1.0, [0.2, 0.6, 0.2], ["lo", "mid", "hi"]),
            "n": NoulAnswer(0.9),
        },
        meta=Meta(backend="fake", model=None, latency_ms=1.0),
    )
    assert r["t"].choice == "a"
    assert set(r.choices) == {"t"} and set(r.scores) == {"s"} and set(r.nouls) == {"n"}
    r2 = r.with_meta(route=["fake:ok"])
    assert r2.meta.route == ["fake:ok"] and r.meta.route == []
