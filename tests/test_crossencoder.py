import math

import pytest

from decide.backends.crossencoder import CrossEncoderBackend, GenericTemplate, KaLMJevTemplate
from decide.types import Choice, Noul, Request, Score


class StubModel:
    """Scores a pair by a table lookup on the document text; records calls."""

    def __init__(self, table):
        self.table = table
        self.calls = []

    def predict(self, pairs, activation_fn=None, batch_size=32, **kw):
        self.calls.append(list(pairs))
        return [next(v for k, v in self.table.items() if k in doc) for _q, doc in pairs]


REQ = Request(
    "charged twice",
    {
        "team": Choice("Which team?", {"billing": "money", "eng": "bugs"}),
        "sev": Score("Severity?", ["minor", "blocked"]),
        "refund": Noul("Refund?"),
    },
)


def test_single_predict_call_and_aggregation():
    m = StubModel({"billing": 2.0, "eng": 0.0, "minor": 0.0, "blocked": 1.0, "Refund?": 1.0})
    r = CrossEncoderBackend(m).decide(REQ)
    assert len(m.calls) == 1 and len(m.calls[0]) == 2 + 2 + 1
    p = r.choices["team"].probabilities
    assert (
        abs(p["billing"] - math.exp(2) / (math.exp(2) + 1)) < 1e-9
        and r.choices["team"].choice == "billing"
    )
    assert abs(r.scores["sev"].score - math.exp(1) / (1 + math.exp(1))) < 1e-9
    assert abs(r.nouls["refund"].noul - 1 / (1 + math.exp(-1))) < 1e-9
    assert r.meta.backend == "crossencoder"


def test_temperature_flattens():
    m = StubModel({"billing": 2.0, "eng": 0.0, "minor": 0, "blocked": 0, "Refund?": 0})
    hot = (
        CrossEncoderBackend(m, temperature=10.0)
        .decide(REQ)
        .choices["team"]
        .probabilities["billing"]
    )
    cold = (
        CrossEncoderBackend(m, temperature=0.5).decide(REQ).choices["team"].probabilities["billing"]
    )
    assert 0.5 < hot < cold


def test_noul_with_criteria_uses_true_minus_false():
    req = Request("s", {"n": Noul("q", {"true": "yes it is", "false": "no it is not"})})
    m = StubModel({"yes it is": 3.0, "no it is not": 1.0})
    r = CrossEncoderBackend(m).decide(req)
    assert abs(r.nouls["n"].noul - 1 / (1 + math.exp(-2))) < 1e-9


def test_templates_include_state_instructions_candidate():
    for t in (GenericTemplate(), KaLMJevTemplate()):
        q, d = t.choice("STATE", "INSTR", "cand", "desc")
        joined = q + d
        assert "STATE" in joined and "INSTR" in joined and "cand" in joined and "desc" in joined


def test_batch_is_one_call_and_results_correspond_to_their_input():
    # Each request has its own candidates and its own winner, so a wrong slice
    # offset in the aggregation would attach the wrong request's answer.
    req_a = Request("a", {"q": Choice("Which team?", {"cat_a1": None, "cat_a2": None})})
    req_b = Request("b", {"q": Choice("Which team?", {"cat_b1": None, "cat_b2": None})})
    req_c = Request("c", {"q": Choice("Which team?", {"cat_c1": None, "cat_c2": None})})

    table = {
        "cat_a1": 5.0,
        "cat_a2": 0.0,
        "cat_b1": 0.0,
        "cat_b2": 5.0,
        "cat_c1": 5.0,
        "cat_c2": 0.0,
    }
    m = StubModel(table)
    rs = CrossEncoderBackend(m).decide_batch([req_a, req_b, req_c])

    assert len(rs) == 3 and len(m.calls) == 1 and len(m.calls[0]) == 6
    assert rs[0].choices["q"].choice == "cat_a1"
    assert rs[1].choices["q"].choice == "cat_b2"
    assert rs[2].choices["q"].choice == "cat_c1"


@pytest.mark.parametrize("bad_temperature", [0, -1.0, float("nan")])
def test_temperature_must_be_positive_finite(bad_temperature):
    m = StubModel({"billing": 1.0, "eng": 0.0, "minor": 0, "blocked": 0, "Refund?": 0})
    with pytest.raises(ValueError, match="temperature"):
        CrossEncoderBackend(m, temperature=bad_temperature)


class MultiLabelStubModel:
    """Returns two scores per pair, like a model with num_labels > 1 and no scalar activation."""

    def __init__(self):
        self.calls = []

    def predict(self, pairs, activation_fn=None, batch_size=32, **kw):
        self.calls.append(list(pairs))
        return [[1.0, 2.0] for _pair in pairs]


def test_num_labels_greater_than_one_gives_config_error():
    from decide.errors import ConfigError

    m = MultiLabelStubModel()
    with pytest.raises(ConfigError, match="num_labels"):
        CrossEncoderBackend(m).decide(REQ)


def test_noul_with_only_true_side_uses_default_for_false():
    req = Request("s", {"n": Noul("q", {"true": "yes it is"})})
    m = StubModel({"yes it is": 2.0, "The answer to the question is no.": 0.0})
    r = CrossEncoderBackend(m).decide(req)
    assert len(m.calls[0]) == 2
    assert abs(r.nouls["n"].noul - 1 / (1 + math.exp(-2))) < 1e-9


class _ExplodingModel:
    def predict(self, pairs, activation_fn=None, batch_size=32, **kw):
        raise RuntimeError("boom")


def test_model_predict_exception_wrapped_as_backend_error():
    from decide.errors import BackendError

    with pytest.raises(BackendError, match="boom"):
        CrossEncoderBackend(_ExplodingModel()).decide(REQ)


def test_string_model_without_extra_gives_config_error(monkeypatch):
    import builtins

    real = builtins.__import__

    def fake(name, *a, **k):
        if name.startswith("sentence_transformers"):
            raise ImportError
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    from decide.errors import ConfigError

    with pytest.raises(ConfigError, match=r"pydecide\[st\]"):
        CrossEncoderBackend("cross-encoder/ms-marco-MiniLM-L6-v2")


@pytest.mark.live
def test_live_crossencoder_routes_to_billing():
    backend = CrossEncoderBackend("cross-encoder/ms-marco-MiniLM-L6-v2")
    req = Request(
        "I was charged twice for the same order last week, please refund the duplicate charge.",
        {
            "team": Choice(
                "Which team should handle this?",
                {
                    "billing": "Charges, invoices, payment problems, refunds",
                    "eng": "Bugs, crashes, broken features",
                    "shipping": "Delivery status, delays, lost packages",
                },
            )
        },
    )
    r = backend.decide(req)
    assert r.choices["team"].choice == "billing"
