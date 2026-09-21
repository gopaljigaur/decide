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


def test_batch_is_one_call():
    m = StubModel({"billing": 1.0, "eng": 0.0, "minor": 0, "blocked": 0, "Refund?": 0})
    rs = CrossEncoderBackend(m).decide_batch([REQ, REQ, REQ])
    assert len(rs) == 3 and len(m.calls) == 1 and len(m.calls[0]) == 15


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
