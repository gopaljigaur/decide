import builtins

import pytest

from decide.errors import BackendError, BadResponseError, ConfigError
from decide.types import Choice, Noul, Request, Score
from decide.wire import to_wire_request

REQ = Request(
    "charged twice",
    {
        "team": Choice("Which team?", {"billing": "money", "eng": "bugs"}),
        "sev": Score("Severity?", ["minor", "blocked"]),
        "refund": Noul("Refund?"),
    },
)

WIRE_ANSWERS = {
    "team": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.9, "eng": 0.1}},
    "sev": {"type": "score", "score": 1.0, "probabilities": {"0": 0.1, "1": 0.9}},
    "refund": {"type": "noul", "noul": 0.7},
}


class FakeAgent:
    """Records `(state, questions)` and returns a fixed TypeSafe wire-shaped response."""

    def __init__(self, response=None, *, predict_batch=False):
        self.response = response if response is not None else {"answers": WIRE_ANSWERS}
        self.calls = []
        self.batch_calls = []
        if predict_batch:
            self.predict_batch = self._predict_batch

    def predict(self, state, questions):
        self.calls.append((state, questions))
        return self.response

    def _predict_batch(self, states, questions):
        self.batch_calls.append((states, questions))
        return [self.response for _ in states]


def _assert_parsed(r, backend_name):
    assert r.meta.backend == backend_name
    assert r.choices["team"].choice == "billing"
    assert r.choices["team"].probabilities == {"billing": 0.9, "eng": 0.1}
    assert r.scores["sev"].score == 1.0
    assert r.scores["sev"].probabilities == [0.1, 0.9]
    assert r.nouls["refund"].noul == 0.7


# ---- LayaBackend --------------------------------------------------------------------------


def test_laya_predict_called_with_wire_questions_and_state():
    from decide.backends.laya import LayaBackend

    agent = FakeAgent()
    backend = LayaBackend(agent=agent)
    r = backend.decide(REQ)

    assert len(agent.calls) == 1
    state, questions = agent.calls[0]
    assert state == REQ.state
    assert questions == to_wire_request(REQ)["questions"]
    _assert_parsed(r, "laya")
    assert r.meta.model == "convaiinnovations/laya"
    assert r.meta.raw == agent.response


def test_laya_meta_model_includes_subfolder():
    from decide.backends.laya import LayaBackend

    backend = LayaBackend(
        model="convaiinnovations/laya", subfolder="multilingual", agent=FakeAgent()
    )
    r = backend.decide(REQ)
    assert r.meta.model == "convaiinnovations/laya/multilingual"


def test_laya_capabilities_batch_false_without_predict_batch():
    from decide.backends.laya import LayaBackend

    backend = LayaBackend(agent=FakeAgent())
    caps = backend.capabilities()
    assert caps.local is True
    assert caps.batch is False


def test_laya_capabilities_batch_true_with_predict_batch():
    from decide.backends.laya import LayaBackend

    backend = LayaBackend(agent=FakeAgent(predict_batch=True))
    assert backend.capabilities().batch is True


def test_laya_decide_batch_uses_predict_batch_in_one_call():
    from decide.backends.laya import LayaBackend

    agent = FakeAgent(predict_batch=True)
    backend = LayaBackend(agent=agent)
    responses = backend.decide_batch([REQ, REQ, REQ])

    assert len(agent.batch_calls) == 1
    states, questions_list = agent.batch_calls[0]
    assert states == [REQ.state, REQ.state, REQ.state]
    assert questions_list == [to_wire_request(REQ)["questions"]] * 3
    assert len(responses) == 3
    for r in responses:
        _assert_parsed(r, "laya")


def test_laya_decide_batch_falls_back_to_decide_loop_without_predict_batch():
    from decide.backends.laya import LayaBackend

    agent = FakeAgent()
    backend = LayaBackend(agent=agent)
    responses = backend.decide_batch([REQ, REQ])

    assert len(agent.calls) == 2
    assert len(responses) == 2
    for r in responses:
        _assert_parsed(r, "laya")


def test_laya_bad_response_shape_raises_bad_response_error():
    from decide.backends.laya import LayaBackend

    backend = LayaBackend(agent=FakeAgent(response={"not_answers": {}}))
    with pytest.raises(BadResponseError):
        backend.decide(REQ)


def test_laya_agent_exception_wrapped_as_backend_error():
    from decide.backends.laya import LayaBackend

    class ExplodingAgent:
        def predict(self, state, questions):
            raise RuntimeError("boom")

    backend = LayaBackend(agent=ExplodingAgent())
    with pytest.raises(BackendError, match="boom"):
        backend.decide(REQ)


def test_laya_score_probabilities_as_list_are_normalized(monkeypatch=None):
    from decide.backends.laya import LayaBackend

    response = {
        "answers": {
            "team": WIRE_ANSWERS["team"],
            "sev": {"type": "score", "score": 1.0, "probabilities": [0.1, 0.9]},
            "refund": WIRE_ANSWERS["refund"],
        }
    }
    backend = LayaBackend(agent=FakeAgent(response=response))
    r = backend.decide(REQ)
    assert r.scores["sev"].probabilities == [0.1, 0.9]


def test_laya_without_package_gives_config_error(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "laya" or name.startswith("laya."):
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from decide.backends.laya import LayaBackend

    with pytest.raises(ConfigError, match=r"pydecide\[laya\]"):
        LayaBackend()


# ---- LayaMLXBackend ------------------------------------------------------------------------


def test_laya_mlx_predict_called_with_wire_questions_and_state():
    from decide.backends.laya_mlx import LayaMLXBackend

    agent = FakeAgent()
    backend = LayaMLXBackend(agent=agent)
    r = backend.decide(REQ)

    assert len(agent.calls) == 1
    state, questions = agent.calls[0]
    assert state == REQ.state
    assert questions == to_wire_request(REQ)["questions"]
    _assert_parsed(r, "laya_mlx")
    assert r.meta.model == "aac6fef/laya-multilingual-mlx"


def test_laya_mlx_capabilities_and_batch():
    from decide.backends.laya_mlx import LayaMLXBackend

    backend = LayaMLXBackend(agent=FakeAgent())
    assert backend.capabilities() == backend.capabilities()
    assert backend.capabilities().local is True
    assert backend.capabilities().batch is False

    batching_backend = LayaMLXBackend(agent=FakeAgent(predict_batch=True))
    assert batching_backend.capabilities().batch is True


def test_laya_mlx_without_package_gives_config_error(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "laya_mlx" or name.startswith("laya_mlx."):
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from decide.backends.laya_mlx import LayaMLXBackend

    with pytest.raises(ConfigError, match=r"pydecide\[mlx\]"):
        LayaMLXBackend()


# ---- Live smoke -----------------------------------------------------------------------------

LIVE_REQ = Request(
    "The espresso machine at the office is leaking water from the bottom.",
    {
        "team": Choice(
            "Which team should handle this?",
            {"facilities": "Building, equipment, appliances", "it": "Computers, network, software"},
        )
    },
)


@pytest.mark.live
def test_live_laya():
    from decide.backends.laya import LayaBackend

    backend = LayaBackend()
    r = backend.decide(LIVE_REQ)
    assert set(r.choices["team"].probabilities) == {"facilities", "it"}
    assert abs(sum(r.choices["team"].probabilities.values()) - 1.0) < 1e-6


@pytest.mark.live
def test_live_laya_mlx():
    from decide.backends.laya_mlx import LayaMLXBackend

    backend = LayaMLXBackend()
    r = backend.decide(LIVE_REQ)
    assert set(r.choices["team"].probabilities) == {"facilities", "it"}
    assert abs(sum(r.choices["team"].probabilities.values()) - 1.0) < 1e-6
