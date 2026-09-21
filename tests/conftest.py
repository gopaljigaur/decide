import os
from collections.abc import Callable
from typing import Any

import pytest

from decide.backends.base import BaseBackend, Capabilities
from decide.errors import BackendError
from decide.types import (
    Answer,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Question,
    Request,
    Score,
    ScoreAnswer,
    render_content,
)


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_LIVE") == "1":
        return
    skip = pytest.mark.skip(reason="live test; set RUN_LIVE=1")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


class FakeBackend(BaseBackend):
    """Test double for Backend. Fabricates confident answers, or returns/fails as configured.

    Records every request it is asked to decide (single or batched) in `self.calls`.
    """

    model = None

    def __init__(
        self,
        name: str = "fake",
        answers: dict[str, Answer] | Callable[[Request], dict[str, Answer]] | None = None,
        fail: BackendError | None = None,
        confidence: float = 0.9,
        batch: bool = False,
    ) -> None:
        self.name = name
        self.answers = answers
        self.fail = fail
        self.confidence = confidence
        self.batch = batch
        self.calls: list[Request] = []

    def capabilities(self) -> Capabilities:
        return Capabilities(batch=self.batch)

    def _fabricate(self, question: Question) -> Answer:
        if isinstance(question, Choice):
            candidates = list(question.criteria)
            top, rest = candidates[0], candidates[1:]
            probabilities = {top: self.confidence}
            remainder = (1 - self.confidence) / len(rest) if rest else 0.0
            for cand in rest:
                probabilities[cand] = remainder
            return ChoiceAnswer(top, probabilities)
        if isinstance(question, Score):
            n = len(question.criteria)
            probabilities = [1 / n] * n
            score = sum(i * p for i, p in enumerate(probabilities))
            levels = [render_content(level) for level in question.criteria]
            return ScoreAnswer(score, probabilities, levels)
        if isinstance(question, Noul):
            return NoulAnswer(self.confidence)
        raise TypeError(f"unknown question type: {type(question)!r}")

    def _decide(self, request: Request) -> tuple[dict[str, Answer], Any, str | None]:
        self.calls.append(request)
        if self.fail is not None:
            raise self.fail
        if callable(self.answers):
            return dict(self.answers(request)), None, None
        if self.answers is not None:
            return dict(self.answers), None, None
        return {name: self._fabricate(q) for name, q in request.questions.items()}, None, None


@pytest.fixture
def fake_backend():
    return FakeBackend()
